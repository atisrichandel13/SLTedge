#!/usr/bin/env python3
"""Adaptation-training harness (C8 / L10 / L11 in WORKSPLIT.md, Block 4.2 in PROJECT-GUIDE.md).

mT5 stays frozen. Trainable: the pose stack (proj_linear, gcn_modules, fusion_gcn_modules),
`part_para` and `pose_proj` (~5.4 M params). The point is to re-fit the pose front-end to an input
shift (lower frame rate, dropped face group, different keypoint model) without touching the LM.

Recipe follows the authors' fine_tuning.py / utils.py defaults for stage-3 SLT training, minus
deepspeed: AdamW (eps 1e-9, weight decay 1e-4), cosine schedule with optional warm-up, gradient
clipping 1.0, label smoothing 0.2, targets truncated to 50 tokens, random frame subsample above
--max-length at train time (deterministic uniform at eval). Authors' lr is 3e-4 for the full model;
this defaults to 1e-4 because we start from a converged checkpoint and only adapt the front-end.

    # Mac CPU smoke (300 clips, ~10 min/epoch)
    python -m unisign.train_adapt --ckpt weights/openasl_pose_only_slt_pruned.pth --mt5 weights/mt5-base \
        --poses data/openasl_train_pose_smoke --labels data/openasl_labels/labels.train \
        --out-dir runs/smoke --epochs 1 --batch-size 4 \
        --eval-poses data/openasl_test_pose --eval-labels data/openasl_labels/labels.test --eval-limit 64

    # Colab, 16 fps adaptation (L11): same, plus --fps 16 --amp --batch-size 16, and pass --fps 16 to
    # unisign.eval_openasl on the adapted checkpoint.

Outputs in --out-dir: last.pt (trainable weights + optimizer + scheduler + epoch, for --resume),
adapted_full.pth (full state dict + keep_ids, loadable by unisign.model.load_model / eval_openasl),
log.jsonl (one line per logged step / epoch / eval).
"""
import argparse
import gzip
import json
import math
import os
import pickle
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.pose_to_unisign import collate, load_pkl, to_model_inputs  # noqa: E402
from unisign.metrics import translation_performance  # noqa: E402
from unisign.model import load_model  # noqa: E402

TRAINABLE_PREFIXES = ("proj_linear.", "gcn_modules.", "fusion_gcn_modules.", "part_para", "pose_proj.")


# ----------------------------------------------------------------------------- data
class PoseTextDataset(Dataset):
    def __init__(self, poses_dir, labels, names, max_length, random_subsample, fps_ratio):
        self.poses_dir, self.labels, self.names = poses_dir, labels, names
        self.max_length, self.random_subsample, self.fps_ratio = max_length, random_subsample, fps_ratio

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        kps, scs, _ = load_pkl(os.path.join(self.poses_dir, n.replace(".mp4", ".pkl")))
        inputs, _ = to_model_inputs(kps, scs, self.max_length, self.random_subsample, self.fps_ratio)
        return n, inputs, self.labels[n]["text"]


def collate_batch(batch):  # module-level so DataLoader workers can pickle it (macOS spawn)
    names = [b[0] for b in batch]
    src = collate([b[1] for b in batch], names)
    return src, [b[2] for b in batch]


def available_names(labels, poses_dir, limit=None, seed=0, shuffle=False):
    names = [n for n in labels if os.path.exists(os.path.join(poses_dir, n.replace(".mp4", ".pkl")))]
    if shuffle:
        random.Random(seed).shuffle(names)
    return names[:limit] if limit else names


# ----------------------------------------------------------------------------- model helpers
def set_trainable(model, freeze_bn=False):
    n_train = 0
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith(TRAINABLE_PREFIXES)
        n_train += p.numel() if p.requires_grad else 0
    if freeze_bn:
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
    return n_train


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith(TRAINABLE_PREFIXES)}


def to_device(src, dev):
    return {k: (v.to(dev).float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in src.items()}


def compute_loss(model, src, texts, label_smoothing, max_target_len=50):
    """Authors' forward(): prefix+pose embeds -> mT5 with labels, pad -> -100, CE with label smoothing."""
    dev = next(model.parameters()).device
    inputs_embeds, attention_mask = model.build_encoder_inputs(src)
    tgt = model.mt5_tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_target_len)
    labels = tgt["input_ids"].to(dev)
    labels[labels == model.mt5_tokenizer.pad_token_id] = -100
    out = model.mt5_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, return_dict=True)
    logits = out.logits.float()
    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1),
                                       ignore_index=-100, label_smoothing=label_smoothing)
    return loss, int((labels != -100).sum())


@torch.no_grad()
def evaluate(model, poses_dir, labels, names, max_length, fps_ratio, batch_size, num_beams, max_new_tokens):
    model.eval()
    dev = next(model.parameters()).device
    refs, preds = [], []
    for b in range(0, len(names), batch_size):
        batch, bn = [], []
        for n in names[b:b + batch_size]:
            kps, scs, _ = load_pkl(os.path.join(poses_dir, n.replace(".mp4", ".pkl")))
            batch.append(to_model_inputs(kps, scs, max_length, False, fps_ratio)[0]); bn.append(n)
        src = to_device(collate(batch, bn), dev)
        emb, mask = model.build_encoder_inputs(src)
        out = model.mt5_model.generate(inputs_embeds=emb, attention_mask=mask, max_new_tokens=max_new_tokens, num_beams=num_beams)
        preds += model.mt5_tokenizer.batch_decode(out, skip_special_tokens=True)
        refs += [labels[n]["text"] for n in bn]
    bleu, rouge = translation_performance(refs, preds)
    return {"n": len(preds), "bleu": bleu, "rouge_l": rouge, "sample": list(zip(refs[:3], preds[:3]))}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="starting checkpoint (released or pruned)")
    ap.add_argument("--mt5", required=True)
    ap.add_argument("--poses", required=True, help="dir of train <clip>.pkl")
    ap.add_argument("--labels", required=True, help="labels.train (gzip pickle)")
    ap.add_argument("--limit", type=int, default=None, help="use the first N available train clips")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA only; authors train in bf16)")
    ap.add_argument("--seed", type=int, default=42)
    # input shift
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--fps", type=float, default=None, help="emulate this camera rate on every clip")
    ap.add_argument("--src-fps", type=float, default=24.0)
    # recipe
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=float, default=0.0)
    ap.add_argument("--clip-grad", type=float, default=1.0)
    ap.add_argument("--label-smoothing", type=float, default=0.2)
    ap.add_argument("--freeze-bn", action="store_true", help="keep BatchNorm running stats fixed (small-batch CPU runs)")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=10)
    # eval
    ap.add_argument("--eval-poses", default=None)
    ap.add_argument("--eval-labels", default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--eval-before", action="store_true", help="also score the un-adapted model first")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    log_f = open(os.path.join(args.out_dir, "log.jsonl"), "a")

    def log(d):
        d["time"] = round(time.time(), 1); log_f.write(json.dumps(d) + "\n"); log_f.flush()

    fps_ratio = (args.fps / args.src_fps) if args.fps else 1.0
    model = load_model(args.ckpt, args.mt5, device=args.device)
    _src = torch.load(args.ckpt, map_location="cpu"); _src = _src.get("model", _src)
    src_dtypes = {k: v.dtype for k, v in _src.items() if torch.is_tensor(v)}; del _src  # keep file size of the source
    n_train = set_trainable(model, args.freeze_bn)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable {n_train/1e6:.2f} M of {n_total/1e6:.1f} M params; fps_ratio {fps_ratio:.3f}; max_length {args.max_length}")

    labels = pickle.load(gzip.open(args.labels, "rb"))
    names = available_names(labels, args.poses, args.limit)
    if not names:
        raise SystemExit(f"no train clips found in {args.poses}")
    ds = PoseTextDataset(args.poses, labels, names, args.max_length, True, fps_ratio)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                    collate_fn=collate_batch, drop_last=False, persistent_workers=args.num_workers > 0)
    print(f"[train] {len(names)} clips, {len(dl)} batches/epoch, batch {args.batch_size} x accum {args.accum}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay, eps=1e-9)
    steps_per_epoch = math.ceil(len(dl) / args.accum)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warm = int(args.warmup_epochs * steps_per_epoch)

    def lr_lambda(step):
        if step < warm:
            return (step + 1) / max(1, warm)
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    start_epoch, step = 0, 0
    last_path = os.path.join(args.out_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location="cpu")
        model.load_state_dict(ck["trainable"], strict=False)
        opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        start_epoch, step = ck["epoch"] + 1, ck["step"]
        print(f"[train] resumed from {last_path}: epoch {start_epoch}, step {step}")

    eval_names = None
    if args.eval_poses and args.eval_labels:
        eval_labels = pickle.load(gzip.open(args.eval_labels, "rb"))
        eval_names = available_names(eval_labels, args.eval_poses, args.eval_limit)
        if args.eval_before and start_epoch == 0:
            r = evaluate(model, args.eval_poses, eval_labels, eval_names, args.max_length, fps_ratio,
                         args.eval_batch_size, args.num_beams, args.max_new_tokens)
            print(f"[eval] before: BLEU-4 {r['bleu']['bleu4']:.2f} on {r['n']} clips"); log({"eval": "before", **r})

    use_amp = args.amp and args.device.startswith("cuda")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if args.freeze_bn:
            for m in model.modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm):
                    m.eval()
        model.mt5_model.eval()  # frozen LM: no dropout inside mT5
        t0, run_loss, run_tok, n_seen = time.perf_counter(), 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, (src, texts) in enumerate(dl):
            src = to_device(src, args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss, ntok = compute_loss(model, src, texts, args.label_smoothing)
            (loss / args.accum).backward()
            run_loss += loss.item() * ntok; run_tok += ntok; n_seen += len(texts)
            if (i + 1) % args.accum == 0 or (i + 1) == len(dl):
                if args.clip_grad:
                    torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
                if step % args.log_every == 0:
                    el = time.perf_counter() - t0
                    msg = {"epoch": epoch, "step": step, "loss": round(run_loss / max(1, run_tok), 4),
                           "lr": sched.get_last_lr()[0], "clips": n_seen, "s_per_clip": round(el / n_seen, 3)}
                    print(f"[train] ep {epoch} step {step} loss {msg['loss']:.4f} lr {msg['lr']:.2e} "
                          f"{n_seen}/{len(names)} clips  {el/n_seen:.2f} s/clip  eta {(len(names)-n_seen)*el/n_seen/60:.1f} min", flush=True)
                    log(msg)
        ep_loss = run_loss / max(1, run_tok)
        print(f"[train] epoch {epoch} done: loss {ep_loss:.4f}, {time.perf_counter()-t0:.0f} s")
        log({"epoch_done": epoch, "loss": round(ep_loss, 4), "wall_s": round(time.perf_counter() - t0)})
        torch.save({"trainable": trainable_state(model), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "step": step, "args": vars(args)}, last_path)
        if eval_names:
            r = evaluate(model, args.eval_poses, eval_labels, eval_names, args.max_length, fps_ratio,
                         args.eval_batch_size, args.num_beams, args.max_new_tokens)
            print(f"[eval] epoch {epoch}: BLEU-4 {r['bleu']['bleu4']:.2f} ROUGE-L {r['rouge_l']:.2f} on {r['n']} clips")
            for ref, pred in r["sample"]:
                print(f"        ref : {ref}\n        pred: {pred}")
            log({"eval": epoch, **r})

    full = {"model": {k: v.detach().cpu().to(src_dtypes.get(k, v.dtype)) if v.is_floating_point() else v.detach().cpu()
                      for k, v in model.state_dict().items()}}
    if getattr(model, "keep_ids", None) is not None:
        full["keep_ids"] = model.keep_ids.tolist()
    full_path = os.path.join(args.out_dir, "adapted_full.pth")
    torch.save(full, full_path)
    print(f"[train] wrote {last_path} and {full_path} ({os.path.getsize(full_path)/1e6:.0f} MB)")


if __name__ == "__main__":
    main()
