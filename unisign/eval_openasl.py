#!/usr/bin/env python3
"""Score a checkpoint on the OpenASL test split with the standalone model (interface C7).

    python -m unisign.eval_openasl --ckpt weights/openasl_pose_only_slt.pth --mt5 weights/mt5-base \
        --poses data/openasl_test_pose --labels data/openasl_labels/labels.test \
        --num-beams 4 --max-new-tokens 100 --out results/eval_test_released.json

Mirrors Uni-Sign's `fine_tuning.py --eval` for the pose-only path: batch of clips, frames capped at
--max-length (deterministic uniform subsample here; theirs is random), beam 4, max_new_tokens 100,
sacrebleu 13a + rouge-l via their SLRT_metrics.translation_performance (vendored copy).
Writes refs/preds and the metric dict. --limit N scores the first N clips for a quick check.
"""
import argparse
import gzip
import json
import os
import pickle
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.pose_to_unisign import collate, load_pkl, to_model_inputs  # noqa: E402
from unisign.model import load_model  # noqa: E402
from unisign.metrics import translation_performance  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mt5", required=True)
    ap.add_argument("--poses", required=True, help="dir of <clip>.pkl")
    ap.add_argument("--labels", required=True, help="Uni-Sign labels.test (gzip pickle)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--w8-runtime", default="dequant", choices=["dequant", "int8"], help="for W8 checkpoints (unisign.quant)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--fps", type=float, default=None, help="emulate this camera rate on every clip (source --src-fps)")
    ap.add_argument("--src-fps", type=float, default=24.0)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = load_model(args.ckpt, args.mt5, device=args.device, dtype=dtype, w8_runtime=args.w8_runtime)
    fps_ratio = (args.fps / args.src_fps) if args.fps else 1.0
    labels = pickle.load(gzip.open(args.labels, "rb"))
    names = list(labels)[: args.limit] if args.limit else list(labels)
    refs, preds, missing = [], [], []
    t0 = time.perf_counter()
    for b in range(0, len(names), args.batch_size):
        batch, batch_names = [], []
        for n in names[b:b + args.batch_size]:
            p = os.path.join(args.poses, n.replace(".mp4", ".pkl"))
            if not os.path.exists(p):
                missing.append(n)
                continue
            kps, scs, _ = load_pkl(p)
            inputs, _ = to_model_inputs(kps, scs, args.max_length, fps_ratio=fps_ratio)
            batch.append(inputs); batch_names.append(n)
        if not batch:
            continue
        src = collate(batch, batch_names)
        with torch.no_grad():
            dev = next(model.parameters()).device
            src = {k: (v.to(dev).float() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in src.items()}
            inputs_embeds, attention_mask = model.build_encoder_inputs(src)
            out = model.mt5_model.generate(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                                           max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
        texts = model.mt5_tokenizer.batch_decode(out, skip_special_tokens=True)
        for n, t in zip(batch_names, texts):
            refs.append(labels[n]["text"]); preds.append(t)
        done = len(preds)
        if (b // args.batch_size) % 10 == 0:
            el = time.perf_counter() - t0
            print(f"[eval] {done}/{len(names)}  {el/ done:.2f} s/clip  eta {(len(names)-done)*el/done/60:.1f} min", flush=True)
    bleu, rouge = translation_performance(refs, preds)
    res = {"n": len(preds), "missing": len(missing), "bleu": bleu, "rouge_l": rouge,
           "wall_s": time.perf_counter() - t0, "config": vars(args)}
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({**res, "refs": refs, "preds": preds}, open(args.out, "w"), indent=1)
    print("[eval] wrote", args.out)


if __name__ == "__main__":
    main()
