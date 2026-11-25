#!/usr/bin/env python3
"""Rebuild the 570 MB adapted_full.pth from a training run's last.pt, with no training data.

    python3 -m unisign.rebuild_adapted --base weights/openasl_pose_only_slt_pruned_traindev.pth \
        --mt5 weights/mt5-base --last weights/block6/fps24_s42/last.pt \
        --out weights/block6/fps24_s42_adapted_full.pth

Why this exists. last.pt holds only the trainable tensors (the pose stack, pose_proj and part_para,
343 tensors) plus optimizer state; the full model is the base checkpoint with those tensors swapped
in. unisign.train_adapt --resume can also write it, but it builds the training dataset first and so
needs the 20,000 train pose files. This needs only the base checkpoint and mT5.

It writes exactly what train_adapt's --full-out writes: the full state dict cast back to the base
file's dtypes, plus keep_ids. Checked bit-exact against runs/c8_adapt16/adapted_full.pth, rebuilt
from runs/c8_adapt16/last.pt (--verify-against).

The base MUST be the one the run trained from -- last.pt records it in args["ckpt"], and this script
refuses a mismatch by file name. The two pruned lineages have different vocabularies (26,078 vs
26,025 tokens, RESULTS.md L26), so the wrong base gives a model that loads and translates badly.
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unisign.model import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the base checkpoint the run trained from")
    ap.add_argument("--mt5", required=True)
    ap.add_argument("--last", required=True, help="the run's last.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify-against", default=None, help="an adapted_full.pth to compare bit for bit")
    args = ap.parse_args()

    ck = torch.load(args.last, map_location="cpu", weights_only=False)
    run_base = os.path.basename(str(ck.get("args", {}).get("ckpt", "")))
    if run_base and os.path.basename(args.base) not in (run_base, "openasl_pose_only_slt_" + run_base):
        raise SystemExit(f"--base {os.path.basename(args.base)!r} is not the base this run trained from "
                         f"({ck['args']['ckpt']!r}). Use the matching checkpoint.")
    if not ck.get("epoch_done", True):
        raise SystemExit(f"{args.last} is a mid-epoch checkpoint (batch {ck.get('batch_in_epoch')}); "
                         f"the run did not finish")

    model = load_model(args.base, args.mt5, device="cpu")
    src = torch.load(args.base, map_location="cpu"); src = src.get("model", src)
    src_dtypes = {k: v.dtype for k, v in src.items() if torch.is_tensor(v)}; del src
    missing, unexpected = model.load_state_dict(ck["trainable"], strict=False)
    if unexpected:
        raise SystemExit(f"last.pt has tensors the model does not: {unexpected[:5]}")
    for k, v in model.state_dict().items():
        if v.is_floating_point() and not torch.isfinite(v).all():
            raise SystemExit(f"non-finite parameter {k}; refusing to write")

    full = {"model": {k: v.detach().cpu().to(src_dtypes.get(k, v.dtype)) if v.is_floating_point() else v.detach().cpu()
                      for k, v in model.state_dict().items()}}
    if getattr(model, "keep_ids", None) is not None:
        full["keep_ids"] = model.keep_ids.tolist()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(full, args.out)
    a = ck.get("args", {})
    print(f"[rebuild] {len(ck['trainable'])} trainable tensors from {args.last} "
          f"(fps {a.get('fps') or 'source'}, seed {a.get('seed')}) onto {args.base}")
    print(f"[rebuild] wrote {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)")

    if args.verify_against:
        ref = torch.load(args.verify_against, map_location="cpu", weights_only=False)
        a_, b_ = full["model"], ref.get("model", ref)
        assert set(a_) == set(b_), "key sets differ"
        bad = [k for k in a_ if not torch.equal(a_[k], b_[k])]
        same_ids = full.get("keep_ids") == ref.get("keep_ids")
        print(f"[rebuild] vs {args.verify_against}: {len(a_) - len(bad)}/{len(a_)} tensors bit-identical, "
              f"keep_ids {'identical' if same_ids else 'DIFFER'}")
        if bad or not same_ids:
            raise SystemExit(f"MISMATCH: {bad[:5]}")


if __name__ == "__main__":
    main()
