#!/usr/bin/env python3
"""Accuracy surface for the frontier plot: decoder strategy x frame rate, pruned FP32, all 976 clips.

Prints BLEU-4 / ROUGE-L per cell plus the paired bootstrap delta of each cell against the
beam-4 / source-rate reference, so the accuracy axis is ready to pair with the board's joules.

    python -m unisign.grid_table [-n 1000]
"""
import argparse, json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unisign.external_metrics import sacrebleu

R = os.path.join(ROOT, "results") + "/"
# (beams, fps) -> eval json.  None fps = unthinned source rate (30/24 mixed, see L7.2).
CELLS = {
    (4, None): "eval_test_pruned_mac.json",
    (4, 24):   "eval_test_pruned_b4_fps24.json",
    (4, 16):   "eval_test_pruned_truefps16.json",
    (2, None): "eval_test_pruned_beam2_mac.json",
    (2, 24):   "eval_test_pruned_b2_fps24.json",
    (2, 16):   "eval_test_pruned_b2_fps16.json",
    (1, None): "eval_test_pruned_greedy_mac.json",
    (1, 24):   "eval_test_pruned_b1_fps24.json",
    (1, 16):   "eval_test_pruned_b1_fps16.json",
}
REF = (4, None)
NAME = {1: "greedy", 2: "beam 2", 4: "beam 4"}


def bleu4(refs, hyps):
    return sacrebleu.corpus_bleu(sys_stream=hyps, ref_streams=[refs], tokenize="13a").scores[3]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="write the surface + CIs here as JSON, for plot_frontier.py to consume")
    args = ap.parse_args()

    have = {k: json.load(open(R + f)) for k, f in CELLS.items() if os.path.exists(R + f)}
    missing = [k for k in CELLS if k not in have]
    if REF not in have:
        sys.exit("reference cell (beam 4, source) missing")
    refs = have[REF]["refs"]
    for k, d in have.items():
        assert d["refs"] == refs, f"refs differ in {CELLS[k]}"
    N = len(refs)

    # One resample draw scores EVERY cell, so all deltas share identical clip draws and the
    # cells are comparable with each other, not just each against the reference. Also 2x cheaper
    # than recomputing the reference per cell.
    rng = np.random.default_rng(args.seed)
    keys = list(have)
    draws = {k: [] for k in keys}
    for _ in range(args.n):
        idx = rng.integers(0, N, N)
        r = [refs[i] for i in idx]
        for k in keys:
            draws[k].append(bleu4(r, [have[k]["preds"][i] for i in idx]))
    full = {k: bleu4(refs, have[k]["preds"]) for k in keys}
    stats = {}
    for k in keys:
        if k == REF:
            stats[k] = (full[k], have[k]["rouge_l"], None); continue
        d = np.array(draws[k]) - np.array(draws[REF])
        lo, hi = np.percentile(d, [2.5, 97.5])
        stats[k] = (full[k], have[k]["rouge_l"], (full[k] - full[REF], lo, hi))

    print(f"\nAccuracy surface, pruned FP32, {N} test clips, cap 64. "
          f"Delta = paired bootstrap vs beam 4 @ source rate ({args.n} resamples).\n")
    print(f"{'decoder':9s} {'fps':>7s} {'BLEU-4':>7s} {'ROUGE-L':>8s}   delta vs ref [95% CI]")
    for b in (4, 2, 1):
        for f in (None, 24, 16):
            k = (b, f)
            lab = "source" if f is None else str(f)
            if k not in stats:
                print(f"{NAME[b]:9s} {lab:>7s} {'pending':>7s}"); continue
            bl, rg, d = stats[k]
            dd = "reference" if d is None else f"{d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]"
            print(f"{NAME[b]:9s} {lab:>7s} {bl:7.2f} {rg:8.2f}   {dd}")
    if missing:
        print("\nmissing cells:", ", ".join(f"{NAME[b]}@{f or 'source'}" for b, f in missing))

    if args.out:
        # Keyed "beams,fps" because JSON has no tuple keys; fps "source" = unthinned.
        out = {"n_clips": N, "n_boot": args.n, "seed": args.seed,
               "ref": "4,source", "cap": 64,
               "shared_draws": True,
               "note": "delta/ci are paired-bootstrap BLEU-4 vs the reference cell; one resample "
                       "draw scores every cell, so cells are comparable with each other too.",
               "cells": {}}
        for (b, f), (bl, rg, d) in stats.items():
            out["cells"][f"{b},{f or 'source'}"] = {
                "beams": b, "fps": f or "source", "bleu4": bl, "rouge_l": rg,
                "source_json": CELLS[(b, f)],
                "delta_bleu4": None if d is None else d[0],
                "ci": None if d is None else [d[1], d[2]]}
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
