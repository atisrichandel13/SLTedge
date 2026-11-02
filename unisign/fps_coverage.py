#!/usr/bin/env python3
"""How much of a split a --fps condition actually changes, before any GPU time is spent on it.

    python -m unisign.fps_coverage --poses data/openasl_dev_pose --rates 16,24

Why this exists: "24 fps" is not a uniform treatment on OpenASL, for two reasons that pull in the
same direction and are invisible in the eval config.

1. OpenASL is not one frame rate. fps_ratio_for_clip derives each clip's own rate from
   frames / duration, so a clip already at or below the target is passed through untouched.
2. max_length caps the sequence AFTER thinning. A clip longer than the cap at source is already
   being subsampled to `max_length` frames; thinning it first changes WHICH frames are picked but
   not HOW MANY reach the model, so most of the nominal reduction never reaches the encoder.

The number that matters for interpreting an adaptation or frame-rate result is therefore not the
target rate but the change in frames actually fed to the model. Report that next to any claim about
"24 fps", or the effect size is being compared against a treatment nobody measured.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.pose_to_unisign import clip_duration_s, fps_ratio_for_clip, load_pkl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses", required=True, help="dir of <clip>.pkl")
    ap.add_argument("--rates", default="16,24", help="comma-separated target rates")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--src-fps", type=float, default=30.0, help="fallback for names without timestamps")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    names = sorted(f[:-4] for f in os.listdir(a.poses) if f.endswith(".pkl"))
    if not names:
        sys.exit(f"no .pkl in {a.poses}")
    rates = [float(r) for r in a.rates.split(",")]

    T, src = [], []
    for n in names:
        _, scs, _ = load_pkl(os.path.join(a.poses, n + ".pkl"))
        d = clip_duration_s(n)
        T.append(len(scs))
        src.append(len(scs) / d if d else a.src_fps)
    T, src = np.array(T), np.array(src)
    base = np.minimum(a.max_length, T)

    out = {"poses": a.poses, "n_clips": len(names), "max_length": a.max_length,
           "frames_at_source": int(base.sum()), "over_cap_at_source": int((T > a.max_length).sum()),
           "rates": {}}
    print(f"{len(names)} clips, max_length {a.max_length}, "
          f"{(T > a.max_length).sum()} already over the cap at source\n")

    for tgt in rates:
        ratio = np.array([fps_ratio_for_clip(n, t, tgt, a.src_fps) for n, t in zip(names, T)])
        thinned = np.where(ratio < 1.0, np.maximum(1, np.round(T * ratio)), T).astype(int)
        eff = np.minimum(a.max_length, thinned)
        untouched = int((ratio >= 1.0).sum())
        same_n = int((eff == base).sum())
        rec = {"clips_not_thinned": untouched,
               "clips_same_frame_count_to_model": same_n,
               "frames_to_model": int(eff.sum()),
               "frames_pct_change": float(100 * (eff.sum() - base.sum()) / base.sum()),
               "median_frames": [int(np.median(base)), int(np.median(eff))]}
        out["rates"][str(tgt)] = rec
        print(f"--- target {tgt:g} fps ---")
        print(f"  native rate already <= target, clip passed through : "
              f"{untouched:4d}/{len(names)}  {100*untouched/len(names):5.1f}%")
        print(f"  SAME NUMBER of frames reaches the model (cap absorbs): "
              f"{same_n:4d}/{len(names)}  {100*same_n/len(names):5.1f}%")
        print("     (same count, not the same frames -- thinning still shifts which are picked)")
        print(f"  frames to model : {base.sum()} -> {eff.sum()}  "
              f"({rec['frames_pct_change']:+.1f}%)")
        print(f"  median frames   : {np.median(base):.0f} -> {np.median(eff):.0f}\n")

    bands = [(0, 20, "<20"), (20, 25.5, "20-25.5"), (25.5, 28, "25.5-28"),
             (28, 32, "28-32"), (32, 1e9, ">32")]
    out["native_rate_bands"] = {}
    print("native rate distribution")
    for lo, hi, lab in bands:
        k = int(((src >= lo) & (src < hi)).sum())
        out["native_rate_bands"][lab] = k
        print(f"  {lab:9s} {k:4d}  {100*k/len(names):5.1f}%")

    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
