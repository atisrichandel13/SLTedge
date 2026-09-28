#!/usr/bin/env python3
"""Drop frames from Uni-Sign pkls to simulate a lower capture rate (the accuracy half of P8).

    python common/subsample_pkl.py --in results/pkl_30clip_rtmw_fp16_sqnorm \
        --keep-fps 24 --out results/pkl_30clip_rtmw_fp16_sqnorm_fps24

Why this is exact rather than an approximation. RTMW/RTMPose extract each frame independently -- there
is no temporal model in the pose stage -- so a frame kept at a reduced rate receives exactly the
keypoints it would have received at full rate. Subsampling the keypoints is therefore identical to
dropping frames before extraction, and it costs no board time.

That equivalence holds for ACCURACY only. Energy and latency at a reduced rate must come from a real
run (`03_infer_frames.py --keep-fps`), because what changes there is how many frames the board actually
processes per second of video.

Uses the same uniform selection as `list_frames(..., keep_fps=...)` so the two paths agree on which
frames a given rate keeps.
"""
import argparse
import glob
import os
import pickle

import numpy as np


def keep_idx(n_frames, keep_fps, src_fps):
    n = max(1, int(round(n_frames * float(keep_fps) / float(src_fps))))
    idx = np.round(np.linspace(0, n_frames - 1, n)).astype(int)
    return sorted(set(idx.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-fps", type=float, required=True)
    ap.add_argument("--src-fps", type=float, default=29.97)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = 0
    kept_total = in_total = 0
    for p in sorted(glob.glob(os.path.join(args.inp, "*.pkl"))):
        d = pickle.load(open(p, "rb"))
        T = len(d["scores"])
        idx = keep_idx(T, args.keep_fps, args.src_fps)
        d = {**d,
             "keypoints": [d["keypoints"][i] for i in idx],
             "scores": [d["scores"][i] for i in idx],
             "subsampled": {"keep_fps": args.keep_fps, "src_fps": args.src_fps,
                            "frames_in": T, "frames_kept": len(idx)}}
        pickle.dump(d, open(os.path.join(args.out, os.path.basename(p)), "wb"))
        in_total += T
        kept_total += len(idx)
        n += 1
    print(f"[subsample] {n} clips at {args.keep_fps} fps (from {args.src_fps}): "
          f"{in_total} -> {kept_total} frames ({100 * kept_total / max(1, in_total):.1f} %) -> {args.out}")


if __name__ == "__main__":
    main()
