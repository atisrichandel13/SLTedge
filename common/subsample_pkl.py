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
import json
import os
import pickle

import numpy as np


def keep_idx(n_frames, keep_fps, src_fps):
    """Keep round(duration x keep_fps) frames, where duration = n_frames / src_fps.

    src_fps MUST be the clip's own rate. OpenASL is not one frame rate: of our 931 clips 76.5 % are
    ~30 fps, 22.2 % are ~24 and 7 are 59.94. Assuming 29.97 for everything meant a 24 fps clip labelled
    "16 fps" was really subsampled to 16 x 24/29.97 = 12.8 fps, so the row measured a mixture of rates.
    That invalidated our first accuracy-vs-rate curve (RESULTS.md 2.9B) and the LM track's first
    attempt independently (their L7.1 -> L7.2).
    """
    if keep_fps >= src_fps:
        return list(range(n_frames))
    n = max(1, int(round(n_frames * float(keep_fps) / float(src_fps))))
    idx = np.round(np.linspace(0, n_frames - 1, n)).astype(int)
    return sorted(set(idx.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep-fps", type=float, required=True)
    ap.add_argument("--src-fps", type=float, default=None,
                    help="override the per-clip rate; normally leave unset so each clip's own rate is "
                         "read from its meta.json (frames / duration). A single assumed rate is what "
                         "broke the first accuracy-vs-rate curve")
    ap.add_argument("--clips", default="data/clips",
                    help="dir of <vid>/meta.json, used to read each clip's true frame rate")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = 0
    kept_total = in_total = 0
    n_assumed = 0
    for p in sorted(glob.glob(os.path.join(args.inp, "*.pkl"))):
        vid = os.path.basename(p)[:-4]
        d = pickle.load(open(p, "rb"))
        T = len(d["scores"])
        src = args.src_fps
        if src is None:
            mp = os.path.join(args.clips, vid, "meta.json")
            if os.path.exists(mp):
                m = json.load(open(mp))
                # frames / duration is the rate that actually applies, not the container's fps field
                src = (m["n_frames"] / m["duration_s"]) if m.get("duration_s") else float(m["fps"])
            else:
                src, n_assumed = 29.97, n_assumed + 1
        idx = keep_idx(T, args.keep_fps, src)
        d = {**d,
             "keypoints": [d["keypoints"][i] for i in idx],
             "scores": [d["scores"][i] for i in idx],
             "subsampled": {"keep_fps": args.keep_fps, "src_fps": round(float(src), 3),
                            "frames_in": T, "frames_kept": len(idx)}}
        pickle.dump(d, open(os.path.join(args.out, os.path.basename(p)), "wb"))
        in_total += T
        kept_total += len(idx)
        n += 1
    if n_assumed:
        print(f"[subsample] WARNING {n_assumed} clips had no meta.json; assumed 29.97 fps for them")
    print(f"[subsample] {n} clips at {args.keep_fps} fps (per-clip source rate): "
          f"{in_total} -> {kept_total} frames ({100 * kept_total / max(1, in_total):.1f} %) -> {args.out}")


if __name__ == "__main__":
    main()
