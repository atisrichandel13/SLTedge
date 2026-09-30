#!/usr/bin/env python3
"""Check that our reimplementation of OpenASL's crop recipe matches what they actually shipped.

    python data/verify_openasl_crop.py --clips data/clips --poses data/openasl_5clip_pose

`data/openasl_fetch.py --crop-style openasl` reproduces `prep/crop_video.py`'s `crop_resize()`: square
the bbox by expanding the shorter side, black-pad whatever falls outside the frame, resize to 224. We
cannot diff the frames directly (they ship no video), but we can falsify the geometry against their
released keypoints, which are normalised over that square:

  every keypoint the authors are CONFIDENT about must land inside the part of the square that holds
  real pixels, because the rest is black and nothing is detectable there.

That is a real test: it fails loudly if the bbox is mis-ordered, if the expansion goes the wrong way,
if padding is applied on the wrong side, or if the square side is off. Judge it on confident keypoints
only (score >= THR) -- argmax on a black region lands anywhere, so unconfident joints legitimately sit
outside the real-pixel box and say nothing about the geometry.

Prints the black-padding fraction too, which is the point of the whole exercise: their frames are
15-50 % black, so in their normalised space the signer never reaches the frame edge, while our
native-resolution crop fills it. That is the distribution shift the pose->BLEU gap may come from.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from openasl_fetch import openasl_square_crop  # noqa: E402

THR = 0.3
SLACK = 0.05  # keypoints may sit slightly outside the real-pixel box: simcc decodes over a bbox
              # padded 1.25x beyond the crop, so an edge joint overshoots by a bin or two


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips", help="dir of <vid>/meta.json (native-crop fetch)")
    ap.add_argument("--poses", default="data/openasl_5clip_pose", help="authors' released <vid>.pkl")
    ap.add_argument("--slack", type=float, default=SLACK)
    args = ap.parse_args()

    metas = sorted(glob.glob(os.path.join(args.clips, "*", "meta.json")))
    if not metas:
        raise SystemExit(f"no <vid>/meta.json under {args.clips}")
    print(f"{'clip':13s} {'black':>6s} | real-pixel region of the square | authors' confident kpts | ok")
    bad = []
    for mp in metas:
        m = json.load(open(mp))
        vid = m["vid"]
        pkl = os.path.join(args.poses, f"{vid}.pkl")
        if not os.path.exists(pkl):
            print(f"{vid[:12]}  -- no released pkl, skipped")
            continue
        W, H = m["source_wh"]
        _, g = openasl_square_crop(m["bbox_norm"], W, H)
        sx, sy, S, _ = g["square_xywh"]
        ix, iy, iw, ih = g["inside_frame_xywh"]
        box = ((ix - sx) / S, (ix + iw - sx) / S, (iy - sy) / S, (iy + ih - sy) / S)

        d = pickle.load(open(pkl, "rb"))
        k = np.concatenate([np.asarray(x).reshape(-1, 133, 2) for x in d["keypoints"]])
        s = np.concatenate([np.asarray(x).reshape(-1, 133) for x in d["scores"]])
        c = s >= THR
        obs = (k[..., 0][c].min(), k[..., 0][c].max(), k[..., 1][c].min(), k[..., 1][c].max())
        ok = (obs[0] > box[0] - args.slack and obs[1] < box[1] + args.slack and
              obs[2] > box[2] - args.slack and obs[3] < box[3] + args.slack)
        if not ok:
            bad.append(vid)
        print(f"{vid[:12]} {100 * g['black_pad_px'] / S ** 2:5.1f}% | "
              f"x {box[0]:.3f}-{box[1]:.3f} y {box[2]:.3f}-{box[3]:.3f} | "
              f"x {obs[0]:.3f}-{obs[1]:.3f} y {obs[2]:.3f}-{obs[3]:.3f} | "
              f"{'YES' if ok else 'NO'}")
    if bad:
        raise SystemExit(f"\ngeometry does NOT reproduce the released poses for: {bad}\n"
                         f"our --crop-style openasl is wrong; do not use it for a comparison")
    print("\nall clips: the authors' confident keypoints fall inside the real-pixel region, so our\n"
          "--crop-style openasl geometry reproduces theirs.")


if __name__ == "__main__":
    main()
