#!/usr/bin/env python3
"""Re-express our keypoints in OpenASL's normalised frame, without re-extracting any poses.

    python common/renorm_to_openasl.py --in results/pkl_30clip_rtmw_fp32 \
        --clips data/clips --out results/pkl_30clip_rtmw_fp32_sqnorm

Why this exists. Our pkls normalise keypoints over the bbox crop we extracted (`crop_xywh`). The
authors normalise over a SQUARE: bbox expanded on its short side, black-padded where it leaves the
frame, then resized to 224. Signer boxes are tall and narrow, so their square side is much wider than
our crop width and the signer occupies a smaller fraction of their frame. Measured over 30 clips, our
coordinates are 1.679 +- 0.083x larger than theirs, and `square_side / our_crop_width` predicts that
per clip with correlation 0.968 (mean abs error 0.028).

This matters because Uni-Sign's frozen encoder consumes the 9 body joints in ABSOLUTE normalised
coordinates (hands are wrist-relative and the face is nose-tip-relative, so a global scale largely
cancels for them). Feeding absolute body coordinates at 1.68x the scale the encoder was trained on is
a domain shift, and it is the largest measured difference between our poses and theirs.

The fix is exact arithmetic, not a re-run: our normalised (u, v) -> source pixels -> their square.
    px = crop_x + u * crop_w                  py = crop_y + v * crop_h
    u' = (px - square_x) / side               v' = (py - square_y) / side
Scores are untouched. Everything else about the poses -- which extractor, which precision, which
frames -- is held fixed, so a BLEU change after this is attributable to the frame alone.
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
from openasl_fetch import openasl_square_crop  # noqa: E402


def renorm(kps, crop_xywh, square_xywh):
    cx, cy, cw, ch = crop_xywh
    sx, sy, side = square_xywh[0], square_xywh[1], square_xywh[2]
    out = []
    for a in kps:
        a = np.asarray(a, dtype=np.float64).copy()
        px = cx + a[..., 0] * cw
        py = cy + a[..., 1] * ch
        a[..., 0] = (px - sx) / side
        a[..., 1] = (py - sy) / side
        out.append(a.astype(np.float32))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="dir of our <vid>.pkl")
    ap.add_argument("--clips", default="data/clips", help="dir of <vid>/meta.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    n = 0
    for p in sorted(glob.glob(os.path.join(args.inp, "*.pkl"))):
        vid = os.path.basename(p)[:-4]
        mp = os.path.join(args.clips, vid, "meta.json")
        if not os.path.exists(mp):
            raise SystemExit(f"no meta.json for {vid}; cannot know the crop it was normalised over")
        m = json.load(open(mp))
        if m.get("crop_style", "native") != "native":
            raise SystemExit(f"{vid} was fetched with crop_style={m['crop_style']}; this tool converts "
                             f"native-crop pkls only")
        W, H = m["source_wh"]
        _, g = openasl_square_crop(m["bbox_norm"], W, H)
        d = pickle.load(open(p, "rb"))
        d["keypoints"] = renorm(d["keypoints"], m["crop_xywh"], g["square_xywh"])
        d["renorm"] = {"from_crop_xywh": m["crop_xywh"], "to_square_xywh": g["square_xywh"],
                       "scale_applied": m["crop_xywh"][2] / g["square_xywh"][2]}
        pickle.dump(d, open(os.path.join(args.out, vid + ".pkl"), "wb"))
        n += 1
    print(f"[renorm] {n} clips -> {args.out}")


if __name__ == "__main__":
    main()
