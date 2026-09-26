#!/usr/bin/env python3
"""Keypoint dumps -> Uni-Sign pkls, one config at a time. Feeds the pose→BLEU path (guide 2.5).

    python common/dumps_to_pkl.py --dumps results/kpts --config rtmw_fp16 --out data/poses_rtmw_fp16
    python -m unisign.eval_openasl --poses data/poses_rtmw_fp16 --labels data/openasl_labels/labels.test \
        --ckpt weights/openasl_pose_only_slt.pth --mt5 weights/mt5-base --out results/eval_rtmw_fp16.json

`03_infer_frames.py` writes keypoints in the cropped frame's pixel coordinates;
`common/pose_to_unisign.py` needs them normalised by that frame's width and height. The crop size
comes from each clip's `data/clips/<vid>/meta.json`, whose `crop_xywh` is the *delivered* JPEG size
(ffmpeg rounds a crop down to even for yuv420p, so the requested size can be 1 px larger — the
first five clips were fetched before that was known and their metadata was corrected in place).

Needs torch, because pose_to_unisign does; on the board that means inside the container.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.pose_to_unisign import load_keypoints_json, save_pkl  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", default="results/kpts", help="dir of <config>__<vid>.json")
    ap.add_argument("--config", required=True, help="config prefix to convert, e.g. rtmw_fp16")
    ap.add_argument("--clips", default="data/clips", help="dir of <vid>/meta.json")
    ap.add_argument("--out", required=True, help="output dir of <vid>.pkl")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dumps, f"{args.config}__*.json")))
    if not paths:
        raise SystemExit(f"no dumps matching {args.config}__*.json in {args.dumps}")
    os.makedirs(args.out, exist_ok=True)
    for p in paths:
        vid = os.path.basename(p)[len(args.config) + 2:-len(".json")]
        meta = os.path.join(args.clips, vid, "meta.json")
        if not os.path.exists(meta):
            raise SystemExit(f"no meta.json for {vid} (looked in {meta})")
        _, _, w, h = json.load(open(meta))["crop_xywh"]
        kps, scs, frames = load_keypoints_json(p, [w, h])
        # A wrong crop size is silent in the pkl but shifts every coordinate, so check the
        # normalised keypoints land in roughly [0,1] before writing.
        mx = max(float(k[..., 0].max()) for k in kps); my = max(float(k[..., 1].max()) for k in kps)
        if mx > 1.05 or my > 1.05:
            raise SystemExit(f"{vid}: normalised keypoints reach {mx:.3f},{my:.3f} with crop {w}x{h} "
                             f"-- crop size is wrong")
        out = os.path.join(args.out, f"{vid}.pkl")
        save_pkl(out, kps, scs)
        print(f"[c3] {vid}  {len(scs)} frames  crop {w}x{h}  max xy {mx:.3f},{my:.3f}  -> {out}")
    print(f"[c3] {len(paths)} clips -> {args.out}")


if __name__ == "__main__":
    main()
