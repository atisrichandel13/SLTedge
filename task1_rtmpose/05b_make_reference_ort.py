#!/usr/bin/env python3
"""ONNX Runtime (CPU) reference for the TensorRT sanity check, for models that ship as ONNX only.

Same output format as 05_make_reference.py (keys: names, inputs, simcc_x, simcc_y, keypoints,
scores, centers, scales) so 06_compare_trt.py works unchanged. Use it for the RTMW SDK exports
(rtmlib's ONNX files) where there is no mmpose checkpoint to run PyTorch against. For RTMPose-x the
Mac PyTorch reference (05) is the stronger check; on that model ORT matched PyTorch to 8e-6.

    python3 task1_rtmpose/05b_make_reference_ort.py --onnx models/rtmw/rtmw-l-m_256x192.onnx \
        --frames data/test_frames --limit 20 --out results/rtmw_reference.npz

Runs anywhere onnxruntime + cv2 are installed (the Jetson container has the CPU wheel).
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rtmpose_utils import load_preproc, postprocess, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--preproc", default=None, help="preproc.json (default: next to the ONNX)")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads (0 = default)")
    args = ap.parse_args()

    import onnxruntime as ort
    pp = load_preproc(args.preproc or os.path.join(os.path.dirname(os.path.abspath(args.onnx)),
                                                    "preproc.json"))
    so = ort.SessionOptions()
    if args.threads:
        so.intra_op_num_threads = args.threads
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    assert out_names[:2] == ["simcc_x", "simcc_y"], out_names
    print(f"[ort] {args.onnx}: input {in_name} {sess.get_inputs()[0].shape} outputs {out_names}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frames = sorted(p for p in os.listdir(args.frames) if p.lower().endswith(exts))[:args.limit]
    inputs, sxs, sys_, kpts, scores, names, centers, scales = [], [], [], [], [], [], [], []
    for name in frames:
        img = cv2.imread(os.path.join(args.frames, name))
        x, c, s = preprocess(img, None, pp)
        sx, sy = sess.run(["simcc_x", "simcc_y"], {in_name: x})
        if np.isnan(sx).any() or np.isnan(sy).any():
            raise RuntimeError(f"NaN in ORT output on {name}")
        k, v = postprocess(sx, sy, c, s, pp)
        inputs.append(x); sxs.append(sx); sys_.append(sy); kpts.append(k); scores.append(v)
        names.append(name); centers.append(c); scales.append(s)
        print(f"[ref] {name}: simcc_x {sx.shape} simcc_y {sy.shape} confident kpts "
              f"{int((v >= 0.3).sum())}/{len(v)}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), inputs=np.concatenate(inputs),
                        simcc_x=np.concatenate(sxs), simcc_y=np.concatenate(sys_),
                        keypoints=np.stack(kpts), scores=np.stack(scores),
                        centers=np.stack(centers), scales=np.stack(scales))
    print("[ref] wrote", args.out, "frames:", len(names))


if __name__ == "__main__":
    main()
