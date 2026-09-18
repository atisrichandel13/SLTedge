#!/usr/bin/env python3
"""PyTorch reference for the sanity check (run where mmpose is installed).

For each frame it stores: the exact preprocessed tensor produced by the mmpose
pipeline + data_preprocessor, the raw simcc_x/simcc_y from model._forward, and
the decoded keypoints from model.test_step. It also checks that our standalone
rtmpose_utils.preprocess reproduces the mmpose tensor, and (if onnxruntime is
installed) that the ONNX file matches PyTorch.

    python task1_rtmpose/05_make_reference.py --config ... --checkpoint ... \
        --frames data/test_frames --limit 20 --onnx models/rtmpose-x.onnx \
        --out results/rtmpose_reference.npz

Then copy the .npz to the Jetson and run 06_compare_trt.py.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from rtmpose_utils import postprocess, preprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--onnx", default=None, help="also compare ONNX Runtime vs PyTorch")
    ap.add_argument("--preproc", default=None)
    ap.add_argument("--out", default="results/rtmpose_reference.npz")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from mmengine.dataset import Compose, pseudo_collate
    from mmengine.registry import init_default_scope
    from mmpose.apis import init_model

    model = init_model(args.config, args.checkpoint, device=args.device,
                       cfg_options=dict(model=dict(test_cfg=dict(flip_test=False))))
    model.eval()
    init_default_scope(model.cfg.get("default_scope", "mmpose"))
    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    pp_path = args.preproc or (os.path.join(os.path.dirname(os.path.abspath(args.onnx)), "preproc.json")
                               if args.onnx else None)
    pp = json.load(open(pp_path)) if pp_path and os.path.exists(pp_path) else None

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frames = sorted(p for p in os.listdir(args.frames) if p.lower().endswith(exts))[:args.limit]
    inputs, sxs, sys_, kpts, scores, names, centers, scales = [], [], [], [], [], [], [], []
    max_pre_diff = 0.0
    for name in frames:
        img = cv2.imread(os.path.join(args.frames, name))
        h, w = img.shape[:2]
        data_info = dict(img=img, bbox=np.array([[0, 0, w, h]], dtype=np.float32),
                         bbox_score=np.ones(1, dtype=np.float32))
        data_info.update(model.dataset_meta)
        data = pipeline(data_info)
        batch = pseudo_collate([data])
        with torch.no_grad():
            pre = model.data_preprocessor(batch, False)
            x = pre["inputs"]
            sx, sy = model._forward(x)
            res = model.test_step(batch)[0]
        inputs.append(x.cpu().numpy()); sxs.append(sx.cpu().numpy()); sys_.append(sy.cpu().numpy())
        kpts.append(res.pred_instances.keypoints[0]); scores.append(res.pred_instances.keypoint_scores[0])
        names.append(name)
        if pp is not None:
            x_np, c, s = preprocess(img, None, pp)
            d = float(np.abs(x_np - x.cpu().numpy()).max())
            max_pre_diff = max(max_pre_diff, d)
            centers.append(c); scales.append(s)
            k_np, _ = postprocess(sx.cpu().numpy(), sy.cpu().numpy(), c, s, pp)
            kd = float(np.abs(k_np - res.pred_instances.keypoints[0]).max())
            if kd > 1e-2:
                print(f"[warn] standalone decode differs from mmpose by {kd:.4f}px on {name}")
        print(f"[ref] {name}: simcc_x {tuple(sx.shape)} kpts {res.pred_instances.keypoints.shape}")

    if pp is not None:
        print(f"[check] standalone preprocess vs mmpose pipeline: max|diff| = {max_pre_diff:.3e} "
              f"({'OK' if max_pre_diff < 1e-3 else 'MISMATCH - fix rtmpose_utils.preprocess'})")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, names=np.array(names), inputs=np.concatenate(inputs),
                        simcc_x=np.concatenate(sxs), simcc_y=np.concatenate(sys_),
                        keypoints=np.stack(kpts), scores=np.stack(scores),
                        centers=np.stack(centers) if centers else np.zeros(0),
                        scales=np.stack(scales) if scales else np.zeros(0))
    print("[ref] wrote", args.out)

    if args.onnx:
        try:
            import onnxruntime as ort
        except ImportError:
            print("[onnx] onnxruntime not installed, skipping ONNX check"); return
        sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
        worst = 0.0
        for x, sx, sy in zip(inputs, sxs, sys_):
            ox, oy = sess.run(None, {"input": x})
            worst = max(worst, float(np.abs(ox - sx).max()), float(np.abs(oy - sy).max()))
        print(f"[check] ONNX Runtime vs PyTorch simcc: max|diff| = {worst:.3e} "
              f"({'OK' if worst < 1e-3 else 'investigate'})")


if __name__ == "__main__":
    main()
