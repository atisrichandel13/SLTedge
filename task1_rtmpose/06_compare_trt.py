#!/usr/bin/env python3
"""On-device sanity check: TensorRT engine vs the PyTorch reference .npz.

    python task1_rtmpose/06_compare_trt.py --engine models/rtmpose-x_fp32.engine \
        --reference results/rtmpose_reference.npz

Compares (a) raw simcc logits, (b) decoded keypoints in pixels, (c) scores.
Expected for an FP32 engine: simcc max|diff| ~1e-4..1e-3, keypoints identical
except for the odd argmax tie flipping by 0.5 px. FP16 later: ~1e-2 on simcc.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common.trt_runner import TrtRunner  # noqa: E402
from rtmpose_utils import load_preproc, postprocess  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="models/rtmpose-x_fp32.engine")
    ap.add_argument("--reference", default="results/rtmpose_reference.npz")
    ap.add_argument("--preproc", default=None)
    ap.add_argument("--simcc-atol", type=float, default=1e-3)
    ap.add_argument("--kpt-atol-px", type=float, default=1.0)
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="judge keypoint px error only where the PyTorch reference score >= this; "
                         "out-of-frame joints have flat heatmaps whose argmax flips on 1e-3 noise")
    args = ap.parse_args()

    pp = load_preproc(args.preproc or os.path.join(os.path.dirname(os.path.abspath(args.engine)),
                                                    "preproc.json"))
    ref = np.load(args.reference)
    runner = TrtRunner(args.engine)
    in_name = runner.inputs[0]
    n = ref["inputs"].shape[0]
    simcc_diffs, simcc_means, kpt_diffs, kpt_diffs_all, score_diffs = [], [], [], [], []
    worst = (0.0, -1, -1)  # (px, frame, keypoint) among confident keypoints
    n_conf = 0
    for i in range(n):
        outs = runner.infer({in_name: ref["inputs"][i:i + 1]})
        sx = outs["simcc_x"].cpu().numpy(); sy = outs["simcc_y"].cpu().numpy()
        dx = np.abs(sx - ref["simcc_x"][i:i + 1]); dy = np.abs(sy - ref["simcc_y"][i:i + 1])
        simcc_diffs.append(max(dx.max(), dy.max()))
        simcc_means.append((dx.sum() + dy.sum()) / (dx.size + dy.size))
        if np.isnan(sx).any() or np.isnan(sy).any():
            raise RuntimeError(f"NaN in engine output on frame {i}")
        if ref["centers"].size:
            k, s = postprocess(sx, sy, ref["centers"][i], ref["scales"][i], pp)
            px = np.abs(k - ref["keypoints"][i]).max(axis=-1).reshape(-1)  # per keypoint
            conf = ref["scores"][i].reshape(-1) >= args.min_score
            n_conf += int(conf.sum())
            kpt_diffs_all.append(px.max())
            if conf.any():
                j = int(np.argmax(np.where(conf, px, -1)))
                kpt_diffs.append(px[j])
                if px[j] > worst[0]:
                    worst = (float(px[j]), i, j)
            score_diffs.append(np.abs(s - ref["scores"][i]).max())
    rep = {"n_frames": n, "simcc_max_abs_diff": float(max(simcc_diffs)),
           "simcc_mean_abs_diff": float(np.mean(simcc_means))}
    if kpt_diffs:
        rep["keypoint_max_abs_diff_px"] = float(max(kpt_diffs))
        rep["keypoint_worst"] = {"px": worst[0], "frame": worst[1], "keypoint": worst[2]}
        rep["n_confident_keypoints"] = n_conf
        rep["min_score"] = args.min_score
        rep["keypoint_max_abs_diff_px_all_incl_unconfident"] = float(max(kpt_diffs_all))
        rep["score_max_abs_diff"] = float(max(score_diffs))
    ok = rep["simcc_max_abs_diff"] <= args.simcc_atol and \
        rep.get("keypoint_max_abs_diff_px", 0.0) <= args.kpt_atol_px
    rep["PASS"] = bool(ok)
    print(json.dumps(rep, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
