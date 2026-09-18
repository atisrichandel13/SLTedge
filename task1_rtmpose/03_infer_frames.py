#!/usr/bin/env python3
"""Run the RTMPose TensorRT engine over a folder of frames; report per-frame latency.

    python task1_rtmpose/03_infer_frames.py --engine models/rtmpose-x_fp32.engine \
        --frames data/test_frames --out results/rtmpose_trt.json [--vis-dir results/vis]

Latency is split into preprocess (CPU: warp+normalise), trt (GPU H2D+exec+sync),
and postprocess (CPU: argmax + coordinate mapping). "trt_ms" is the number to
quote as engine latency; "total_ms" is the end-to-end per-frame cost.

Bounding boxes: by default the whole frame is the person box (no detector in
this project yet). Pass --bboxes bboxes.json ({"frame.jpg": [x1,y1,x2,y2]}) to
use real boxes, e.g. from RTMDet, later.
"""
import argparse
import glob
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common.envinfo import print_env  # noqa: E402
from common.trt_runner import TrtRunner  # noqa: E402
from rtmpose_utils import draw, load_preproc, postprocess, preprocess  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_frames(folder, limit=None):
    frames = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith(EXTS))
    if not frames:
        raise SystemExit(f"no frames in {folder}")
    return frames[:limit] if limit else frames


def add_args(ap):
    ap.add_argument("--engine", default="models/rtmpose-x_fp32.engine")
    ap.add_argument("--preproc", default=None, help="preproc.json (default: next to engine)")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--bboxes", default=None)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--out", default=None, help="JSON with keypoints + latencies")
    ap.add_argument("--vis-dir", default=None)


def run_folder(args, runner=None, quiet=False):
    pp = load_preproc(args.preproc or os.path.join(os.path.dirname(os.path.abspath(args.engine)),
                                                    "preproc.json"))
    runner = runner or TrtRunner(args.engine)
    if not quiet:
        runner.describe()
    frames = list_frames(args.frames, args.limit)
    bboxes = json.load(open(args.bboxes)) if args.bboxes else {}
    in_name = runner.inputs[0]

    # Warm-up (first executions include lazy CUDA/cuDNN init and clock ramp-up).
    x0, _, _ = preprocess(cv2.imread(frames[0]), None, pp)
    for _ in range(args.warmup):
        runner.infer({in_name: x0})

    results, lat = [], {"preprocess_ms": [], "trt_ms": [], "postprocess_ms": [], "total_ms": []}
    t_all0 = time.perf_counter()
    for path in frames:
        img = cv2.imread(path)
        t0 = time.perf_counter()
        x, center, scale = preprocess(img, bboxes.get(os.path.basename(path)), pp)
        t1 = time.perf_counter()
        outs = runner.infer({in_name: x})  # synchronises the stream
        t2 = time.perf_counter()
        sx = outs["simcc_x"].cpu().numpy()
        sy = outs["simcc_y"].cpu().numpy()
        kpts, scores = postprocess(sx, sy, center, scale, pp)
        t3 = time.perf_counter()
        lat["preprocess_ms"].append((t1 - t0) * 1e3)
        lat["trt_ms"].append((t2 - t1) * 1e3)
        lat["postprocess_ms"].append((t3 - t2) * 1e3)
        lat["total_ms"].append((t3 - t0) * 1e3)
        results.append({"frame": os.path.basename(path), "keypoints": kpts.round(2).tolist(),
                        "scores": scores.round(4).tolist()})
        if args.vis_dir:
            os.makedirs(args.vis_dir, exist_ok=True)
            cv2.imwrite(os.path.join(args.vis_dir, os.path.basename(path)), draw(img, kpts, scores))
    wall_s = time.perf_counter() - t_all0

    summary = {"n_frames": len(frames), "wall_s": round(wall_s, 3),
               "fps_end_to_end": round(len(frames) / wall_s, 2)}
    for k, v in lat.items():
        summary[k] = {"mean": round(statistics.mean(v), 3), "median": round(statistics.median(v), 3),
                      "p95": round(float(np.percentile(v, 95)), 3), "min": round(min(v), 3),
                      "max": round(max(v), 3)}
    if not quiet:
        print(json.dumps(summary, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "per_frame_latency": lat, "results": results}, f)
        if not quiet:
            print("[infer] wrote", args.out)
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    add_args(ap)
    a = ap.parse_args()
    print_env()
    run_folder(a)
