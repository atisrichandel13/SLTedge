#!/usr/bin/env python3
"""Run the RTMW ONNX through ONNX Runtime on a chosen execution provider and dump keypoints in the
03_infer_frames.py format, so 07_kpt_agreement.py scores it against the board engines unchanged.

    python3 task1_rtmpose/05c_infer_frames_ort.py --onnx <rtmw .onnx> --preproc models/rtmw/preproc.json \
        --frames data/test_frames --provider coreml --compute all --out results/ort_coreml_all.json

Why this exists. The mobile track runs the pose stage on the phone through ONNX Runtime's CoreML
execution provider. Before a single line of Swift, the same ONNX bytes the board engine was built from
(sha256 68db326c... == the OpenMMLab SDK end2end.onnx) can be run on the Mac's CoreML stack and scored
against results/rtmw_trt_fp32.json. That answers, on real frames, how far the Apple path sits from the
board path -- the number the phone's keypoint-agreement gate will be judged against.

--provider cpu       ORT CPU EP, the reference for "is it the EP or the model".
--provider coreml    ORT CoreML EP, --compute selects MLComputeUnits: all | cpu_gpu | cpu_ne | cpu_only.
                     The Neural Engine is FP16-only, so cpu_ne is the setting that predicts the phone.

The batch axis is overridden to 1 so the CoreML EP sees static shapes; the board engine is batch 1 too.
ORT's partition log is captured so the dump records how many nodes actually ran on CoreML rather than
falling back to CPU -- a builder flag is a permission, not an instruction (same lesson as TensorRT).
"""
import argparse
import io
import json
import os
import platform
import re
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from rtmpose_utils import load_preproc, postprocess, preprocess  # noqa: E402

COMPUTE = {"all": "ALL", "cpu_gpu": "CPUAndGPU", "cpu_ne": "CPUAndNeuralEngine", "cpu_only": "CPUOnly"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--preproc", default=None, help="preproc.json (default: next to the ONNX)")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--provider", choices=["cpu", "coreml"], default="cpu")
    ap.add_argument("--compute", choices=sorted(COMPUTE), default="all")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import onnxruntime as ort
    pp = load_preproc(args.preproc or os.path.join(os.path.dirname(os.path.abspath(args.onnx)),
                                                    "preproc.json"))
    so = ort.SessionOptions()
    so.add_free_dimension_override_by_name("batch", 1)
    so.log_severity_level = 1  # INFO: the CoreML EP reports its partition here
    if args.threads:
        so.intra_op_num_threads = args.threads
    if args.provider == "coreml":
        providers = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram",
                                                  "MLComputeUnits": COMPUTE[args.compute]}),
                     "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    # ORT writes the partition summary to stderr (C++ logger), so capture at the fd level -- into a
    # FILE, not a pipe. At INFO level the CoreML EP writes more than a pipe buffer holds, and ORT keeps
    # the GIL for the whole of session construction, so nothing in this process can drain a pipe
    # meanwhile: the writer blocks and InferenceSession() never returns. A file never blocks.
    import tempfile
    err_fd = sys.stderr.fileno()
    saved = os.dup(err_fd)
    logf = tempfile.TemporaryFile(mode="w+b")
    os.dup2(logf.fileno(), err_fd)
    try:
        sess = ort.InferenceSession(args.onnx, so, providers=providers)
    finally:
        os.dup2(saved, err_fd)
        os.close(saved)
    logf.seek(0)
    log = logf.read().decode(errors="replace")
    logf.close()
    partition = None
    m = re.search(r"number of nodes in the graph: (\d+) number of nodes supported by CoreML: (\d+)", log)
    if m:
        partition = {"graph_nodes": int(m.group(1)), "coreml_nodes": int(m.group(2))}
    in_name = sess.get_inputs()[0].name
    assert [o.name for o in sess.get_outputs()][:2] == ["simcc_x", "simcc_y"]
    print(f"[ort] providers in use: {sess.get_providers()}  partition: {partition}")

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    frames = sorted(p for p in os.listdir(args.frames) if p.lower().endswith(exts))
    if args.limit:
        frames = frames[:args.limit]

    results, run_ms, pre_ms, post_ms = [], [], [], []
    n_nonfinite = 0
    t_wall = time.perf_counter()
    for i, name in enumerate(frames):
        img = cv2.imread(os.path.join(args.frames, name))
        t0 = time.perf_counter()
        x, c, s = preprocess(img, None, pp)
        t1 = time.perf_counter()
        sx, sy = sess.run(["simcc_x", "simcc_y"], {in_name: x})
        t2 = time.perf_counter()
        if not (np.isfinite(sx).all() and np.isfinite(sy).all()):
            n_nonfinite += 1
        k, v = postprocess(sx, sy, c, s, pp)
        t3 = time.perf_counter()
        if i >= args.warmup:
            pre_ms.append((t1 - t0) * 1e3); run_ms.append((t2 - t1) * 1e3); post_ms.append((t3 - t2) * 1e3)
        results.append({"frame": name, "keypoints": np.asarray(k).reshape(133, 2).tolist(),
                        "scores": np.asarray(v).reshape(133).tolist()})
    wall = time.perf_counter() - t_wall

    def ms(a):
        a = sorted(a)
        return {"mean": round(statistics.mean(a), 3), "median": round(statistics.median(a), 3),
                "p95": round(a[int(0.95 * (len(a) - 1))], 3), "min": round(a[0], 3), "max": round(a[-1], 3)}

    out = {
        "summary": {
            "n_frames": len(results), "n_timed": len(run_ms), "warmup": args.warmup,
            "wall_s": round(wall, 3), "n_nonfinite": n_nonfinite,
            "provider": args.provider, "compute_units": COMPUTE[args.compute] if args.provider == "coreml" else None,
            "providers_in_use": sess.get_providers(), "partition": partition,
            "onnxruntime": ort.__version__, "machine": platform.machine(), "platform": platform.platform(),
            "preprocess_ms": ms(pre_ms), "run_ms": ms(run_ms), "postprocess_ms": ms(post_ms),
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(out, open(args.out, "w"))
    print(f"[ort] {len(results)} frames, run_ms {out['summary']['run_ms']}, nonfinite {n_nonfinite} -> {args.out}")


if __name__ == "__main__":
    main()
