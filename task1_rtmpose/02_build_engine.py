#!/usr/bin/env python3
"""Build a TensorRT FP32 engine for the RTMPose ONNX on the Jetson.

    python task1_rtmpose/02_build_engine.py --onnx models/rtmpose-x.onnx --engine models/rtmpose-x_fp32.engine

TensorRT 10.3 / JetPack 6.2.1. Engines are NOT portable across TRT versions or
GPUs; always build on the device you run on. Equivalent trtexec command:
    /usr/src/tensorrt/bin/trtexec --onnx=models/rtmpose-x.onnx --saveEngine=models/rtmpose-x_fp32.engine \
        --minShapes=input:1x3x384x288 --optShapes=input:1x3x384x288 --maxShapes=input:1x3x384x288
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.trt_runner import build_engine_from_onnx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/rtmpose-x.onnx")
    ap.add_argument("--engine", default=None)
    ap.add_argument("--fp16", action="store_true", help="FP16 (not for the FP32 baseline)")
    ap.add_argument("--max-batch", type=int, default=1)
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    pp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(args.onnx)), "preproc.json")))
    w, h = pp["input_size"]
    engine = args.engine or args.onnx.replace(".onnx", "_fp16.engine" if args.fp16 else "_fp32.engine")
    profile = {"input": ((1, 3, h, w), (1, 3, h, w), (args.max_batch, 3, h, w))}
    build_engine_from_onnx(args.onnx, engine, fp16=args.fp16, workspace_gb=args.workspace_gb,
                           profiles=[profile], verbose=args.verbose)


if __name__ == "__main__":
    main()
