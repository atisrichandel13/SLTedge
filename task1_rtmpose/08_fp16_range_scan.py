#!/usr/bin/env python3
"""Find where an FP16 engine would overflow: run the ONNX on ORT CPU with every intermediate tensor
exposed and report the tensors whose |activation| gets near/over the FP16 max (65504).

    python3 task1_rtmpose/08_fp16_range_scan.py --onnx models/rtmpose-x.onnx \
        --reference results/rtmpose_reference.npz --frames 3 --out results/rtmposex_fp16_range.json

Used in P2: RTMPose-x FP16 failed the keypoint gate by hundreds of px while RTMW-l-m passed the
simcc check, so the question is which layers to keep in FP32 (02_build_engine.py --fp32-layers).
"""
import argparse
import json
import re

import numpy as np

FP16_MAX = 65504.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--reference", required=True, help="npz with 'inputs' (N,3,H,W) from 05*_make_reference")
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import onnx
    import onnxruntime as ort
    m = onnx.load(args.onnx)
    producer = {}
    for node in m.graph.node:
        for o in node.output:
            producer[o] = (node.name or "?", node.op_type)
    existing = {o.name for o in m.graph.output}
    for name in producer:
        if name not in existing:
            m.graph.output.append(onnx.ValueInfoProto(name=name))
    so = ort.SessionOptions(); so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_names = [o.name for o in sess.get_outputs()]
    x = np.load(args.reference)["inputs"][:args.frames].astype(np.float32)

    peak = {}
    for i in range(x.shape[0]):
        outs = sess.run(out_names, {in_name: x[i:i + 1]})
        for n, v in zip(out_names, outs):
            if not isinstance(v, np.ndarray) or v.dtype.kind != "f" or v.size == 0:
                continue
            a = float(np.abs(v).max())
            if a > peak.get(n, (0.0,))[0]:
                peak[n] = (a, float(np.abs(v).mean()))
    rows = sorted(((a, mean, n) + producer.get(n, ("?", "?")) for n, (a, mean) in peak.items()), reverse=True)
    over = [r for r in rows if r[0] > FP16_MAX]
    near = [r for r in rows if FP16_MAX / 4 < r[0] <= FP16_MAX]
    print(f"[scan] {len(rows)} tensors over {x.shape[0]} frames; {len(over)} overflow FP16, {len(near)} within 4x of it")
    for a, mean, n, node, op in rows[:args.top]:
        flag = "OVERFLOW" if a > FP16_MAX else ("near" if a > FP16_MAX / 4 else "")
        print(f"  {a:12.1f}  mean {mean:9.3f}  {op:12s} {node:60s} -> {n}  {flag}")
    # a layer prefix to pin: the ONNX node name up to its last '/' component
    prefixes = sorted({re.sub(r"/[^/]*$", "", r[3]) for r in over + near})
    print("[scan] node-name prefixes of overflow/near tensors:", prefixes)
    if args.out:
        json.dump({"onnx": args.onnx, "frames": int(x.shape[0]), "fp16_max": FP16_MAX,
                   "overflow": [{"max": a, "mean": mn, "tensor": n, "node": node, "op": op} for a, mn, n, node, op in over],
                   "near": [{"max": a, "mean": mn, "tensor": n, "node": node, "op": op} for a, mn, n, node, op in near],
                   "top": [{"max": a, "mean": mn, "tensor": n, "node": node, "op": op} for a, mn, n, node, op in rows[:args.top]],
                   "prefixes": prefixes}, open(args.out, "w"), indent=1)
        print("[scan] wrote", args.out)


if __name__ == "__main__":
    main()
