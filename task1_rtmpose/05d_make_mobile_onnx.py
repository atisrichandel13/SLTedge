#!/usr/bin/env python3
"""Phone variants of the RTMW ONNX: batch fixed to 1, and an FP16 copy for the Neural Engine.

    python3 task1_rtmpose/05d_make_mobile_onnx.py --onnx models/rtmw/rtmw-l-m_256x192.onnx \
        --out-dir models/rtmw

writes rtmw-l-m_256x192_b1.onnx (FP32) and rtmw-l-m_256x192_b1_fp16.onnx (FP16 weights and compute,
FP32 inputs/outputs so the app's tensors do not change).

Why batch 1. On the phone, ORT's CoreML EP took 344 of 397 nodes in 6 partitions with the dynamic
batch axis; every partition boundary is a CoreML<->CPU round trip per frame. The Mac run, where a
free-dimension override fixed the batch, took 385 of 391. Fixing the axis in the graph is the
override the Objective-C API does not expose.

Why FP16. The A14's Neural Engine executes FP16 only, so an FP32 MLProgram runs on the GPU/CPU.
RTMW-l-m FP16 is the precision the board ships and passed its BLEU gate; on the phone it must pass
the same keypoint-agreement check against results/rtmw_trt_fp32.json before it is used.
"""
import argparse
import os

import onnx
from onnx import shape_inference


def fix_batch(m):
    for vi in list(m.graph.input) + list(m.graph.output):
        d0 = vi.type.tensor_type.shape.dim[0]
        d0.ClearField("dim_param"); d0.dim_value = 1
    for vi in list(m.graph.value_info):
        m.graph.value_info.remove(vi)
    return shape_inference.infer_shapes(m)


def unshare_initializers(m):
    """Give every node its own copy of any initializer that several nodes read. Needed for a partial
    FP16 conversion: onnxconverter_common converts an initializer if ANY reader is converted, so a
    Clip min/max shared between an FP16 backbone Clip and an FP32 head Clip ends up one type, and
    the other Clip fails type checking (seen on Clip_350 vs Clip_369/379)."""
    inits = {i.name: i for i in m.graph.initializer}
    seen = {}
    for node in m.graph.node:
        for j, name in enumerate(node.input):
            if name not in inits:
                continue
            seen[name] = seen.get(name, 0) + 1
            if seen[name] > 1:
                dup = onnx.TensorProto(); dup.CopyFrom(inits[name])
                dup.name = f"{name}__{node.name}"
                m.graph.initializer.append(dup)
                node.input[j] = dup.name
    return m


def head_nodes(m):
    """The SimCC head: everything after the backbone's last Conv (final_layer conv onward is kept
    in the backbone; the GAU, its ScaleNorms and the cls_x / cls_y Linears are the head)."""
    n = m.graph.node
    last_conv = max(i for i, x in enumerate(n) if x.op_type == "Conv")
    return [x.name for x in n[last_conv + 1:]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    stem = os.path.splitext(os.path.basename(args.onnx))[0]
    m = fix_batch(onnx.load(args.onnx))
    p32 = os.path.join(args.out_dir, f"{stem}_b1.onnx")
    onnx.save(m, p32)
    from onnxconverter_common import float16
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    p16 = os.path.join(args.out_dir, f"{stem}_b1_fp16.onnx")
    onnx.save(m16, p16)
    # FP16 backbone, FP32 SimCC head: a blanket FP16 conversion put only 46.8 % of Uni-Sign's
    # keypoints within 1 px of the FP32 engine on the Mac (max 267 px), with no tensor anywhere near
    # the FP16 range (largest activation 211, results/rtmw_fp16_range.json) -- so it is precision in
    # the argmax head, not overflow, and the head is where FP32 is kept.
    mh = unshare_initializers(onnx.load(p32))
    mh16 = float16.convert_float_to_float16(mh, keep_io_types=True, node_block_list=head_nodes(mh))
    ph = os.path.join(args.out_dir, f"{stem}_b1_fp16head32.onnx")
    onnx.save(mh16, ph)
    for p in (p32, p16, ph):
        onnx.checker.check_model(p)
        g = onnx.load(p).graph
        print(f"{p}: {os.path.getsize(p) / 2**20:.1f} MB, in {[(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim]) for i in g.input]}")


if __name__ == "__main__":
    main()
