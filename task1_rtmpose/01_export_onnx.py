#!/usr/bin/env python3
"""Export RTMPose (MMPose 1.x) to ONNX + write preproc.json.

Run this wherever mmpose installs easily (x86 host is fine; ONNX is portable,
only the TensorRT engine must be built on the Jetson):

    pip install -U openmim && mim install "mmengine>=0.10" "mmcv>=2.0.1" "mmpose>=1.3"
    mim download mmpose --config rtmpose-x_8xb32-270e_coco-wholebody-384x288 --dest weights/
    python task1_rtmpose/01_export_onnx.py \
        --config weights/rtmpose-x_8xb32-270e_coco-wholebody-384x288.py \
        --checkpoint weights/rtmpose-x_simcc-coco-wholebody_*.pth \
        --out models/rtmpose-x.onnx

Which RTMPose-x? Uni-Sign consumes COCO-WholeBody keypoints (body+hands+face,
133 kpts), so the wholebody-384x288 model is the default suggestion. The
26-kpt body-only alternative is `rtmpose-x_8xb256-700e_body8-halpe26-384x288`.
Use `mim search mmpose --model rtmpose` to list exact config names.

Alternative official route (MMDeploy) producing an equivalent graph:
    python mmdeploy/tools/deploy.py \
        mmdeploy/configs/mmpose/pose-detection_simcc_onnxruntime_dynamic.py \
        <config.py> <checkpoint.pth> demo.jpg --work-dir out --device cpu
We use torch.onnx.export directly to avoid the MMDeploy dependency. The graph
is `model._forward(x) -> (simcc_x, simcc_y)`; normalisation is NOT inside the
graph (same as MMDeploy), which is why preproc.json exists.
"""
import argparse
import json
import os

import torch


def _dynamo_kw():
    """torch>=2.5 has torch.onnx.export(dynamo=...); >=2.9 defaults it to True. We want the
    TorchScript exporter, so pass dynamo=False where the kwarg exists (Jetson torch 2.8 has it)."""
    major, minor = (int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    return {"dynamo": False} if (major, minor) >= (2, 5) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="models/rtmpose-x.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--simplify", action="store_true", help="run onnxsim (pip install onnxsim)")
    args = ap.parse_args()

    from mmpose.apis import init_model

    # flip_test off so the PyTorch reference is a single forward pass like the engine.
    model = init_model(args.config, args.checkpoint, device=args.device,
                       cfg_options=dict(model=dict(test_cfg=dict(flip_test=False))))
    model.eval()
    cfg = model.cfg
    codec = cfg.codec if "codec" in cfg else cfg.model.head.decoder
    dp = cfg.model.data_preprocessor
    pp = {
        "input_size": list(codec["input_size"]),  # (w, h)
        "simcc_split_ratio": float(codec.get("simcc_split_ratio", 2.0)),
        "mean": [float(v) for v in dp["mean"]],
        "std": [float(v) for v in dp["std"]],
        "bgr_to_rgb": bool(dp.get("bgr_to_rgb", True)),
        "bbox_padding": 1.25,  # GetBBoxCenterScale default in the RTMPose test pipeline
        "config": os.path.abspath(args.config),
        "checkpoint": os.path.abspath(args.checkpoint),
    }
    for t in cfg.test_dataloader.dataset.pipeline:
        if t.get("type") == "GetBBoxCenterScale" and "padding" in t:
            pp["bbox_padding"] = float(t["padding"])

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, x):
            simcc_x, simcc_y = self.m._forward(x)
            return simcc_x, simcc_y

    w, h = pp["input_size"]
    dummy = torch.randn(1, 3, h, w, device=args.device)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    wrapper = Wrapper(model).eval()  # see 02_export_manual.py: export restores the wrapper's mode on exit
    with torch.no_grad():
        torch.onnx.export(
            wrapper, dummy, args.out,
            input_names=["input"], output_names=["simcc_x", "simcc_y"],
            dynamic_axes={"input": {0: "batch"}, "simcc_x": {0: "batch"}, "simcc_y": {0: "batch"}},
            opset_version=args.opset, do_constant_folding=True,
            **_dynamo_kw(),  # TorchScript exporter; torch>=2.9 flips the default to dynamo
        )
    wrapper.eval()
    print("[export] wrote", args.out)

    import onnx
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    for o in m.graph.output:
        print("[export] output", o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
    pp["num_keypoints"] = m.graph.output[0].type.tensor_type.shape.dim[1].dim_value

    if args.simplify:
        import onnxsim
        m, ok = onnxsim.simplify(m)
        assert ok, "onnxsim failed"
        onnx.save(m, args.out)
        print("[export] simplified")

    pp_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "preproc.json")
    with open(pp_path, "w") as f:
        json.dump(pp, f, indent=2)
    print("[export] wrote", pp_path, json.dumps(pp, indent=2))


if __name__ == "__main__":
    main()
