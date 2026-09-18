#!/usr/bin/env python3
"""Minimal TensorRT 10.x runner using PyTorch CUDA tensors for device memory.

Why torch instead of pycuda: PyTorch is already on the JetPack image, torch
tensors give us device pointers (`.data_ptr()`) and a CUDA stream, so no
extra dependency. This is the TRT 10 "v3" API (set_tensor_address +
execute_async_v3); the old bindings API is gone in TRT 10.

    from common.trt_runner import TrtRunner
    r = TrtRunner("model.engine")
    outs = r.infer({"input": np_or_torch_array})     # dict name -> torch.Tensor (on GPU)
"""
import os
import numpy as np
import tensorrt as trt
import torch

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

_TRT_TO_TORCH = {
    trt.DataType.FLOAT: torch.float32,
    trt.DataType.HALF: torch.float16,
    trt.DataType.INT32: torch.int32,
    trt.DataType.INT64: torch.int64,
    trt.DataType.BOOL: torch.bool,
    trt.DataType.INT8: torch.int8,
}


def load_engine(path):
    with open(path, "rb") as f, trt.Runtime(TRT_LOGGER) as rt:
        engine = rt.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError(f"failed to deserialize {path}")
    return engine


class TrtRunner:
    def __init__(self, engine_path, device="cuda:0"):
        self.engine = load_engine(engine_path)
        self.ctx = self.engine.create_execution_context()
        self.device = torch.device(device)
        self.stream = torch.cuda.Stream(device=self.device)
        self.inputs, self.outputs = [], []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(name)
            else:
                self.outputs.append(name)
        self._out_bufs = {}

    def dtype(self, name):
        return _TRT_TO_TORCH[self.engine.get_tensor_dtype(name)]

    def shape(self, name):
        return tuple(self.engine.get_tensor_shape(name))

    def describe(self):
        for n in self.inputs:
            print(f"  input  {n:40s} {self.shape(n)} {self.engine.get_tensor_dtype(n)}")
        for n in self.outputs:
            print(f"  output {n:40s} {self.shape(n)} {self.engine.get_tensor_dtype(n)}")

    def _to_dev(self, name, arr):
        if isinstance(arr, np.ndarray):
            t = torch.from_numpy(np.ascontiguousarray(arr))
        else:
            t = arr
        return t.to(self.device, dtype=self.dtype(name), non_blocking=True).contiguous()

    def infer(self, feed, sync=True):
        """feed: dict input_name -> np.ndarray | torch.Tensor. Returns dict of GPU tensors."""
        missing = set(self.inputs) - set(feed)
        if missing:
            raise KeyError(f"missing inputs {missing}; engine inputs are {self.inputs}")
        keep = []  # keep device tensors alive until the stream is synced
        with torch.cuda.stream(self.stream):
            for name in self.inputs:
                t = self._to_dev(name, feed[name])
                keep.append(t)
                if not self.ctx.set_input_shape(name, tuple(t.shape)):
                    raise RuntimeError(f"set_input_shape failed for {name} {tuple(t.shape)} "
                                       f"(check optimization profile)")
                self.ctx.set_tensor_address(name, t.data_ptr())
            outs = {}
            for name in self.outputs:
                shape = tuple(self.ctx.get_tensor_shape(name))
                buf = self._out_bufs.get(name)
                if buf is None or tuple(buf.shape) != shape:
                    buf = torch.empty(shape, dtype=self.dtype(name), device=self.device)
                    self._out_bufs[name] = buf
                self.ctx.set_tensor_address(name, buf.data_ptr())
                outs[name] = buf
            ok = self.ctx.execute_async_v3(self.stream.cuda_stream)
            if not ok:
                raise RuntimeError("execute_async_v3 returned False")
        if sync:
            self.stream.synchronize()
        return outs


def build_engine_from_onnx(onnx_path, engine_path, fp16=False, workspace_gb=2.0,
                           profiles=None, verbose=False):
    """Build a TRT engine. `profiles` = list of dict name -> (min, opt, max) shape tuples."""
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)  # TRT 10: explicit batch is the only mode
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("ONNX parse error:", parser.get_error(i))
            raise RuntimeError("ONNX parse failed")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    else:
        # TF32 is ON by default on Ampere (Orin). It rounds matmul/conv inputs to 10-bit mantissa,
        # so an "FP32" engine would not match PyTorch. Keep the FP32 baseline honest.
        config.clear_flag(trt.BuilderFlag.TF32)
        print("[trt] TF32 disabled for the FP32 baseline")
    for prof in (profiles or []):
        p = builder.create_optimization_profile()
        for name, (mn, opt, mx) in prof.items():
            p.set_shape(name, mn, opt, mx)
        config.add_optimization_profile(p)
    print(f"[trt] building {engine_path} (fp16={fp16}) ... this can take minutes on Orin Nano")
    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError("engine build failed")
    with open(engine_path, "wb") as f:
        f.write(plan)
    print(f"[trt] wrote {engine_path} ({os.path.getsize(engine_path) / 1e6:.1f} MB)")
    return engine_path
