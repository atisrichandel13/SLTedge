#!/usr/bin/env python3
"""ONNX Runtime stand-in for TrtRunner, so 09_batch_clips.py can extract on the pose-track Mac.

Same interface as common/trt_runner.py -- `.inputs` is the list of input names and `infer()` takes a
dict name -> array and returns a dict name -> array -- so the batch loop does not care which one it
holds. CPU provider only, on purpose: CoreML is available on the Mac but may run in FP16 on the ANE,
and the point of this path is an FP32 comparison against the board's FP32 TensorRT arm.

    from common.ort_runner import OrtRunner
    r = OrtRunner("end2end.onnx")
    outs = r.infer({r.inputs[0]: x})     # dict name -> np.ndarray
"""
import numpy as np
import onnxruntime as ort


class OrtRunner:
    def __init__(self, onnx_path, threads=0):
        so = ort.SessionOptions()
        so.log_severity_level = 3  # the SDK exports carry unused initialisers that warn on every load
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"])
        self.inputs = [i.name for i in self.sess.get_inputs()]
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def infer(self, feeds):
        feeds = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in feeds.items()}
        return dict(zip(self.outputs, self.sess.run(self.outputs, feeds)))
