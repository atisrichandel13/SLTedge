#!/usr/bin/env python3
"""YOLOX person box in front of the pose model, as the authors' extraction does it.

The authors' `demo/pose_extraction.py` calls rtmlib `Wholebody`, which runs a YOLOX person detector
and gives the pose model the detected box, not the frame. Their vendored rtmlib is edited to keep only
the LARGEST box (tools/object_detection/yolox.py), which is why every released pkl has exactly one
person per frame. With no box, rtmlib falls back to the full frame -- which is what we have always fed.

On the 5 openasl-crop test clips (2026-11-03) that box is the one pipeline difference that moves our
keypoints toward theirs: median used69 distance 0.0058 -> 0.0040, against 0.0046 for our native crop.
Colour order (they feed BGR) moved it by 0.0001 and is not reproduced here.

Mirrors rtmlib exactly: letterbox into a 114-grey square, raw uint8 pixels (no mean/std), then keep
score > 0.3 and take the largest area (rtmlib ignores the class label, so we do too).

Two backends:
  * the full mmdeploy ONNX on ORT (NMS in-graph, dets (1, N, 5)) -- `YoloxPersonDet(onnx)`
  * a graph CUT BEFORE NMS (`cut_before_nms`), for TensorRT: NonMaxSuppression has a data-dependent
    output shape that TrtRunner cannot allocate for, so NMS runs here in numpy with the graph's own
    constants (IoU 0.65, <= 200 per class). Filtering at 0.3 BEFORE NMS gives the same survivors as
    the graph's 0.01 followed by rtmlib's 0.3, because greedy NMS only lets a higher-scoring box
    suppress a lower one. `YoloxPersonDet(cut_onnx_or_engine, pre_nms=True)`.

    det = YoloxPersonDet("yolox_tiny/end2end.onnx", size=416)    # rtmlib "lightweight"
    box = det(img_bgr)          # [x1, y1, x2, y2] in image pixels, or None
"""
import cv2
import numpy as np

NMS_IOU = 0.65          # NonMaxSuppression_490 input 1193 in the yolox-tiny/-m humanart SDK exports
NMS_PER_CLASS = 200     # input 1192
PRE_NMS_OUTPUTS = ("1211", "1214")   # boxes (1, A, 4) after TopK_485/Gather_487, scores (1, 80, A)


def authors_224(bbox_norm, W, H, crop_xy, target=224):
    """Rebuild the authors' 224 input from our native crop: paste it into their square (black where the
    square reaches past our crop -- an approximation, their square has real background there), then
    cv2.resize INTER_LINEAR as their prep/crop_video.py does. On the 5 openasl-crop clips this matches
    the true 224 frames to 0.0041 vs 0.0040 used69 distance from the authors' poses.

    Returns (to_224(img), back(k)): back maps keypoints from 224 pixels to native-crop pixels, so every
    downstream normalisation (square-norm, crop-norm) is unchanged."""
    from openasl_fetch import openasl_square_crop
    cx, cy = crop_xy[0], crop_xy[1]
    _, g = openasl_square_crop(bbox_norm, W, H)
    sx, sy, side = g["square_xywh"][0], g["square_xywh"][1], g["square_xywh"][2]
    ox, oy = cx - sx, cy - sy

    def to_224(img):
        canvas = np.zeros((side, side, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        x0, y0 = max(ox, 0), max(oy, 0)
        x1, y1 = min(ox + w, side), min(oy + h, side)
        canvas[y0:y1, x0:x1] = img[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
        return cv2.resize(canvas, (target, target), interpolation=cv2.INTER_LINEAR)

    def back(k):
        return (k * (side / target) + np.array([sx - cx, sy - cy], dtype=np.float32)).astype(np.float32)
    return to_224, back


def cut_before_nms(onnx_in, onnx_out):
    import onnx.utils
    onnx.utils.extract_model(onnx_in, onnx_out, ["input"], list(PRE_NMS_OUTPUTS))


def _nms(boxes, scores, iou, cap):
    order = np.argsort(-scores, kind="stable")
    keep = []
    while order.size and len(keep) < cap:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0]); yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2]); yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        ov = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-12)
        order = rest[ov <= iou]
    return keep


class YoloxPersonDet:
    def __init__(self, model_path, size=416, threads=0, score_thr=0.3, pre_nms=False):
        self.size, self.score_thr, self.pre_nms = size, score_thr, pre_nms
        if model_path.endswith(".engine"):
            from common.trt_runner import TrtRunner
            self.trt = TrtRunner(model_path)
            self.inp = self.trt.inputs[0]
            self.pre_nms = True
        else:
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            if threads:
                so.intra_op_num_threads = threads
            self.trt = None
            self.sess = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name

    def _run(self, x):
        if self.trt is not None:
            o = self.trt.infer({self.inp: x})
            return [o[n].float().cpu().numpy() for n in PRE_NMS_OUTPUTS]
        return self.sess.run(None, {self.inp: x})

    def __call__(self, img):
        s = self.size
        padded = np.full((s, s, 3), 114, dtype=np.uint8)
        ratio = min(s / img.shape[0], s / img.shape[1])
        r = cv2.resize(img, (int(img.shape[1] * ratio), int(img.shape[0] * ratio)),
                       interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded[:r.shape[0], :r.shape[1]] = r
        x = np.ascontiguousarray(padded.transpose(2, 0, 1), dtype=np.float32)[None]
        out = self._run(x)
        if self.pre_nms:
            bx, sc = out[0][0], out[1][0]                     # (A, 4), (C, A)
            sel_b, sel_s = [], []
            for c in range(sc.shape[0]):
                m = sc[c] > self.score_thr
                if not m.any():
                    continue
                b, v = bx[m], sc[c][m]
                k = _nms(b, v, NMS_IOU, NMS_PER_CLASS)
                sel_b.append(b[k]); sel_s.append(v[k])
            if not sel_b:
                return None
            boxes = np.concatenate(sel_b) / ratio
        else:
            dets = out[0]                                     # (1, N, 5), NMS inside the graph
            boxes, scores = dets[0, :, :4] / ratio, dets[0, :, 4]
            boxes = boxes[scores > self.score_thr]
            if len(boxes) == 0:
                return None
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes[int(np.argmax(areas))]
