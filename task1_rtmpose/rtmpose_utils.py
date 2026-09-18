#!/usr/bin/env python3
"""Standalone (numpy/cv2-only) RTMPose top-down pre/post-processing.

This mirrors the MMPose 1.x test pipeline for RTMPose:
    GetBBoxCenterScale(padding=1.25) -> TopdownAffine(input_size) -> PackPoseInputs
    -> PoseDataPreprocessor(bgr_to_rgb, mean, std)
and the SimCCLabel decoder + TopdownPoseEstimator.add_pred_to_datasample mapping
back to image coordinates.

The exact constants (input size, mean/std, bgr_to_rgb, simcc_split_ratio) are read
from `preproc.json`, which 01_export_onnx.py writes from the mmpose config, so
this file never hard-codes model-specific numbers.

05_make_reference.py verifies that `preprocess()` here reproduces mmpose's own
pipeline tensor bit-for-bit (up to float rounding), so any drift in these
helpers shows up as a failed check rather than silently wrong keypoints.
"""
import json

import cv2
import numpy as np


def load_preproc(path):
    with open(path) as f:
        return json.load(f)


# --------------------------------------------------------------------- bbox -> affine
def bbox_xyxy2cs(bbox, padding=1.25):
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)[:4]
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    scale = np.array([x2 - x1, y2 - y1], dtype=np.float32) * padding
    return center, scale


def fix_aspect_ratio(scale, aspect_ratio):
    w, h = scale
    if w > h * aspect_ratio:
        return np.array([w, w / aspect_ratio], dtype=np.float32)
    return np.array([h * aspect_ratio, h], dtype=np.float32)


def _rotate_point(pt, angle_rad):
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot = np.array([[cs, -sn], [sn, cs]])
    return rot @ pt


def _get_3rd_point(a, b):
    direction = a - b
    return b + np.r_[-direction[1], direction[0]]


def get_warp_matrix(center, scale, rot, output_size, shift=(0.0, 0.0), inv=False):
    """Same math as mmpose.structures.bbox.get_warp_matrix (fix_aspect_ratio=True)."""
    shift = np.array(shift)
    src_w = scale[0]
    dst_w, dst_h = output_size[:2]
    rot_rad = np.deg2rad(rot)
    src_dir = _rotate_point(np.array([src_w * -0.5, 0.0]), rot_rad)
    dst_dir = np.array([dst_w * -0.5, 0.0])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def preprocess(img_bgr, bbox, pp):
    """img_bgr: HxWx3 uint8. bbox: [x1,y1,x2,y2] or None for full frame.
    Returns (tensor 1x3xHxW float32, center, scale) with scale already aspect-fixed."""
    w, h = pp["input_size"]  # (width, height), e.g. (288, 384)
    if bbox is None:
        bbox = [0, 0, img_bgr.shape[1], img_bgr.shape[0]]
    center, scale = bbox_xyxy2cs(bbox, pp.get("bbox_padding", 1.25))
    scale = fix_aspect_ratio(scale, w / h)
    warp = get_warp_matrix(center, scale, 0.0, (w, h))
    crop = cv2.warpAffine(img_bgr, warp, (int(w), int(h)), flags=cv2.INTER_LINEAR)
    if pp.get("bgr_to_rgb", True):
        crop = crop[..., ::-1]
    x = crop.astype(np.float32)
    x = (x - np.array(pp["mean"], dtype=np.float32)) / np.array(pp["std"], dtype=np.float32)
    x = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    return x, center, scale


# ------------------------------------------------------------------------- decode
def get_simcc_maximum(simcc_x, simcc_y):
    """simcc_x: (N,K,Wx), simcc_y: (N,K,Wy) -> locs (N,K,2) float, vals (N,K)."""
    n, k, _ = simcc_x.shape
    sx = simcc_x.reshape(n * k, -1)
    sy = simcc_y.reshape(n * k, -1)
    x_locs = np.argmax(sx, axis=1)
    y_locs = np.argmax(sy, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_x = np.amax(sx, axis=1)
    max_y = np.amax(sy, axis=1)
    mask = max_x > max_y
    max_x[mask] = max_y[mask]
    vals = max_x
    locs[vals <= 0.0] = -1
    return locs.reshape(n, k, 2), vals.reshape(n, k)


def postprocess(simcc_x, simcc_y, center, scale, pp):
    """Returns keypoints (K,2) in original-image pixels and scores (K,)."""
    locs, vals = get_simcc_maximum(np.asarray(simcc_x), np.asarray(simcc_y))
    kpts = locs / pp["simcc_split_ratio"]  # input-crop pixel space
    input_size = np.array(pp["input_size"], dtype=np.float32)
    kpts = kpts / input_size * scale + center - 0.5 * scale
    return kpts[0], vals[0]


def draw(img_bgr, kpts, scores, thr=0.3):
    out = img_bgr.copy()
    for (x, y), s in zip(kpts, scores):
        if s > thr:
            cv2.circle(out, (int(x), int(y)), 3, (0, 255, 0), -1)
    return out


if __name__ == "__main__":
    # Self-test: a synthetic SimCC peak must map to the expected image pixel.
    pp = {"input_size": [288, 384], "mean": [123.675, 116.28, 103.53],
          "std": [58.395, 57.12, 57.375], "bgr_to_rgb": True, "simcc_split_ratio": 2.0,
          "bbox_padding": 1.25}
    img = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    x, center, scale = preprocess(img, None, pp)
    assert x.shape == (1, 3, 384, 288), x.shape
    # Peak at crop pixel (100, 200) -> simcc bin (200, 400)
    K = 133
    sx = np.zeros((1, K, 288 * 2), np.float32); sy = np.zeros((1, K, 384 * 2), np.float32)
    sx[0, :, 200] = 1.0; sy[0, :, 400] = 0.9
    kp, sc = postprocess(sx, sy, center, scale, pp)
    # crop (100,200) -> image: inverse affine
    inv = get_warp_matrix(center, scale, 0.0, (288, 384), inv=True)
    exp = inv @ np.array([100.0, 200.0, 1.0])
    assert np.allclose(kp[0], exp, atol=1e-3), (kp[0], exp)
    assert np.allclose(sc, 0.9)
    print("rtmpose_utils self-test OK: peak ->", kp[0], "expected", exp)
