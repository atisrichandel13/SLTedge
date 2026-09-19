#!/usr/bin/env python3
"""Keypoints -> Uni-Sign model inputs (pose-only path). Interface C3 in WORKSPLIT.md.

Two input formats:
  * our JSON from task1_rtmpose/03_infer_frames.py  (pixel coords in the cropped frame, --crop-wh)
  * Uni-Sign pkl: {"keypoints": [T x (1,133,2) normalised by [W,H]], "scores": [T x (1,133)]}

Two outputs:
  * the Uni-Sign pkl (so any keypoints file can be dropped into their dataset folder and scored), and
  * the model-ready dict {body, left, right, face_all} tensors + a batched `src_input`, exactly what
    `datasets.S2T_Dataset.load_pose` + `collate_fn` produce.

`load_part_kp` and `crop_scale` are copied verbatim from Uni-Sign `datasets.py` (lines 14-105,
commit eed438b) so the numbers match the released checkpoint's training pipeline bit for bit.
Only change: frame subsampling above --max-length is deterministic (uniform) here instead of
random.sample, so results are reproducible; pass --random-subsample to mimic their eval exactly.

    python common/pose_to_unisign.py --keypoints results/rtmpose_trt_fp32.json --crop-wh 666 720 \
        --out-pkl results/jetson_fp32_Ads-4j06eJY.pkl --print
    python common/pose_to_unisign.py --pkl data/openasl_ref_pose/Ads-4j06eJY-*.pkl --print
"""
import argparse
import copy
import json
import os
import pickle
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# COCO-WholeBody index groups used by Uni-Sign (see load_part_kp below)
BODY_IDX = [0] + list(range(3, 11))                                  # 9: nose, ears, shoulders, elbows, wrists
LHAND_IDX = list(range(91, 112))                                     # 21, root = wrist (91)
RHAND_IDX = list(range(112, 133))                                    # 21, root = wrist (112)
FACE_IDX = list(range(23, 23 + 17))[::2] + list(range(83, 83 + 8)) + [53]  # 18, root = 53 (nose tip), last
THR = 0.3


# ----------------------------------------------------------------------------- Uni-Sign, verbatim
def crop_scale(motion, thr):
    '''
        Motion: [(M), T, 17, 3].
        Normalize to [-1, 1]
    '''
    result = copy.deepcopy(motion)
    valid_coords = motion[motion[..., 2] > thr][:, :2]
    if len(valid_coords) < 4:
        return np.zeros(motion.shape), 0, None
    xmin = min(valid_coords[:, 0])
    xmax = max(valid_coords[:, 0])
    ymin = min(valid_coords[:, 1])
    ymax = max(valid_coords[:, 1])
    ratio = 1
    scale = max(xmax - xmin, ymax - ymin) * ratio
    if scale == 0:
        return np.zeros(motion.shape), 0, None
    xs = (xmin + xmax - scale) / 2
    ys = (ymin + ymax - scale) / 2
    result[..., :2] = (motion[..., :2] - [xs, ys]) / scale
    result[..., :2] = (result[..., :2] - 0.5) * 2
    result = np.clip(result, -1, 1)
    result[result[..., 2] <= thr] = 0
    return result, scale, [xs, ys]


def load_part_kp(skeletons, confs, force_ok=False):
    thr = THR
    kps_with_scores = {}
    scale = None
    for part in ['body', 'left', 'right', 'face_all']:
        kps = []
        confidences = []
        for skeleton, conf in zip(skeletons, confs):
            skeleton = skeleton[0]
            conf = conf[0]
            if part == 'body':
                hand_kp2d = skeleton[BODY_IDX, :]
                confidence = conf[BODY_IDX]
            elif part == 'left':
                hand_kp2d = skeleton[91:112, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[91:112]
            elif part == 'right':
                hand_kp2d = skeleton[112:133, :]
                hand_kp2d = hand_kp2d - hand_kp2d[0, :]
                confidence = conf[112:133]
            elif part == 'face_all':
                hand_kp2d = skeleton[FACE_IDX, :]
                hand_kp2d = hand_kp2d - hand_kp2d[-1, :]
                confidence = conf[FACE_IDX]
            else:
                raise NotImplementedError
            kps.append(hand_kp2d)
            confidences.append(confidence)
        kps = np.stack(kps, axis=0)
        confidences = np.stack(confidences, axis=0)
        if part == 'body':
            result, scale, _ = crop_scale(np.concatenate([kps, confidences[..., None]], axis=-1), thr)
        else:
            assert scale is not None
            result = np.concatenate([kps, confidences[..., None]], axis=-1)
            if scale == 0:
                result = np.zeros(result.shape)
            else:
                result[..., :2] = (result[..., :2]) / scale
                result = np.clip(result, -1, 1)
                result[result[..., 2] <= thr] = 0
        kps_with_scores[part] = torch.tensor(result)
    return kps_with_scores
# ----------------------------------------------------------------------------- end verbatim


def load_keypoints_json(path, crop_wh):
    """03_infer_frames.py output -> (list of (1,133,2) normalised, list of (1,133))."""
    d = json.load(open(path))
    rows = sorted(d["results"], key=lambda r: r["frame"])
    wh = np.asarray(crop_wh, dtype=np.float32)
    kps = [np.asarray(r["keypoints"], dtype=np.float32).reshape(1, 133, 2) / wh[None, None] for r in rows]
    scs = [np.asarray(r["scores"], dtype=np.float32).reshape(1, 133) for r in rows]
    return kps, scs, [r["frame"] for r in rows]


def load_pkl(path):
    d = pickle.load(open(path, "rb"))
    kps = [np.asarray(k, dtype=np.float32).reshape(1, 133, 2) for k in d["keypoints"]]
    scs = [np.asarray(s, dtype=np.float32).reshape(1, 133) for s in d["scores"]]
    return kps, scs, None


def save_pkl(path, kps, scs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"keypoints": [k.astype(np.float32) for k in kps],
                     "scores": [s.astype(np.float32) for s in scs]}, f)


def subsample(kps, scs, max_length, random_subsample=False, fps_ratio=1.0):
    """Mirror of S2T_Dataset.load_pose: keep at most max_length frames.
    fps_ratio < 1 first thins EVERY clip uniformly to round(T * fps_ratio) frames (emulates a lower
    camera rate, e.g. 16/24 for 16 fps from 24 fps source); the max_length cap then applies as usual."""
    T = len(scs)
    if fps_ratio < 1.0:
        keep = max(1, int(round(T * fps_ratio)))
        idx0 = np.round(np.linspace(0, T - 1, keep)).astype(int).tolist()
        kps, scs = [kps[i] for i in idx0], [scs[i] for i in idx0]
        k2, s2, idx1 = subsample(kps, scs, max_length, random_subsample)
        return k2, s2, [idx0[i] for i in idx1]
    if T <= max_length:
        return kps, scs, list(range(T))
    if random_subsample:
        idx = sorted(random.sample(range(T), k=max_length))
    else:
        idx = np.round(np.linspace(0, T - 1, max_length)).astype(int).tolist()
    return [kps[i] for i in idx], [scs[i] for i in idx], idx


def to_model_inputs(kps, scs, max_length=256, random_subsample=False, fps_ratio=1.0):
    kps, scs, idx = subsample(kps, scs, max_length, random_subsample, fps_ratio)
    return load_part_kp(kps, scs, force_ok=True), idx


def collate(samples, names):
    """Mirror of S2T_Dataset.collate_fn for a list of load_part_kp dicts -> src_input."""
    src_input = {}
    keys = samples[0].keys()
    for key in keys:
        max_len = max(len(s[key]) for s in samples)
        video_length = torch.LongTensor([len(s[key]) for s in samples])
        padded = [torch.cat((s[key], s[key][-1][None].expand(max_len - len(s[key]), -1, -1)), dim=0)
                  for s in samples]
        src_input[key] = torch.stack(padded, 0)
        if 'attention_mask' not in src_input:
            mask_gen = [torch.ones([i]) + 7 for i in video_length]
            mask_gen = pad_sequence(mask_gen, padding_value=0, batch_first=True)
            src_input['attention_mask'] = (mask_gen != 0).long()
            src_input['name_batch'] = names
            src_input['src_length_batch'] = video_length
    return src_input


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--keypoints", help="03_infer_frames.py JSON")
    g.add_argument("--pkl", help="Uni-Sign pkl")
    ap.add_argument("--crop-wh", type=int, nargs=2, default=[666, 720], help="frame W H for --keypoints")
    ap.add_argument("--out-pkl", default=None, help="write Uni-Sign pkl")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--random-subsample", action="store_true")
    ap.add_argument("--print", action="store_true")
    args = ap.parse_args()

    if args.keypoints:
        kps, scs, frames = load_keypoints_json(args.keypoints, args.crop_wh)
    else:
        kps, scs, frames = load_pkl(args.pkl)
    if args.out_pkl:
        save_pkl(args.out_pkl, kps, scs)
        print("[c3] wrote", args.out_pkl)
    inputs, idx = to_model_inputs(kps, scs, args.max_length, args.random_subsample)
    src = collate([inputs], ["clip"])
    if args.print:
        print(f"[c3] frames in={len(scs)} used={len(idx)}")
        for k in ("body", "left", "right", "face_all"):
            t = inputs[k]
            nz = (t[..., 2] > 0).float().mean().item()
            print(f"[c3] {k:8s} {tuple(t.shape)}  xy range [{t[..., :2].min():+.3f}, {t[..., :2].max():+.3f}]  "
                  f"conf>thr frac {nz:.2f}")
        print("[c3] src_input keys:", {k: (tuple(v.shape) if torch.is_tensor(v) else v) for k, v in src.items()})


if __name__ == "__main__":
    main()
