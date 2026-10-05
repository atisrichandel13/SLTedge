#!/usr/bin/env python3
"""Extract keypoints for MANY clips in one process, and convert to Uni-Sign pkl as it goes.

    python task1_rtmpose/09_batch_clips.py --engine models/rtmw/rtmw-l-m_256x192_fp16.engine \
        --clips-dir data/clips --config rtmw_fp16 --pkl-out results/pkl_full_rtmw_fp16 \
        --square-norm --limit-clips 100

Why this exists. `jetson/p4_clips.sh` launches one container exec per (config, clip), and we measured
~15.4 s per clip of which only ~6 s is inference: the other ~9 s is docker exec plus engine load plus
CUDA init. At 974 clips that is 2.4 HOURS of pure overhead per configuration. Here the engine is built
once and every clip reuses it.

It also writes the Uni-Sign pkl directly and (with --drop-dumps) never keeps the intermediate keypoint
JSON, because 974 clips x 4 configs of dumps is ~5.6 GB on a shared board while the pkls are ~1.4 GB.

RESUMABILITY: a clip whose pkl already exists is skipped, so an interrupted run resumes at the clip it
died on. Nothing is rewritten, so re-running is free.

--square-norm applies the coordinate-frame correction (RESULTS.md 2.5d) while the keypoints are still in
crop pixels, which is where it belongs: the frozen encoder consumes absolute body coordinates and our
bbox crop makes them ~1.68x larger than the square frame the checkpoint was trained on. Needs each
clip's meta.json, which the fetcher writes.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common.trt_runner import TrtRunner  # noqa: E402
from rtmpose_utils import load_preproc, postprocess, preprocess  # noqa: E402

import cv2  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def square_fn(meta):
    from openasl_fetch import openasl_square_crop
    W, H = meta["source_wh"]
    cx, cy, _, _ = meta["crop_xywh"]
    _, g = openasl_square_crop(meta["bbox_norm"], W, H)
    sx, sy, side = g["square_xywh"][0], g["square_xywh"][1], g["square_xywh"][2]

    def f(k):
        o = np.empty_like(k, dtype=np.float32)
        o[..., 0] = (cx + k[..., 0] - sx) / side
        o[..., 1] = (cy + k[..., 1] - sy) / side
        return o
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--preproc", default=None)
    ap.add_argument("--clips-dir", default="data/clips")
    ap.add_argument("--config", required=True, help="label, e.g. rtmw_fp16 (used in --dumps-out names)")
    ap.add_argument("--pkl-out", required=True)
    ap.add_argument("--dumps-out", default=None, help="also write the raw keypoint JSON here")
    ap.add_argument("--square-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--pkl-out-raw", default=None,
                    help="ALSO write crop-normalised pkls here. The square-frame and crop-frame forms "
                         "are the same keypoints under two exact normalisations, so writing both costs "
                         "no GPU work and saves a whole second extraction pass over the split")
    ap.add_argument("--limit-clips", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--progress-every", type=int, default=10)
    args = ap.parse_args()

    metas = sorted(glob.glob(os.path.join(args.clips_dir, "*", "meta.json")))
    if args.limit_clips:
        metas = metas[: args.limit_clips]
    if not metas:
        raise SystemExit(f"no <vid>/meta.json under {args.clips_dir}")
    os.makedirs(args.pkl_out, exist_ok=True)
    if args.pkl_out_raw:
        os.makedirs(args.pkl_out_raw, exist_ok=True)
    if args.dumps_out:
        os.makedirs(args.dumps_out, exist_ok=True)

    pp = load_preproc(args.preproc or os.path.join(os.path.dirname(os.path.abspath(args.engine)),
                                                   "preproc.json"))
    runner = TrtRunner(args.engine)
    in_name = runner.inputs[0]
    warmed = False

    def complete(vid):
        if not os.path.exists(os.path.join(args.pkl_out, vid + ".pkl")):
            return False
        if args.pkl_out_raw and not os.path.exists(os.path.join(args.pkl_out_raw, vid + ".pkl")):
            return False
        return True

    todo = []
    for mp in metas:
        vid = os.path.basename(os.path.dirname(mp))
        if not complete(vid):
            todo.append((mp, vid))
    print(f"[batch] {args.config}: {len(metas)} clips, {len(todo)} to do, "
          f"{len(metas) - len(todo)} already have a pkl", flush=True)

    t_all = time.perf_counter()
    done = frames_done = 0
    skipped = []
    for mp, vid in todo:
        meta = json.load(open(mp))
        fdir = os.path.join(os.path.dirname(mp), "frames")
        frames = sorted(os.path.join(fdir, f) for f in os.listdir(fdir)
                        if f.lower().endswith(EXTS)) if os.path.isdir(fdir) else []
        if not frames:
            print(f"[batch] skip {vid}: no frames on disk", flush=True)
            continue
        if meta.get("n_frames") and len(frames) != meta["n_frames"]:
            print(f"[batch] skip {vid}: {len(frames)} frames but meta says {meta['n_frames']} "
                  f"-- re-fetch it rather than measuring a truncated clip", flush=True)
            continue
        if not warmed:
            x0, _, _ = preprocess(cv2.imread(frames[0]), None, pp)
            for _ in range(args.warmup):
                runner.infer({in_name: x0})
            warmed = True

        sqf = square_fn(meta) if args.square_norm else None
        kps, scs, raw, kps_raw = [], [], [], []
        wh_crop = np.asarray([meta["crop_xywh"][2], meta["crop_xywh"][3]], dtype=np.float32)
        # A single unreadable JPEG used to abort the entire pass: cv2.imread returns None and
        # preprocess dies on None.shape at rtmpose_utils.py:81. On 2026-10-05 that killed a 531-clip
        # J12 batch after 3 clips, and because the driver only checked that SOME pkls appeared it went
        # on to delete the frames. One corrupt frame among 111,388 is a clip to skip, not a run to
        # lose, so the clip is isolated and named and the pass continues.
        bad = None
        for path in frames:
            img = cv2.imread(path)
            if img is None:
                bad = path
                break
            x, center, scale = preprocess(img, None, pp)
            outs = runner.infer({in_name: x})
            k, s = postprocess(outs["simcc_x"].cpu().numpy(), outs["simcc_y"].cpu().numpy(),
                              center, scale, pp)
            if args.dumps_out:
                raw.append({"frame": os.path.basename(path), "keypoints": k.round(2).tolist(),
                            "scores": s.round(4).tolist()})
            kn = sqf(k) if sqf is not None else (k / wh_crop[None]).astype(np.float32)
            kps.append(kn.reshape(1, 133, 2).astype(np.float32))
            scs.append(np.asarray(s, dtype=np.float32).reshape(1, 133))
            if args.pkl_out_raw:
                kps_raw.append((k / wh_crop[None]).astype(np.float32).reshape(1, 133, 2))

        if bad is not None:
            print(f"[batch] SKIP {vid}: unreadable frame {os.path.basename(bad)} "
                  f"(cv2.imread returned None -- truncated or corrupt). Re-push this clip's frames "
                  f"from the pose-track Mac; no pkl written.", flush=True)
            skipped.append(vid)
            continue

        # write to a temp name then rename: a pkl that exists is the resume marker, so it must never
        # exist in a half-written state
        def save(d, payload):
            t = os.path.join(d, vid + ".pkl.part")
            with open(t, "wb") as f:
                pickle.dump(payload, f)
            os.replace(t, os.path.join(d, vid + ".pkl"))

        save(args.pkl_out, {"keypoints": kps, "scores": scs,
                            "square_norm": bool(sqf is not None), "vid": vid})
        if args.pkl_out_raw:
            save(args.pkl_out_raw, {"keypoints": kps_raw, "scores": scs,
                                    "square_norm": False, "vid": vid})
        if args.dumps_out:
            json.dump({"summary": {"n_frames": len(frames)}, "results": raw},
                      open(os.path.join(args.dumps_out, f"{args.config}__{vid}.json"), "w"))

        done += 1
        frames_done += len(frames)
        if done % args.progress_every == 0 or done == len(todo):
            el = time.perf_counter() - t_all
            rate = frames_done / el
            left = len(todo) - done
            print(f"[batch] {done}/{len(todo)} clips  {frames_done} frames  "
                  f"{rate:.1f} frames/s  eta {left * (el / done) / 60:.1f} min", flush=True)

    el = time.perf_counter() - t_all
    print(f"[batch] {args.config} done: {done} clips, {frames_done} frames in {el / 60:.1f} min "
          f"({frames_done / max(el, 1e-9):.1f} frames/s) -> {args.pkl_out}", flush=True)
    if skipped:
        print(f"[batch] {len(skipped)} clip(s) SKIPPED for unreadable frames: "
              f"{', '.join(skipped)}", flush=True)


if __name__ == "__main__":
    main()
