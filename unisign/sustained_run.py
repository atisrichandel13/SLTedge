#!/usr/bin/env python3
"""Sustained-load run for the C9 protocol: does the pipeline hold up for 30 minutes at 15 W?

    python -m unisign.sustained_run --mode pose --duration-s 1800 \
        --engine models/rtmw/rtmw-l-m_256x192_fp16.engine \
        --frames data/clips/<vid>/frames --power-json results/c9_sustained_pose.json \
        --power-csv results/c9_sustained_pose.csv --out results/c9_sustained_pose_iters.json

    python -m unisign.sustained_run --mode e2e --duration-s 1800 ... --meta <meta.json> \
        --ckpt weights/openasl_pose_only_slt.pth --mt5 weights/mt5-base

Why this exists. Every latency and energy number in this project comes from a window of 8-25 seconds,
and Tj has never been observed above 53 C. Nothing tells us what happens after half an hour of
continuous work: if the board thermally throttles, latency rises, the governor drops clocks, and every
row in RESULTS.md plus the accuracy-energy frontier shifts with them. C9 asks for one sustained run for
exactly this reason and it has never been done.

The point is DRIFT, not averages. Each iteration records its own wall-clock offset and stage latencies,
and the power logger samples watts, clocks and temperatures every 100 ms into the CSV, so the two can be
joined afterwards. The summary compares the first and last 20 % of iterations: if latency, clock or
temperature move between them, the board is throttling and the short-window rows are optimistic.

--mode pose loops the pose engine over the clip's frames (the stage that runs continuously in
deployment). --mode e2e loops the whole pose->text pipeline, which is the real duty cycle but holds both
models resident; it is the more honest test and the more likely to hit memory trouble over hundreds of
iterations, so run pose first.
"""
import argparse
import json
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "task1_rtmpose"))

from common.envinfo import collect  # noqa: E402
from common.power_logger import add_power_args, run_with_power  # noqa: E402
from common.pose_to_unisign import collate, to_model_inputs  # noqa: E402
from common.trt_runner import TrtRunner  # noqa: E402

import cv2  # noqa: E402
from rtmpose_utils import load_preproc, postprocess, preprocess  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp")
THERM = "/sys/devices/virtual/thermal"


def read_thermals():
    """Tj and friends straight from sysfs, so an iteration carries its own temperature."""
    out = {}
    try:
        for zone in sorted(os.listdir(THERM)):
            if not zone.startswith("thermal_zone"):
                continue
            d = os.path.join(THERM, zone)
            try:
                name = open(os.path.join(d, "type")).read().strip()
                milli = int(open(os.path.join(d, "temp")).read().strip())
                out[name] = round(milli / 1000.0, 2)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pose", "e2e"], default="pose")
    ap.add_argument("--duration-s", type=float, default=1800.0)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--preproc", default=None)
    ap.add_argument("--meta", default=None, help="required for --mode e2e (square-norm geometry)")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--mt5", default=None)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True, help="per-iteration JSON (the drift record)")
    add_power_args(ap)
    args = ap.parse_args()

    frames = sorted(os.path.join(args.frames, f) for f in os.listdir(args.frames)
                    if f.lower().endswith(EXTS))
    if args.limit:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit(f"no frames under {args.frames}")
    pp = load_preproc(args.preproc or os.path.join(os.path.dirname(os.path.abspath(args.engine)),
                                                   "preproc.json"))
    runner = TrtRunner(args.engine)
    in_name = runner.inputs[0]

    model = sqf = None
    if args.mode == "e2e":
        if not (args.ckpt and args.mt5 and args.meta):
            raise SystemExit("--mode e2e needs --ckpt, --mt5 and --meta")
        from openasl_fetch import openasl_square_crop
        from unisign.model import load_model
        m = json.load(open(args.meta))
        W, H = m["source_wh"]
        cx, cy, cw, ch = m["crop_xywh"]
        _, g = openasl_square_crop(m["bbox_norm"], W, H)
        sx, sy, side = g["square_xywh"][0], g["square_xywh"][1], g["square_xywh"][2]

        def sqf(k):  # noqa: E306
            o = np.empty_like(k, dtype=np.float32)
            o[..., 0] = (cx + k[..., 0] - sx) / side
            o[..., 1] = (cy + k[..., 1] - sy) / side
            return o
        t0 = time.perf_counter()
        model = load_model(args.ckpt, args.mt5, device="cuda", dtype=torch.float32)
        print(f"[c9] LM loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    x0, _, _ = preprocess(cv2.imread(frames[0]), None, pp)
    for _ in range(args.warmup):
        runner.infer({in_name: x0})

    def one_iter():
        kps, scs = [], []
        tp = time.perf_counter()
        for path in frames:
            img = cv2.imread(path)
            x, center, scale = preprocess(img, None, pp)
            outs = runner.infer({in_name: x})
            k, s = postprocess(outs["simcc_x"].cpu().numpy(), outs["simcc_y"].cpu().numpy(),
                               center, scale, pp)
            if model is not None:
                kps.append((sqf(k)).reshape(1, 133, 2))
                scs.append(np.asarray(s, dtype=np.float32).reshape(1, 133))
        pose_ms = (time.perf_counter() - tp) * 1e3
        lm_ms = 0.0
        if model is not None:
            inputs, _ = to_model_inputs(kps, scs, args.max_length)
            src = collate([inputs], ["clip"])
            tl = time.perf_counter()
            model.translate(src, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            lm_ms = (time.perf_counter() - tl) * 1e3
        return pose_ms, lm_ms

    print(f"[c9] mode={args.mode} target {args.duration_s:.0f}s, {len(frames)} frames/iteration",
          flush=True)
    one_iter()  # warm-up iteration, not recorded

    iters = []

    def job():
        t_start = time.perf_counter()
        i = 0
        while time.perf_counter() - t_start < args.duration_s:
            th_before = read_thermals()
            pose_ms, lm_ms = one_iter()
            th = read_thermals()
            i += 1
            rec = {"i": i, "t_rel_s": round(time.perf_counter() - t_start, 2),
                   "pose_ms": round(pose_ms, 1), "lm_ms": round(lm_ms, 1),
                   "total_ms": round(pose_ms + lm_ms, 1),
                   "tj_C": th.get("tj-thermal", th.get("TJ", max(th.values()) if th else -1)),
                   "thermals": th}
            iters.append(rec)
            if i % 10 == 0 or i == 1:
                print(f"[c9] iter {i:4d} t={rec['t_rel_s']:7.1f}s total={rec['total_ms']:8.1f} ms "
                      f"Tj={rec['tj_C']}C", flush=True)
        return {"n_iters": i, "n_frames_total": i * len(frames)}

    res, power = run_with_power(args, job, lambda s: s["n_frames_total"])

    # Throttling check: compare the first and last fifth of the run. Averages hide drift, which is the
    # only thing a sustained run is for.
    k = max(1, len(iters) // 5)
    first, last = iters[:k], iters[-k:]

    def seg(rs):
        tj = [r["tj_C"] for r in rs if isinstance(r["tj_C"], (int, float)) and r["tj_C"] > 0]
        return {"n": len(rs),
                "total_ms_mean": round(statistics.mean([r["total_ms"] for r in rs]), 1),
                "pose_ms_mean": round(statistics.mean([r["pose_ms"] for r in rs]), 1),
                "lm_ms_mean": round(statistics.mean([r["lm_ms"] for r in rs]), 1),
                "tj_C_mean": round(statistics.mean(tj), 2) if tj else None,
                "tj_C_max": round(max(tj), 2) if tj else None}

    a, b = seg(first), seg(last)
    drift_pct = (100.0 * (b["total_ms_mean"] - a["total_ms_mean"]) / a["total_ms_mean"]
                 if a["total_ms_mean"] else 0.0)
    all_tj = [r["tj_C"] for r in iters if isinstance(r["tj_C"], (int, float)) and r["tj_C"] > 0]
    summary = {
        "mode": args.mode, "duration_s": round(iters[-1]["t_rel_s"], 1) if iters else 0,
        "n_iters": res["n_iters"], "frames_per_iter": len(frames),
        "first_fifth": a, "last_fifth": b,
        "total_ms_drift_pct": round(drift_pct, 2),
        "tj_C_max_overall": round(max(all_tj), 2) if all_tj else None,
        "avg_W": round(power["avg_watts"], 2), "idle_W": round(power["baseline_watts"], 2),
        "mJ_per_frame": round(power["mJ_per_frame"], 1),
        "gpu_MHz_mean": round(power["aux_avg"].get("gpu_MHz", -1), 0),
        "cpu0_MHz_mean": round(power["aux_avg"].get("cpu0_MHz", -1), 0),
        "throttled": bool(drift_pct > 5.0),
    }
    print("\n[c9] " + json.dumps(summary, indent=2))
    print(f"[c9] VERDICT: latency drift {drift_pct:+.2f} % first-fifth -> last-fifth, "
          f"Tj max {summary['tj_C_max_overall']} C -> "
          f"{'THROTTLING' if summary['throttled'] else 'no throttling detected'}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"env": collect(), "summary": summary, "iters": iters, "power": power,
               "config": vars(args)}, open(args.out, "w"), indent=2)
    print("[c9] wrote", args.out)


if __name__ == "__main__":
    main()
