#!/usr/bin/env python3
"""Frames -> English in ONE process on the board: the C6 / M1 end-to-end baseline.

    python -m unisign.e2e_translate \
        --engine models/rtmw/rtmw-l-m_256x192_fp16.engine \
        --frames data/clips/<vid>/frames --meta data/clips/<vid>/meta.json \
        --ckpt weights/openasl_pose_only_slt.pth --mt5 weights/mt5-base \
        --square-norm --repeat 3 --power-json results/m1_e2e.json

Why this exists separately from `unisign_infer.py`. That script starts from a keypoints JSON someone
else produced; every latency and energy number in the project so far is per-stage, measured with the
board otherwise idle and one model resident. This runs the pose engine and the language model in the
same process so the two stages share the 15 W cap and the 8 GB, which is the only configuration that
answers "does the pipeline hold together on the device".

It exists to be compared against the sum of the separate measurements. §2.2b of RESULTS.md found that
CPU-side stages slow by x1.281 when the GPU takes more of the 15 W budget, so the prediction is that
end-to-end is WORSE than pose-alone plus LM-alone. Reporting the sum without checking would overstate
the system.

Stage timing (all per sentence unless marked per frame):
  load_pose_s / load_lm_s   outside the power window, reported separately
  imread / preprocess / trt / postprocess    per frame, same meaning as 03_infer_frames.py
  convert_ms                keypoints -> Uni-Sign part tensors (common/pose_to_unisign.py)
  gcn / encoder / decoder   from model.translate, the LM's own breakdown
  pose_total_ms             sum over frames of the four per-frame stages
  sentence_total_ms         pose_total + convert + LM, i.e. frames in -> text out

--square-norm applies the coordinate-frame correction from RESULTS.md 2.5d. The frozen encoder consumes
the 9 body joints in absolute normalised coordinates, and our bbox crop makes them ~1.68x larger than
the square frame the checkpoint was trained on. Without this the deployed pipeline reproduces the
mismatch the offline results corrected; with it, the pose->text path matches what 2.5c measured. It
needs --meta for the clip's bbox and source resolution. Defaults ON for that reason; --no-square-norm
measures the uncorrected path for comparison.
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
from unisign.model import load_model  # noqa: E402

import cv2  # noqa: E402
from rtmpose_utils import load_preproc, postprocess, preprocess  # noqa: E402

from common.subsample_pkl import keep_idx  # noqa: E402

EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_frames(d, limit=None, keep_fps=None, meta_path=None, src_fps=None):
    """Frames on disk, optionally thinned to simulate a lower capture rate.

    The source rate comes from the clip's own meta.json as frames / duration, NOT from a constant.
    OpenASL is not one frame rate (76.5 % ~30, 22.2 % ~24, a few 59.94), and assuming 29.97 for
    everything is what invalidated the first accuracy-vs-rate curve on both tracks -- a 24 fps clip
    labelled "16 fps" was really thinned to 12.8. See RESULTS.md 2.9B.

    Selection is delegated to common.subsample_pkl.keep_idx so the energy path here and the accuracy
    path there cannot drift apart: a rate means the same set of frames in both.
    """
    fs = sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(EXTS))
    if not fs:
        raise SystemExit(f"no frames under {d}")
    if limit:
        fs = fs[:limit]
    info = {"keep_fps": None, "src_fps": None, "frames_in": len(fs), "frames_kept": len(fs)}
    if not keep_fps:
        return fs, info
    src = src_fps
    if src is None:
        if not meta_path:
            raise SystemExit("--keep-fps needs --meta (or an explicit --src-fps) to read the clip's "
                             "own frame rate; a hardcoded rate is the bug this flag exists to avoid")
        m = json.load(open(meta_path))
        src = (m["n_frames"] / m["duration_s"]) if m.get("duration_s") else float(m["fps"])
    idx = keep_idx(len(fs), keep_fps, src)
    fs = [fs[i] for i in idx]
    info.update(keep_fps=float(keep_fps), src_fps=round(float(src), 3), frames_kept=len(fs))
    return fs, info


def square_norm_fn(meta_path):
    """-> f(kpts_px) mapping source-crop pixels into the authors' square frame, or None."""
    from openasl_fetch import openasl_square_crop
    m = json.load(open(meta_path))
    if m.get("crop_style", "native") != "native":
        raise SystemExit(f"--square-norm expects a native-crop clip; {meta_path} says "
                         f"crop_style={m.get('crop_style')}")
    W, H = m["source_wh"]
    cx, cy, cw, ch = m["crop_xywh"]
    _, g = openasl_square_crop(m["bbox_norm"], W, H)
    sx, sy, side = g["square_xywh"][0], g["square_xywh"][1], g["square_xywh"][2]

    def f(k):  # k: (133,2) in crop pixels
        out = np.empty_like(k, dtype=np.float32)
        out[..., 0] = (cx + k[..., 0] - sx) / side
        out[..., 1] = (cy + k[..., 1] - sy) / side
        return out
    return f, {"crop_xywh": [cx, cy, cw, ch], "square_xywh": [sx, sy, side],
               "scale_vs_crop": cw / side}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--meta", default=None, help="clip meta.json; required for --square-norm")
    ap.add_argument("--preproc", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mt5", required=True)
    ap.add_argument("--square-norm", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--crop-wh", type=int, nargs=2, default=None,
                    help="with --no-square-norm: divide pixels by this (default: the frame size)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--repeat", type=int, default=1, help="timed sentences after one warm-up")
    ap.add_argument("--keep-fps", type=float, default=None,
                    help="thin frames to this capture rate; the source rate is read per clip from "
                         "--meta as frames/duration, never assumed")
    ap.add_argument("--src-fps", type=float, default=None,
                    help="override the clip's own rate; normally leave unset")
    ap.add_argument("--out", default=None)
    add_power_args(ap)
    args = ap.parse_args()

    if args.square_norm and not args.meta:
        raise SystemExit("--square-norm needs --meta (the clip's bbox and source resolution)")

    frames, fps_info = list_frames(args.frames, args.limit, keep_fps=args.keep_fps,
                                   meta_path=args.meta, src_fps=args.src_fps)
    pp = load_preproc(args.preproc or os.path.join(os.path.dirname(os.path.abspath(args.engine)),
                                                   "preproc.json"))
    sqf, geom = (square_norm_fn(args.meta) if args.square_norm else (None, None))

    # Both models are loaded BEFORE the power window: this is the deployed configuration (pose engine
    # and LM resident together, sharing the 15 W cap), and load cost is a startup number, not a
    # per-sentence one.
    t0 = time.perf_counter()
    runner = TrtRunner(args.engine)
    load_pose_s = time.perf_counter() - t0
    in_name = runner.inputs[0]

    t0 = time.perf_counter()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = load_model(args.ckpt, args.mt5, device=args.device, dtype=dtype)
    load_lm_s = time.perf_counter() - t0
    print(f"[m1] pose engine {load_pose_s:.1f}s, LM {load_lm_s:.1f}s, {len(frames)} frames, "
          f"square_norm={args.square_norm}")
    if args.keep_fps:
        print(f"[m1] rate {fps_info['src_fps']} -> {fps_info['keep_fps']} fps: "
              f"{fps_info['frames_in']} -> {fps_info['frames_kept']} frames")

    x0, _, _ = preprocess(cv2.imread(frames[0]), None, pp)
    for _ in range(args.warmup):
        runner.infer({in_name: x0})

    def one_sentence(pl=None, tag=None):
        """pl/tag are the per-stage power windows. Marks are named <tag>_<stage>_{0,1} and are
        carved out of the same sample stream afterwards, so a stage's joules are integrated from
        its own samples rather than apportioned from the sentence total by latency share -- the
        stages draw different power, so the apportionment would be wrong."""
        def mk(name):
            if pl is not None and tag is not None:
                pl.mark(f"{tag}_{name}")
        lat = {"imread_ms": [], "preprocess_ms": [], "trt_ms": [], "postprocess_ms": []}
        kps, scs = [], []
        mk("pose_0")
        t_pose0 = time.perf_counter()
        for path in frames:
            tr0 = time.perf_counter()
            img = cv2.imread(path)
            t1 = time.perf_counter()
            x, center, scale = preprocess(img, None, pp)
            t2 = time.perf_counter()
            outs = runner.infer({in_name: x})
            t3 = time.perf_counter()
            sx = outs["simcc_x"].cpu().numpy()
            sy = outs["simcc_y"].cpu().numpy()
            k, s = postprocess(sx, sy, center, scale, pp)
            t4 = time.perf_counter()
            lat["imread_ms"].append((t1 - tr0) * 1e3)
            lat["preprocess_ms"].append((t2 - t1) * 1e3)
            lat["trt_ms"].append((t3 - t2) * 1e3)
            lat["postprocess_ms"].append((t4 - t3) * 1e3)
            if sqf is not None:
                kn = sqf(k)
            else:
                wh = np.asarray(args.crop_wh or [img.shape[1], img.shape[0]], dtype=np.float32)
                kn = (k / wh[None]).astype(np.float32)
            kps.append(kn.reshape(1, 133, 2))
            scs.append(np.asarray(s, dtype=np.float32).reshape(1, 133))
        pose_total_ms = (time.perf_counter() - t_pose0) * 1e3
        mk("pose_1")

        mk("convert_0")
        t_c0 = time.perf_counter()
        inputs, idx = to_model_inputs(kps, scs, args.max_length)
        src = collate([inputs], ["clip"])
        convert_ms = (time.perf_counter() - t_c0) * 1e3
        mk("convert_1")

        mk("lm_0")
        r = model.translate(src, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
        mk("lm_1")
        out = {"text": r["text"], "lm_ms": r["timing_ms"], "convert_ms": convert_ms,
               "pose_total_ms": pose_total_ms, "frames_in": len(frames), "frames_used": len(idx),
               "sentence_total_ms": pose_total_ms + convert_ms + r["timing_ms"]["total"]}
        for k, v in lat.items():
            out[k] = {"mean": round(statistics.mean(v), 3), "p95": round(float(np.percentile(v, 95)), 3)}
        return out

    print("[m1] warm-up sentence ...")
    w = one_sentence()
    print(f"[m1] warm-up {w['sentence_total_ms']:.0f} ms -> {w['text']!r}")

    plbox = {}

    def job(pl):
        plbox["pl"] = pl          # the logger outlives the with-block; its samples/marks persist
        runs = []
        for i in range(args.repeat):
            r = one_sentence(pl, f"r{i}")
            runs.append(r)
            print(f"[m1] run {i+1}: pose {r['pose_total_ms']:.0f} + convert {r['convert_ms']:.1f} + "
                  f"LM {r['lm_ms']['total']:.0f} = {r['sentence_total_ms']:.0f} ms")
        return {"runs": runs, "n_frames_total": sum(r["frames_in"] for r in runs),
                "n_sentences": len(runs)}

    res, power = run_with_power(args, job, lambda s: s["n_frames_total"], pass_logger=True)
    runs = res["runs"]
    tot = [r["sentence_total_ms"] for r in runs]
    # energy per SENTENCE is what the accuracy-energy frontier needs; mJ_per_frame from the logger is
    # the pose-stage basis and the two must not be mixed
    j_per_sentence = power["energy_mJ"] / 1e3 / res["n_sentences"]
    dyn_j_per_sentence = power["dynamic_energy_mJ"] / 1e3 / res["n_sentences"]

    # --- per-stage energy (LM track ask, ASK-E2E-KNEE 2026-10-03) ---------------------------
    # The composition the frontier uses is pose_J_per_s * seconds + LM_J, so checking it needs the
    # two terms separately, not one per-sentence total. These windows are integrated from the
    # stage's own samples. Dividing the sentence total by latency share would be wrong: the pose
    # stage is TRT/GPU-bound and the decoder host-bound, so their average power differs.
    #
    # Read `n_samples` before trusting a stage: at a 100 ms interval a 1.4 s LM stage gets ~14
    # samples and the window can miss up to one interval at each edge, so run with a smaller
    # --power-interval-ms when the per-stage split is the point of the run.
    pl = plbox.get("pl")
    stage_power = None
    if pl is not None:
        base = ("idle_start", "idle_end") if args.idle_seconds > 0 else None
        stage_power = {}
        for st in ("pose", "convert", "lm"):
            per = []
            for i in range(res["n_sentences"]):
                w = (f"r{i}_{st}_0", f"r{i}_{st}_1")
                if w[0] in pl.marks and w[1] in pl.marks:
                    per.append(pl.summarize(window=w, baseline=base))
            ok = [x for x in per if "error" not in x]
            if not ok:
                continue
            stage_power[st] = {
                "n_windows": len(ok),
                "J_per_sentence": round(statistics.mean(x["energy_mJ"] for x in ok) / 1e3, 3),
                "dyn_J_per_sentence": round(statistics.mean(
                    x.get("dynamic_energy_mJ", float("nan")) for x in ok) / 1e3, 3),
                "avg_watts": round(statistics.mean(x["avg_watts"] for x in ok), 3),
                "duration_s": round(statistics.mean(x["duration_s"] for x in ok), 3),
                "n_samples_per_window": round(statistics.mean(x["n_samples"] for x in ok), 1),
            }
        # Auditable arithmetic: the three stages should reconstruct the sentence total. They will
        # not match exactly -- each window loses up to one sample interval at each edge, and the
        # gaps between stages belong to neither -- so the residual is reported, not hidden.
        if stage_power:
            ssum = sum(v["J_per_sentence"] for v in stage_power.values())
            stage_power["_check"] = {
                "stage_sum_J_per_sentence": round(ssum, 3),
                "window_J_per_sentence": round(j_per_sentence, 3),
                "residual_J": round(j_per_sentence - ssum, 3),
                "residual_pct": round(100.0 * (j_per_sentence - ssum) / j_per_sentence, 2),
                "note": ("residual is edge quantisation plus the inter-stage gaps; compare it "
                         "against n_samples_per_window before reading anything into a stage"),
            }

    summary = {
        "text": runs[-1]["text"],
        "n_sentences": res["n_sentences"], "frames_in": runs[-1]["frames_in"],
        "frames_used": runs[-1]["frames_used"],
        "sentence_total_ms": {"mean": round(statistics.mean(tot), 1),
                              "std": round(statistics.pstdev(tot), 1) if len(tot) > 1 else 0.0,
                              "runs": [round(x, 1) for x in tot]},
        "pose_total_ms": round(statistics.mean([r["pose_total_ms"] for r in runs]), 1),
        "convert_ms": round(statistics.mean([r["convert_ms"] for r in runs]), 2),
        "lm_total_ms": round(statistics.mean([r["lm_ms"]["total"] for r in runs]), 1),
        "per_frame_ms": {k: runs[-1][k] for k in
                         ("imread_ms", "preprocess_ms", "trt_ms", "postprocess_ms")},
        "J_per_sentence": round(j_per_sentence, 2),
        "dyn_J_per_sentence": round(dyn_j_per_sentence, 2),
        "square_norm": args.square_norm, "square_geom": geom,
        "fps": fps_info, "num_beams": args.num_beams,
        "load_pose_s": round(load_pose_s, 2), "load_lm_s": round(load_lm_s, 2),
    }
    if args.device == "cuda":
        summary["peak_gpu_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    print("\n[m1] " + json.dumps({k: v for k, v in summary.items() if k != "square_geom"}, indent=2))
    print(f"[m1] text: {summary['text']!r}")
    if stage_power:
        ck = stage_power.get("_check", {})
        for st in ("pose", "convert", "lm"):
            v = stage_power.get(st)
            if v:
                print(f"[stage] {st:8s} {v['J_per_sentence']:7.3f} J/sentence  "
                      f"{v['avg_watts']:5.3f} W  {v['duration_s']:6.3f} s  "
                      f"n={v['n_samples_per_window']:.0f} samples/window")
        if ck:
            print(f"[stage] sum {ck['stage_sum_J_per_sentence']:.3f} J vs window "
                  f"{ck['window_J_per_sentence']:.3f} J  "
                  f"residual {ck['residual_J']:+.3f} J ({ck['residual_pct']:+.2f}%)")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({"env": collect(), "summary": summary, "runs": runs, "power": power,
                   "stage_power": stage_power,
                   "config": vars(args)}, open(args.out, "w"), indent=2)
        print("[m1] wrote", args.out)


if __name__ == "__main__":
    main()
