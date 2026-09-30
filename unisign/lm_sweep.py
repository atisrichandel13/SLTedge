#!/usr/bin/env python3
"""LM latency and power on the board across decode width and encoder length, in ONE process.

    python -m unisign.lm_sweep --ckpt weights/openasl_pose_only_slt_pruned.pth \
        --mt5 weights/mt5-base-openasl-pruned --poses results/pkl_30clip_rtmw_fp16_sqnorm \
        --beams 1 2 4 --lengths 256 205 137 103 68 --repeat 3 --out results/lm_sweep_pruned.json

This is the energy axis the accuracy-energy frontier needs. Its accuracy axis is already measured on
976 clips with established signs (RESULTS.md L6.1: beam 4 22.87, beam 2 22.06 at -0.81 CI
[-1.29, -0.39], greedy 20.88 at -2.00 CI [-2.63, -1.41]), so beam width is the knob and only J per
sentence was missing.

One process for the whole sweep because loading the model costs ~64 s on this board (M1, RESULTS.md
5.1): fifteen configurations launched separately would spend 16 minutes doing nothing but loading.
The model is loaded once, outside every power window, and each configuration gets its own window with
its own idle baseline so the energies stay comparable.

`--lengths` are pose frame counts fed to the encoder, i.e. the L7 sweep: 256 / 205 / 137 / 103 / 68
correspond to 30 / 24 / 16 / 12 / 8 fps-equivalent for a ~8.5 s sentence. A length longer than the clip
is silently the whole clip, so the row records `frames_used` as well as the requested length.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.envinfo import collect  # noqa: E402
from common.power_logger import add_power_args, run_with_power  # noqa: E402
from common.pose_to_unisign import collate, load_pkl, to_model_inputs  # noqa: E402
from unisign.model import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mt5", required=True)
    ap.add_argument("--poses", required=True, help="dir of <clip>.pkl; the first N are used")
    ap.add_argument("--n-clips", type=int, default=3, help="clips averaged per configuration")
    ap.add_argument("--beams", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--lengths", type=int, nargs="+", default=[256, 205, 137, 103, 68])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=3, help="timed passes per configuration (C9 wants 3)")
    ap.add_argument("--out", required=True)
    add_power_args(ap)
    args = ap.parse_args()

    pkls = sorted(p for p in os.listdir(args.poses) if p.endswith(".pkl"))[: args.n_clips]
    if not pkls:
        raise SystemExit(f"no .pkl under {args.poses}")
    raw = [load_pkl(os.path.join(args.poses, p))[:2] for p in pkls]
    print(f"[sweep] {len(pkls)} clips, beams {args.beams}, lengths {args.lengths}, "
          f"{args.repeat} repeats -> {len(args.beams) * len(args.lengths)} configurations")

    t0 = time.perf_counter()
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model = load_model(args.ckpt, args.mt5, device=args.device, dtype=dtype)
    print(f"[sweep] model loaded in {time.perf_counter() - t0:.1f}s "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")

    # one warm-up pass so no configuration pays lazy CUDA/cuDNN init
    kps, scs = raw[0]
    inputs, _ = to_model_inputs(kps, scs, args.lengths[0])
    model.translate(collate([inputs], ["w"]), max_new_tokens=args.max_new_tokens, num_beams=1)

    rows = []
    for length in args.lengths:
        prepared = []
        for (kps, scs), name in zip(raw, pkls):
            inputs, idx = to_model_inputs(kps, scs, length)
            prepared.append((collate([inputs], [name]), len(idx)))
        for beams in args.beams:
            def job(_src=prepared, _b=beams):
                per = []
                for _ in range(args.repeat):
                    for src, _n in _src:
                        r = model.translate(src, max_new_tokens=args.max_new_tokens, num_beams=_b)
                        per.append(r["timing_ms"])
                return {"timings": per, "n_sentences": len(per)}

            res, power = run_with_power(args, job, lambda s: s["n_sentences"])
            per = res["timings"]
            tot = [t["total"] for t in per]
            row = {
                "beams": beams, "requested_length": length,
                "frames_used": [n for _, n in prepared],
                "n_sentences": res["n_sentences"],
                "total_ms": {"mean": round(statistics.mean(tot), 1),
                             "std": round(statistics.pstdev(tot), 1) if len(tot) > 1 else 0.0},
                "gcn_ms": round(statistics.mean([t["gcn"] for t in per]), 1),
                "encoder_ms": round(statistics.mean([t["encoder"] for t in per]), 1),
                "decoder_ms": round(statistics.mean([t["decoder"] for t in per]), 1),
                "decoded_tokens": round(statistics.mean([t.get("decoded_tokens", 0) for t in per]), 1),
                "avg_W": round(power["avg_watts"], 2), "idle_W": round(power["baseline_watts"], 2),
                "J_per_sentence": round(power["energy_mJ"] / 1e3 / res["n_sentences"], 2),
                "dyn_J_per_sentence": round(power["dynamic_energy_mJ"] / 1e3 / res["n_sentences"], 2),
                "gpu_MHz": round(power["aux_avg"].get("gpu_MHz", -1), 0),
                "cpu0_MHz": round(power["aux_avg"].get("cpu0_MHz", -1), 0),
            }
            rows.append(row)
            print(f"[sweep] beams={beams} len={length} used={row['frames_used'][0]}: "
                  f"{row['total_ms']['mean']:.0f} ms, {row['J_per_sentence']:.2f} J/sentence, "
                  f"{row['avg_W']:.2f} W, gpu {row['gpu_MHz']:.0f} MHz", flush=True)

    print(f"\n{'beams':>5s} {'len':>4s} {'used':>5s} {'ms':>7s} {'J/sent':>7s} {'dynJ':>6s} "
          f"{'W':>5s} {'gpuMHz':>7s}")
    for r in rows:
        print(f"{r['beams']:5d} {r['requested_length']:4d} {r['frames_used'][0]:5d} "
              f"{r['total_ms']['mean']:7.0f} {r['J_per_sentence']:7.2f} {r['dyn_J_per_sentence']:6.2f} "
              f"{r['avg_W']:5.2f} {r['gpu_MHz']:7.0f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump({"env": collect(), "rows": rows, "clips": pkls, "config": vars(args)},
              open(args.out, "w"), indent=2)
    print("[sweep] wrote", args.out)


if __name__ == "__main__":
    main()
