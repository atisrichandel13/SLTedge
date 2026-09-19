#!/usr/bin/env python3
"""Keypoints -> text with the released Uni-Sign pose-only checkpoint. Interface C5.

    python -m unisign.unisign_infer --keypoints results/rtmpose_trt_fp32.json --crop-wh 666 720 \
        --ckpt weights/openasl_pose_only_slt.pth --mt5 weights/mt5-base --out results/translation.json
    python -m unisign.unisign_infer --pkl results/rtmlib_lightweight_Ads-4j06eJY.pkl ...

Plain PyTorch. Runs on CPU or CUDA. No mmpose / mmcv / deepspeed / decord.
Output JSON: {"text", "tokens", "token_logprobs", "timing_ms": {gcn, encoder, decoder, total,
encoder_len, decoded_tokens}, "config": {...}}. `--repeat N` re-runs generation N times after a
warm-up and reports the per-run timings. `--power-json/--power-csv` (Jetson, queue J3) wrap the N
timed runs in the INA3221 power logger; the summary's "per frame" numbers are then per sentence.
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.pose_to_unisign import collate, load_keypoints_json, load_pkl, to_model_inputs  # noqa: E402
from common.power_logger import add_power_args, run_with_power  # noqa: E402
from unisign.model import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--keypoints", help="03_infer_frames.py JSON (pixel coords)")
    g.add_argument("--pkl", help="Uni-Sign pkl (normalised coords)")
    ap.add_argument("--crop-wh", type=int, nargs=2, default=[666, 720])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mt5", required=True, help="mt5-base directory")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--w8-runtime", default="dequant", choices=["dequant", "int8"], help="for W8 checkpoints (unisign.quant)")
    ap.add_argument("--max-length", type=int, default=256, help="max pose frames fed to the model")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default=None)
    add_power_args(ap)
    args = ap.parse_args()

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    t0 = time.perf_counter()
    model = load_model(args.ckpt, args.mt5, device=args.device, dtype=dtype, w8_runtime=args.w8_runtime)
    load_s = time.perf_counter() - t0
    n_params = sum(p.numel() for p in model.parameters())
    n_lm = sum(p.numel() for p in model.mt5_model.parameters())
    print(f"[infer] model loaded in {load_s:.1f}s on {args.device}/{args.dtype}: "
          f"{n_params/1e6:.1f}M params, mT5 {n_lm/1e6:.1f}M, pose stack {(n_params-n_lm)/1e6:.2f}M")

    if args.keypoints:
        kps, scs, _ = load_keypoints_json(args.keypoints, args.crop_wh)
    else:
        kps, scs, _ = load_pkl(args.pkl)
    inputs, idx = to_model_inputs(kps, scs, args.max_length)
    src = collate([inputs], ["clip"])
    print(f"[infer] frames in={len(scs)} used={len(idx)}")

    r = model.translate(src, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
    print(f"[infer] warm-up: {r['timing_ms']['total']:.0f} ms")

    def timed_runs():
        runs = []
        for i in range(1, args.repeat + 1):
            r = model.translate(src, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)
            runs.append(r)
            print(f"[infer] run {i}: {json.dumps({k: round(v, 1) for k, v in r['timing_ms'].items()})}")
        return runs

    power = None
    if args.power_json or args.power_csv:
        runs, power = run_with_power(args, timed_runs, lambda rs: len(rs))  # "frame" = one sentence
        toks = sum(x["timing_ms"]["decoded_tokens"] for x in runs)
        power["n_sentences"] = len(runs); power["n_tokens"] = toks
        power["J_per_sentence"] = round(power["energy_mJ"] / len(runs) / 1000, 3)
        if "dynamic_energy_mJ" in power:
            power["dynamic_J_per_sentence"] = round(power["dynamic_energy_mJ"] / len(runs) / 1000, 3)
        power["mJ_per_token"] = round(power["energy_mJ"] / max(1, toks), 1)
        print(f"[power] {power.get('avg_watts')} W avg, {power['J_per_sentence']} J/sentence, "
              f"{power['mJ_per_token']} mJ/token over {len(runs)} runs")
        if args.power_json:
            json.dump(power, open(args.power_json, "w"), indent=2)  # re-write with the per-sentence keys
    else:
        runs = timed_runs()
    r = runs[-1]
    print(f"[infer] text: {r['text']}")
    if r["token_logprobs"]:
        lp = r["token_logprobs"]
        print(f"[infer] {len(lp)} tokens, mean logprob {sum(lp)/len(lp):.3f}, min {min(lp):.3f}")
    if args.device == "cuda":
        print(f"[infer] peak GPU mem {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump({**r, "runs_timing_ms": [x["timing_ms"] for x in runs], "power": power,
                   "config": {**vars(args), "load_s": load_s, "n_params": n_params, "n_lm_params": n_lm,
                              "frames_used": len(idx), "torch": torch.__version__}},
                  open(args.out, "w"), indent=2)
        print("[infer] wrote", args.out)


if __name__ == "__main__":
    main()
