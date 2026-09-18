#!/usr/bin/env python3
"""Uni-Sign mT5-base decoder: plain-PyTorch baseline on the Jetson GPU + power.

Installs beyond the JetPack image:
    pip3 install "transformers>=4.45" sentencepiece protobuf accelerate

    # smoke test with the stock HF weights (downloads ~2.3 GB once):
    python task2_mt5_baseline/01_mt5_torch_baseline.py --model google/mt5-base --seq-len 64
    # with the Uni-Sign fine-tuned weights:
    python task2_mt5_baseline/01_mt5_torch_baseline.py --model google/mt5-base \
        --unisign-ckpt weights/unisign_best.pth --unisign-prefix mt5_model. \
        --seq-len 64 --power-json results/mt5_power_7W.json

What it does:
  1. loads MT5ForConditionalGeneration (fp32 by default; see --dtype),
  2. builds a placeholder encoder input of shape (batch, seq_len, d_model) as
     `inputs_embeds` -- Uni-Sign feeds projected pose features into the mT5
     encoder this way rather than token ids. Replace --seq-len (and --batch)
     with the shape you confirm from the Uni-Sign repo. --mode ids instead
     tokenises --prompt so you can also check real text goes in and out.
  3. runs greedy generate(), decodes text, times encoder-only and full generate,
  4. wraps the timed runs with tegrastats/jtop power logging
     (mJ per generate call and per generated token).

Uni-Sign specifics that NEED VERIFICATION against the repo:
  * attribute prefix of the mT5 weights inside the Uni-Sign checkpoint
    (default guess "mt5_model."); the script prints the prefixes it finds.
  * whether Uni-Sign changes the vocab / tokenizer (resize_token_embeddings).
  * the encoder input shape (T, d_model=768 for mt5-base).

Numerics: T5/mT5 in fp16 is known to overflow (NaN/inf in FF layers). Use fp32
for the baseline; bf16 (Orin is Ampere, bf16 supported) is the safe fast path.
"""
import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.envinfo import collect, print_env  # noqa: E402
from common.power_logger import add_power_args, run_with_power  # noqa: E402

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def load_unisign_weights(model, ckpt_path, prefix):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    for k in ("model", "state_dict", "module"):
        if isinstance(sd, dict) and k in sd and isinstance(sd[k], dict):
            sd = sd[k]
    tops = sorted({k.split(".")[0] for k in sd})
    print(f"[unisign] top-level key prefixes in checkpoint: {tops}")
    sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    if not sub:
        raise SystemExit(f"no keys start with '{prefix}'. Pick one of {tops} (+ '.')")
    # VERIFY: if Uni-Sign resized the vocab, shapes of shared/lm_head will differ here.
    for name in ("shared.weight", "lm_head.weight"):
        if name in sub and sub[name].shape != model.state_dict()[name].shape:
            print(f"[unisign] WARNING {name} shape {tuple(sub[name].shape)} != model "
                  f"{tuple(model.state_dict()[name].shape)} -> resize_token_embeddings needed")
    missing, unexpected = model.load_state_dict(sub, strict=False)
    print(f"[unisign] loaded {len(sub)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print("[unisign]   missing (first 10):", missing[:10])
    if unexpected:
        print("[unisign]   unexpected (first 10):", unexpected[:10])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/mt5-base", help="HF hub id or local dir")
    ap.add_argument("--unisign-ckpt", default=None)
    ap.add_argument("--unisign-prefix", default="mt5_model.")
    ap.add_argument("--mode", choices=["embeds", "ids"], default="embeds")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=64, help="PLACEHOLDER encoder length")
    ap.add_argument("--prompt", default="Translate: Hallo Welt", help="for --mode ids")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--dtype", choices=list(DTYPES), default="fp32")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    add_power_args(ap)
    args = ap.parse_args()
    print_env()

    from transformers import AutoTokenizer, MT5ForConditionalGeneration

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[mt5] loading {args.model} in {args.dtype} on {dev}")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = MT5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=DTYPES[args.dtype])
    if args.unisign_ckpt:
        load_unisign_weights(model, args.unisign_ckpt, args.unisign_prefix)
    model.to(dev).eval()
    cfg = model.config
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[mt5] params={n_params / 1e6:.0f}M d_model={cfg.d_model} layers={cfg.num_layers}/"
          f"{cfg.num_decoder_layers} vocab={cfg.vocab_size}")
    if dev.type == "cuda":
        print(f"[mt5] GPU mem after load: {torch.cuda.memory_allocated() / 2**20:.0f} MB")

    torch.manual_seed(args.seed)
    if args.mode == "embeds":
        # PLACEHOLDER input: replace shape with the confirmed Uni-Sign projection output.
        enc_in = {"inputs_embeds": torch.randn(args.batch, args.seq_len, cfg.d_model,
                                               device=dev, dtype=DTYPES[args.dtype]),
                  "attention_mask": torch.ones(args.batch, args.seq_len, device=dev, dtype=torch.long)}
    else:
        t = tok([args.prompt] * args.batch, return_tensors="pt", padding=True).to(dev)
        enc_in = {"input_ids": t.input_ids, "attention_mask": t.attention_mask}
    print(f"[mt5] encoder input: {[(k, tuple(v.shape)) for k, v in enc_in.items()]}")

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, num_beams=args.num_beams,
                      do_sample=False)

    def sync():
        if dev.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def one_generate():
        sync(); t0 = time.perf_counter()
        out = model.generate(**enc_in, **gen_kwargs)
        sync()
        return out, (time.perf_counter() - t0) * 1e3

    @torch.no_grad()
    def one_encoder():
        sync(); t0 = time.perf_counter()
        model.encoder(**enc_in)
        sync()
        return (time.perf_counter() - t0) * 1e3

    for _ in range(args.warmup):
        out, _ = one_generate()
    text = tok.batch_decode(out, skip_special_tokens=True)
    print(f"[mt5] sample output ids shape {tuple(out.shape)}; text: {text!r}")
    if not torch.isfinite(model(**enc_in, decoder_input_ids=out[:, :1]).logits).all():
        print("[mt5] WARNING: non-finite logits (fp16 overflow?) -> use --dtype fp32/bf16")

    def timed_runs():
        enc_ms, gen_ms, n_tok = [], [], 0
        for _ in range(args.runs):
            enc_ms.append(one_encoder())
            o, ms = one_generate()
            gen_ms.append(ms)
            n_tok += int((o[:, 1:] != cfg.pad_token_id).sum())  # generated tokens excl. start pad
        return {"runs": args.runs, "generated_tokens": n_tok,
                "encoder_ms": {"mean": statistics.mean(enc_ms), "median": statistics.median(enc_ms)},
                "generate_ms": {"mean": statistics.mean(gen_ms), "median": statistics.median(gen_ms),
                                "min": min(gen_ms), "max": max(gen_ms)},
                "ms_per_generated_token": statistics.mean(gen_ms) * args.runs / max(n_tok, 1)}

    lat, power = run_with_power(args, timed_runs, lambda r: r["runs"])
    if "energy_mJ" in power and lat["generated_tokens"]:
        power["mJ_per_generated_token"] = round(power["energy_mJ"] / lat["generated_tokens"], 2)
    power["note"] = "n_frames == generate() calls; mJ_per_frame == mJ per sequence"
    report = {"env": collect(), "model": args.model, "dtype": args.dtype, "mode": args.mode,
              "encoder_input_shape": [args.batch, args.seq_len, cfg.d_model] if args.mode == "embeds"
              else list(enc_in["input_ids"].shape),
              "gen_kwargs": gen_kwargs, "sample_text": text, "latency": lat, "power": power,
              "gpu_mem_peak_MB": torch.cuda.max_memory_allocated() / 2**20 if dev.type == "cuda" else None}
    print(json.dumps(report, indent=2, default=str))
    if args.power_json:
        with open(args.power_json, "w") as f:
            json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()
