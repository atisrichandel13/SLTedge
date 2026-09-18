#!/usr/bin/env python3
"""Greedy decode with the mT5 TensorRT engines on the Jetson (L8 board half, queue J4).

Same loop as unisign/onnx_decode.py, engines instead of ORT sessions. Verified nowhere yet: run it
on the board and paste the [trt]/[check] lines. PASS for FP32 engines = tokens identical to PyTorch.
For FP16 engines expect a few token flips on low-margin steps; report first divergence + logprob diff.

    python3 -m unisign.trt_decode --engine-dir models/mt5_pruned_onnx \
        --ckpt weights/openasl_pose_only_slt_pruned.pth --mt5 weights/mt5-base \
        --pkl data/openasl_ref_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl [--fp16] [--repeat 3]

Engines come from task3_mt5_onnx/03_build_engines.py: <engine-dir>/{encoder,decoder_init,decoder_step}_{fp32|fp16}.engine
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.pose_to_unisign import collate, load_keypoints_json, load_pkl, to_model_inputs  # noqa: E402
from common.trt_runner import TrtRunner  # noqa: E402
from unisign.model import load_model  # noqa: E402


def decode_once(enc, init, step, emb, mask, max_new_tokens, eos=1):
    torch.cuda.synchronize(); t0 = time.perf_counter()
    enc_out = enc.infer({enc.inputs[0]: emb, enc.inputs[1]: mask})
    hidden = enc_out[enc.outputs[0]]
    t_enc = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    ids = np.zeros((1, 1), dtype=np.int64)
    present = init.infer({"input_ids": ids, "encoder_hidden_states": hidden, "encoder_attention_mask": mask})
    present = {k: v.clone() for k, v in present.items()}  # runner reuses output buffers
    logits = present.pop("logits")
    tokens, logprobs = [], []
    for _ in range(max_new_tokens):
        last = logits[0, -1].float()
        lp = torch.log_softmax(last, -1)
        nxt = int(last.argmax())
        tokens.append(nxt); logprobs.append(float(lp[nxt]))
        if nxt == eos:
            break
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64), "encoder_attention_mask": mask}
        for n in step.inputs:
            if n.startswith("past_key_values."):
                feed[n] = present[n.replace("past_key_values.", "present.")]
        out = step.infer(feed)
        out = {k: v.clone() for k, v in out.items()}
        logits = out.pop("logits")
        present.update(out)
    torch.cuda.synchronize()
    t_dec = (time.perf_counter() - t0) * 1000
    return tokens, logprobs, t_enc, t_dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--fp16", action="store_true", help="use *_fp16.engine files")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mt5", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pkl"); g.add_argument("--keypoints")
    ap.add_argument("--crop-wh", type=int, nargs=2, default=[666, 720])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    model = load_model(args.ckpt, args.mt5, device="cuda")  # PyTorch reference on the same board
    kps, scs, _ = load_pkl(args.pkl) if args.pkl else load_keypoints_json(args.keypoints, args.crop_wh)
    src = collate([to_model_inputs(kps, scs, 256)[0]], ["clip"])
    ref = model.translate(src, max_new_tokens=args.max_new_tokens, num_beams=1)
    with torch.no_grad():
        emb, mask = model.build_encoder_inputs({k: (v.cuda() if torch.is_tensor(v) else v) for k, v in src.items()})
    emb, mask = emb.float().contiguous(), mask.long().contiguous()
    torch_mem = torch.cuda.max_memory_allocated() / 1e9

    tag = "fp16" if args.fp16 else "fp32"
    enc = TrtRunner(os.path.join(args.engine_dir, f"encoder_{tag}.engine"))
    init = TrtRunner(os.path.join(args.engine_dir, f"decoder_init_{tag}.engine"))
    step = TrtRunner(os.path.join(args.engine_dir, f"decoder_step_{tag}.engine"))
    runs = []
    for i in range(args.repeat + 1):
        tokens, logprobs, t_enc, t_dec = decode_once(enc, init, step, emb, mask, args.max_new_tokens)
        if i == 0:
            print(f"[trt] warm-up enc {t_enc:.0f} ms dec {t_dec:.0f} ms")
            continue
        runs.append({"encoder_ms": t_enc, "decoder_ms": t_dec, "tokens": len(tokens)})
        print(f"[trt] {tag} run {i}: encoder {t_enc:.0f} ms, decoder {t_dec:.0f} ms for {len(tokens)} tokens ({t_dec/max(1,len(tokens)):.1f} ms/token)")
    text = model.mt5_tokenizer.decode(torch.tensor([0] + tokens), skip_special_tokens=True)
    print(f"[trt] text  : {text}")
    print(f"[torch] text: {ref['text']}  (torch dec {ref['timing_ms']['decoder']:.0f} ms, enc {ref['timing_ms']['encoder']:.0f} ms)")
    same = tokens == ref["tokens"]
    n = min(len(tokens), len(ref["tokens"]))
    first = next((i for i in range(n) if tokens[i] != ref["tokens"][i]), None)
    print(f"[check] tokens identical: {same}" + ("" if same else f"  first divergence step {first}, lens {len(tokens)} vs {len(ref['tokens'])}"))
    if ref["token_logprobs"]:
        d = max(abs(a - b) for a, b in zip(logprobs, ref["token_logprobs"]))
        print(f"[check] max |logprob diff| over shared steps: {d:.2e}")
    print(f"[mem] torch peak {torch_mem:.2f} GB; total peak incl. engines {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    if args.out:
        json.dump({"tag": tag, "text": text, "tokens": tokens, "logprobs": logprobs, "ref": ref, "runs": runs,
                   "same_tokens": same}, open(args.out, "w"), indent=1)
        print("[trt] wrote", args.out)


if __name__ == "__main__":
    main()
