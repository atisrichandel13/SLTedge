#!/usr/bin/env python3
"""Greedy decode with the exported mT5 ONNX graphs (encoder / decoder_init / decoder_step) driven by
real pose features from the standalone model. Checks the ONNX path end to end on a clip and is the
reference for the TensorRT decode loop on the Jetson (same tensor names, same step order).

    python -m unisign.onnx_decode --onnx-dir models/mt5_pruned_onnx \
        --ckpt weights/openasl_pose_only_slt_pruned.pth --mt5 weights/mt5-base \
        --pkl data/openasl_ref_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl
"""
import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.pose_to_unisign import collate, load_keypoints_json, load_pkl, to_model_inputs  # noqa: E402
from unisign.model import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mt5", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pkl"); g.add_argument("--keypoints")
    ap.add_argument("--crop-wh", type=int, nargs=2, default=[666, 720])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--providers", default="CPUExecutionProvider")
    args = ap.parse_args()

    model = load_model(args.ckpt, args.mt5)  # pose stack + PyTorch reference
    kps, scs, _ = load_pkl(args.pkl) if args.pkl else load_keypoints_json(args.keypoints, args.crop_wh)
    src = collate([to_model_inputs(kps, scs, 256)[0]], ["clip"])
    ref = model.translate(src, max_new_tokens=args.max_new_tokens, num_beams=1)
    with torch.no_grad():
        emb, mask = model.build_encoder_inputs(src)
    emb, mask = emb.numpy().astype(np.float32), mask.numpy().astype(np.int64)

    so = ort.SessionOptions(); prov = args.providers.split(",")
    enc = ort.InferenceSession(os.path.join(args.onnx_dir, "encoder.onnx"), so, providers=prov)
    init = ort.InferenceSession(os.path.join(args.onnx_dir, "decoder_init.onnx"), so, providers=prov)
    step = ort.InferenceSession(os.path.join(args.onnx_dir, "decoder_step.onnx"), so, providers=prov)
    enc_in = [i.name for i in enc.get_inputs()]
    t0 = time.perf_counter()
    hidden = enc.run(None, {enc_in[0]: emb, enc_in[1]: mask})[0]
    t_enc = (time.perf_counter() - t0) * 1000

    init_names = [o.name for o in init.get_outputs()]
    step_in = [i.name for i in step.get_inputs()]
    step_out = [o.name for o in step.get_outputs()]
    eos, tokens, logprobs = 1, [], []
    t0 = time.perf_counter()
    ids = np.zeros((1, 1), dtype=np.int64)  # decoder_start = pad = 0
    outs = init.run(None, {"input_ids": ids, "encoder_hidden_states": hidden, "encoder_attention_mask": mask})
    present = dict(zip(init_names, outs))
    logits = present.pop("logits")
    for _ in range(args.max_new_tokens):
        nxt = int(logits[0, -1].argmax()); lp = logits[0, -1] - np.log(np.exp(logits[0, -1] - logits[0, -1].max()).sum()) - logits[0, -1].max()
        tokens.append(nxt); logprobs.append(float(lp[nxt]))
        if nxt == eos:
            break
        feed = {"input_ids": np.array([[nxt]], dtype=np.int64), "encoder_attention_mask": mask}
        for n in step_in:
            if n.startswith("past_key_values."):
                feed[n] = present[n.replace("past_key_values.", "present.")]
        outs = step.run(None, feed)
        new = dict(zip(step_out, outs))
        logits = new.pop("logits")
        present.update(new)  # decoder K/V grow; encoder K/V untouched
    t_dec = (time.perf_counter() - t0) * 1000

    text = model.mt5_tokenizer.decode(torch.tensor([0] + tokens), skip_special_tokens=True)
    ref_tokens = ref["tokens"]
    print(f"[onnx] encoder {t_enc:.0f} ms, decoder {t_dec:.0f} ms for {len(tokens)} tokens ({t_dec/len(tokens):.1f} ms/token)")
    print(f"[onnx] text : {text}")
    print(f"[torch] text: {ref['text']}")
    same = tokens == ref_tokens
    n = min(len(tokens), len(ref_tokens)); first_diff = next((i for i in range(n) if tokens[i] != ref_tokens[i]), None)
    print(f"[check] tokens identical: {same}" + ("" if same else f"  (first divergence at step {first_diff}, lens {len(tokens)} vs {len(ref_tokens)})"))
    if ref["token_logprobs"]:
        d = max(abs(a - b) for a, b in zip(logprobs, ref["token_logprobs"]))
        print(f"[check] max |logprob diff| over shared steps: {d:.2e}")
    sys.exit(0 if same else 1)


if __name__ == "__main__":
    main()
