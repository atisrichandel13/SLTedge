#!/usr/bin/env python3
"""Export the Uni-Sign front end -- proj_linear + ST-GCN chains + fusion + part_para + pose_proj,
with the constant prefix embeddings prepended -- as front.onnx, so the whole translator can run on
the phone: front.onnx -> encoder.onnx -> decoder_{init,step}.onnx (task3_mt5_onnx/02_export_manual.py).

    python3 task3_mt5_onnx/04_export_front.py --ckpt weights/openasl_pose_only_slt_pruned.pth \
        --mt5 weights/mt5-base --out models/mt5_pruned_onnx/front.onnx --verify

front.onnx
    body [1,T,9,3], left [1,T,21,3], right [1,T,21,3], face_all [1,T,18,3]  (load_part_kp tensors)
    -> inputs_embeds [1,P+T,768]
where P is the length of the tokenised prefix "Translate sign language video to English: ". The
attention mask the encoder needs is all ones of length P+T for a single unpadded clip, so it is
not an output. --verify runs ORT at two lengths different from the export sample against
PoseOnlyUniSign.build_encoder_inputs on the same tensors (the [V3] frozen-shape check).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from unisign.model import load_model  # noqa: E402


class Front(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        tok = m.mt5_tokenizer([m.prefix], padding="longest", truncation=True, return_tensors="pt")
        emb = m.mt5_model.encoder.embed_tokens(tok["input_ids"]).detach().clone()
        self.register_buffer("prefix_embeds", emb)
        self.prefix_len = int(tok["input_ids"].shape[1])

    def forward(self, body, left, right, face_all):
        pose = self.m.encode_pose({"body": body, "left": left, "right": right, "face_all": face_all})
        return torch.cat([self.prefix_embeds, pose], dim=1)


def sample(T, rng):
    def part(V):
        x = rng.uniform(-1, 1, size=(1, T, V, 3)).astype(np.float32)
        x[..., 2] = rng.uniform(0, 1, size=(1, T, V))
        x[x[..., 2] <= 0.3] = 0
        return torch.from_numpy(x)
    return part(9), part(21), part(21), part(18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mt5", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    m = load_model(args.ckpt, args.mt5, device="cpu", dtype=torch.float32).eval()
    front = Front(m).eval()
    print(f"[front] prefix length P = {front.prefix_len}")

    rng = np.random.default_rng(0)
    ex = sample(48, rng)
    names = ["body", "left", "right", "face_all"]
    dyn = {n: {1: "T"} for n in names}
    dyn["inputs_embeds"] = {1: "P_plus_T"}
    kw = {"dynamo": False} if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 5) else {}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(front, ex, args.out, input_names=names, output_names=["inputs_embeds"],
                          dynamic_axes=dyn, opset_version=args.opset, do_constant_folding=True, **kw)
    print(f"[front] wrote {args.out} ({os.path.getsize(args.out) / 2**20:.1f} MB)")

    if args.verify:
        import onnxruntime as ort
        s = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        for T in (17, 203):
            x = sample(T, rng)
            with torch.no_grad():
                src = {n: v for n, v in zip(names, x)}
                src["attention_mask"] = torch.ones(1, T, dtype=torch.long)
                ref, mask = m.build_encoder_inputs(src)
            got = s.run(None, {n: v.numpy() for n, v in zip(names, x)})[0]
            assert got.shape == tuple(ref.shape), (got.shape, ref.shape)
            assert mask.shape[1] == front.prefix_len + T
            err = float(np.abs(got - ref.numpy()).max())
            print(f"[front] T={T}: shape {got.shape}, max |ORT - PyTorch| = {err:.2e}")
            assert err < 1e-3, err
        print("[front] verify OK")


if __name__ == "__main__":
    main()
