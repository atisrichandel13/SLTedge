"""Weight-only INT8 (W8A16) for the mT5 part of the standalone Uni-Sign model. Task L9.

Every 2-D mT5 weight with >= 1e5 elements (q/k/v/o, wi_0/wi_1/wo, shared embedding, lm_head) is
stored as int8 with one fp16 scale per output row (symmetric, absmax / 127). Layer norms, the
relative-attention bias and the whole pose stack (5.4 M params) stay in full precision.

Two ways to run a quantised checkpoint (see load_model(..., w8_runtime=...)):
  "dequant": weights are expanded back to float at load time. Exact W8A16 numerics for accuracy
             work, but RAM = the float model. Default, and what the accuracy numbers use.
  "int8":    nn.Linear modules are swapped for W8Linear, which keeps the int8 tensor in memory and
             dequantises per forward. Real RAM saving; a little slower. For board memory rows.
Both give bit-identical outputs (same dequantised weights), so accuracy is measured once.

    python -m unisign.quant --ckpt weights/openasl_pose_only_slt_pruned.pth \
        --out weights/openasl_pose_only_slt_pruned_w8.pth
"""
import argparse
import os

import torch
from torch import nn
from torch.nn import functional as F

MIN_NUMEL = 100_000


def quantize_tensor(w):
    """(rows, cols) float -> int8 (rows, cols), fp16 scale (rows,). Symmetric per-row absmax."""
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale).clamp(-127, 127).to(torch.int8)
    return q, scale.squeeze(1).to(torch.float16)


def dequantize_tensor(q, scale, dtype=torch.float32):
    return q.to(dtype) * scale.to(dtype)[:, None]


def is_quant_target(key, t):
    return key.startswith("mt5_model.") and key.endswith(".weight") and t.dim() == 2 and t.numel() >= MIN_NUMEL


def quantize_state_dict(sd):
    """Returns (new_sd, w8_keys). Quantised entries: key -> int8, key + '__scale' -> fp16."""
    out, keys = {}, []
    for k, t in sd.items():
        if is_quant_target(k, t):
            q, s = quantize_tensor(t)
            out[k], out[k + "__scale"] = q, s
            keys.append(k)
        else:
            out[k] = t
    return out, keys


def dequantize_state_dict(sd, w8_keys, dtype=torch.float32):
    out = dict(sd)
    for k in w8_keys:
        out[k] = dequantize_tensor(out[k], out.pop(k + "__scale"), dtype)
    return out


class W8Linear(nn.Module):
    """nn.Linear with int8 weight + fp16 per-row scale kept in memory; dequantised per forward."""

    def __init__(self, q, scale, bias=None, compute_dtype=torch.float32):
        super().__init__()
        self.register_buffer("q", q)
        self.register_buffer("scale", scale)
        self.bias = None if bias is None else nn.Parameter(bias)
        self.in_features, self.out_features = q.shape[1], q.shape[0]
        self.compute_dtype = compute_dtype  # transformers casts activations to wo.weight.dtype

    @property
    def weight(self):  # for code that inspects .weight (shape/dtype); dequantised on demand
        return dequantize_tensor(self.q, self.scale, self.compute_dtype)

    def forward(self, x):
        x = x.to(self.compute_dtype)
        w = self.q.to(self.compute_dtype) * self.scale.to(self.compute_dtype)[:, None]
        return F.linear(x, w, None if self.bias is None else self.bias.to(self.compute_dtype))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, int8 weight + fp16 scale"


def swap_linears_to_w8(model, sd, w8_keys):
    """Replace the nn.Linear at each quantised key with W8Linear holding the checkpoint's int8
    tensors. Embedding weights (shared / embed_tokens) are left dequantised: an embedding lookup
    only touches one row per token, so int8 storage there is a memory-only concern handled by the
    dequant path. Returns number of modules swapped."""
    n = 0
    for k in w8_keys:
        path = k[: -len(".weight")]
        parent_path, _, attr = path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        mod = getattr(parent, attr)
        if not isinstance(mod, nn.Linear):
            continue
        dev, cdt = mod.weight.device, mod.weight.dtype
        new = W8Linear(sd[k].to(dev), sd[k + "__scale"].to(dev),
                       None if mod.bias is None else mod.bias.data.clone(), compute_dtype=cdt)
        setattr(parent, attr, new)
        n += 1
    return n


def quantize_checkpoint(ckpt_in, ckpt_out):
    ck = torch.load(ckpt_in, map_location="cpu")
    sd = ck.get("model", ck)
    new_sd, keys = quantize_state_dict(sd)
    out = {"model": new_sd, "w8_keys": keys}
    if "keep_ids" in ck:
        out["keep_ids"] = ck["keep_ids"]
    torch.save(out, ckpt_out)
    q_params = sum(sd[k].numel() for k in keys)
    total = sum(t.numel() for t in sd.values())
    err = max(((dequantize_tensor(new_sd[k], new_sd[k + "__scale"]) - sd[k].float()).abs().max()
               / sd[k].float().abs().max()).item() for k in keys)
    print(f"[quant] {len(keys)} tensors, {q_params/1e6:.1f}M of {total/1e6:.1f}M params -> int8; "
          f"max per-tensor rel. error {err:.2e}")
    print(f"[quant] {os.path.getsize(ckpt_in)/1e6:.0f} MB -> {os.path.getsize(ckpt_out)/1e6:.0f} MB  wrote {ckpt_out}")
    return keys


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    quantize_checkpoint(ap.parse_args().ckpt, ap.parse_args().out)
