#!/usr/bin/env python3
"""mT5 -> ONNX with explicit KV-cache I/O via torch.onnx.export (route B).

    pip3 install "transformers>=4.45" sentencepiece onnx onnxruntime
    python task3_mt5_onnx/02_export_manual.py --model google/mt5-base --out models/mt5_onnx_manual \
        --encoder-input embeds --verify

Produces three graphs with the SAME tensor names optimum uses, so
03_build_engines.py and a future TRT decode loop work with either route:

  encoder.onnx
      inputs_embeds [B,T,d_model] (or input_ids [B,T]), attention_mask [B,T]
      -> last_hidden_state [B,T,d_model]
  decoder_init.onnx           (first step, no past)
      input_ids [B,S], encoder_hidden_states [B,T,d_model], encoder_attention_mask [B,T]
      -> logits [B,S,V], present.{i}.decoder.key/value [B,H,S,d_kv],
         present.{i}.encoder.key/value [B,H,T,d_kv]
  decoder_step.onnx           (every later step)
      input_ids [B,1], encoder_attention_mask [B,T],
      past_key_values.{i}.decoder.key/value [B,H,P,d_kv],
      past_key_values.{i}.encoder.key/value [B,H,T,d_kv]
      -> logits [B,1,V], present.{i}.decoder.key/value [B,H,P+1,d_kv]
      (encoder K/V are constant across steps: feed the same tensors every step.)

Symbolic dims: batch_size, encoder_sequence_length, decoder_sequence_length,
past_decoder_sequence_length, decoder_total_sequence_length.

=========================== VERIFY-BY-HAND LIST ==============================
[V1] Cache plumbing depends on transformers internals (EncoderDecoderCache,
     from_legacy_cache/to_legacy_cache, is_updated). Written against 4.57;
     older (<4.46) versions use plain tuples. --verify is the real test.
[V2] The step graph must NOT receive encoder_hidden_states (it would be an
     unused input and torch.onnx.export silently drops unused inputs, which
     misaligns input_names). We synthesise a dummy from the mask instead; the
     script asserts the exported input names match the expected list.
[V3] T5 relative position bias + causal mask are built from traced shapes
     (torch.arange over cache length). The TorchScript tracer usually keeps
     these dynamic, but any TracerWarning about "converting a tensor to a
     Python boolean/int" means a branch was frozen at the sample length.
     --verify runs at lengths different from the export sample precisely to
     catch this; do not trust the graphs without it.
[V4] attn_implementation="eager" is forced for export (SDPA export -> TRT is
     flaky). Numerics are identical; speed of the PyTorch reference is not.
[V5] mT5-base lm_head is 250112 x 768 (untied) -> ~770 MB alone; fp32 graphs
     will exceed the 2 GB protobuf limit and be written with external data
     (*.onnx.data next to the file). Keep both files together for TensorRT.
==============================================================================
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

EXPECTED_MIN_TF = (4, 46)


def _tf_version():
    import transformers
    return tuple(int(x) for x in transformers.__version__.split(".")[:2])


def _dynamo_kw():
    """torch>=2.5 has torch.onnx.export(dynamo=...); >=2.9 defaults it to True. We want the
    TorchScript exporter, so pass dynamo=False where the kwarg exists (Jetson torch 2.8 has it)."""
    major, minor = (int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    return {"dynamo": False} if (major, minor) >= (2, 5) else {}


class EncoderWrapper(nn.Module):
    def __init__(self, model, use_ids):
        super().__init__()
        self.enc = model.encoder
        self.use_ids = use_ids

    def forward(self, x, attention_mask):
        kw = {"input_ids": x} if self.use_ids else {"inputs_embeds": x}
        return self.enc(attention_mask=attention_mask, return_dict=True, **kw).last_hidden_state


def _lm_logits(model, hidden):
    if model.config.tie_word_embeddings:  # False for mT5, True for t5-v1.0
        hidden = hidden * (model.model_dim ** -0.5)
    return model.lm_head(hidden)


def _flatten_cache(cache):
    """EncoderDecoderCache -> [sk0, sv0, ck0, cv0, sk1, ...] (legacy layer order)."""
    legacy = cache.to_legacy_cache()  # tuple(layer) of (self_k, self_v, cross_k, cross_v)
    flat = []
    for layer in legacy:
        flat.extend(layer)
    return flat


class DecoderInitWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask):
        out = self.model.decoder(input_ids=input_ids, encoder_hidden_states=encoder_hidden_states,
                                 encoder_attention_mask=encoder_attention_mask, use_cache=True,
                                 return_dict=True)
        return (_lm_logits(self.model, out.last_hidden_state), *_flatten_cache(out.past_key_values))


class DecoderStepWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_layers = model.config.num_decoder_layers
        self.d_model = model.config.d_model

    def forward(self, input_ids, encoder_attention_mask, *past):
        from transformers.cache_utils import EncoderDecoderCache
        assert len(past) == 4 * self.n_layers, f"expected {4 * self.n_layers} past tensors, got {len(past)}"
        layers = tuple(tuple(past[4 * i:4 * i + 4]) for i in range(self.n_layers))
        cache = EncoderDecoderCache.from_legacy_cache(layers)  # sets is_updated -> cross K/V reused [V1]
        # [V2] T5Block skips cross-attention entirely when encoder_hidden_states is None, so we pass a
        # dummy whose only role is being non-None and having the right length; its values are never read.
        dummy_ehs = encoder_attention_mask.unsqueeze(-1).to(past[0].dtype).expand(-1, -1, self.d_model)
        out = self.model.decoder(input_ids=input_ids, encoder_hidden_states=dummy_ehs,
                                 encoder_attention_mask=encoder_attention_mask, past_key_values=cache,
                                 use_cache=True, return_dict=True)
        flat = _flatten_cache(out.past_key_values)
        self_kv = [t for i, t in enumerate(flat) if i % 4 in (0, 1)]  # drop cross K/V (unchanged)
        return (_lm_logits(self.model, out.last_hidden_state), *self_kv)


def kv_names(prefix, n_layers, kinds=("decoder", "encoder")):
    names = []
    for i in range(n_layers):
        for kind in kinds:
            names += [f"{prefix}.{i}.{kind}.key", f"{prefix}.{i}.{kind}.value"]
    return names


def _export(module, args_tuple, path, input_names, output_names, dynamic_axes, opset):
    print(f"[export] {os.path.basename(path)} ...")
    t0 = time.time()
    # GOTCHA: torch.onnx.export restores the *wrapper's* original train/eval flag on exit, and a
    # fresh nn.Module wrapper defaults to training=True. That would flip the shared mT5 submodules
    # into train mode (dropout on!) after export and silently break the PyTorch reference. So: eval
    # before AND after.
    module.eval()
    with torch.no_grad():
        torch.onnx.export(module, args_tuple, path, input_names=input_names, output_names=output_names,
                          dynamic_axes=dynamic_axes, opset_version=opset, do_constant_folding=True,
                          **_dynamo_kw())
    module.eval()
    import onnx
    m = onnx.load(path, load_external_data=False)
    got_in = [i.name for i in m.graph.input]
    got_out = [o.name for o in m.graph.output]
    assert got_in == input_names, f"[V2] input names mismatch\n expected {input_names}\n got {got_in}"
    assert got_out == output_names, f"output names mismatch\n expected {output_names}\n got {got_out}"
    print(f"[export]   ok in {time.time() - t0:.0f}s: {len(got_in)} inputs, {len(got_out)} outputs")


def export_all(model, out_dir, opset, encoder_input, sample_T=13, sample_P=7):
    cfg = model.config
    L, H, dkv, d = cfg.num_decoder_layers, cfg.num_heads, cfg.d_kv, cfg.d_model
    B = 1
    os.makedirs(out_dir, exist_ok=True)
    bs, es, ds, ps, ts = ("batch_size", "encoder_sequence_length", "decoder_sequence_length",
                          "past_decoder_sequence_length", "decoder_total_sequence_length")

    # ---- encoder
    use_ids = encoder_input == "ids"
    x = torch.randint(0, cfg.vocab_size, (B, sample_T)) if use_ids else torch.randn(B, sample_T, d)
    mask = torch.ones(B, sample_T, dtype=torch.long)
    xname = "input_ids" if use_ids else "inputs_embeds"
    _export(EncoderWrapper(model, use_ids), (x, mask), os.path.join(out_dir, "encoder.onnx"),
            [xname, "attention_mask"], ["last_hidden_state"],
            {xname: {0: bs, 1: es}, "attention_mask": {0: bs, 1: es}, "last_hidden_state": {0: bs, 1: es}},
            opset)

    # ---- decoder init
    ehs = torch.randn(B, sample_T, d)
    dec_ids = torch.full((B, 1), cfg.decoder_start_token_id, dtype=torch.long)
    outs = ["logits"] + kv_names("present", L)
    dyn = {"input_ids": {0: bs, 1: ds}, "encoder_hidden_states": {0: bs, 1: es},
           "encoder_attention_mask": {0: bs, 1: es}, "logits": {0: bs, 1: ds}}
    for n in outs[1:]:
        dyn[n] = {0: bs, 2: ds if ".decoder." in n else es}
    _export(DecoderInitWrapper(model), (dec_ids, ehs, mask), os.path.join(out_dir, "decoder_init.onnx"),
            ["input_ids", "encoder_hidden_states", "encoder_attention_mask"], outs, dyn, opset)

    # ---- decoder step
    past_in = kv_names("past_key_values", L)
    past = []
    for n in past_in:
        past.append(torch.randn(B, H, sample_P if ".decoder." in n else sample_T, dkv))
    outs = ["logits"] + kv_names("present", L, kinds=("decoder",))
    dyn = {"input_ids": {0: bs}, "encoder_attention_mask": {0: bs, 1: es}, "logits": {0: bs}}
    for n in past_in:
        dyn[n] = {0: bs, 2: ps if ".decoder." in n else es}
    for n in outs[1:]:
        dyn[n] = {0: bs, 2: ts}
    _export(DecoderStepWrapper(model), (dec_ids, mask, *past), os.path.join(out_dir, "decoder_step.onnx"),
            ["input_ids", "encoder_attention_mask"] + past_in, outs, dyn, opset)


# ------------------------------------------------------------------- verification
@torch.no_grad()
def verify(model, out_dir, encoder_input, T=21, steps=6, atol=2e-3):
    """Greedy decode with the ONNX graphs (onnxruntime) vs PyTorch with its own cache.
    Uses T != export sample and P growing 1..steps, so frozen shapes would show up here. [V3]"""
    import onnxruntime as ort
    cfg = model.config
    L = cfg.num_decoder_layers
    so = ort.SessionOptions()
    enc = ort.InferenceSession(os.path.join(out_dir, "encoder.onnx"), so, providers=["CPUExecutionProvider"])
    dec0 = ort.InferenceSession(os.path.join(out_dir, "decoder_init.onnx"), so, providers=["CPUExecutionProvider"])
    dec1 = ort.InferenceSession(os.path.join(out_dir, "decoder_step.onnx"), so, providers=["CPUExecutionProvider"])

    torch.manual_seed(1)
    use_ids = encoder_input == "ids"
    x = torch.randint(0, cfg.vocab_size, (1, T)) if use_ids else torch.randn(1, T, cfg.d_model)
    mask = torch.ones(1, T, dtype=torch.long)
    xname = "input_ids" if use_ids else "inputs_embeds"

    # --- PyTorch reference (step by step with cache)
    kw = {"input_ids": x} if use_ids else {"inputs_embeds": x}
    h_ref = model.encoder(attention_mask=mask, **kw).last_hidden_state
    ids = torch.full((1, 1), cfg.decoder_start_token_id, dtype=torch.long)
    cache = None
    ref_logits, ref_tokens = [], []
    for s in range(steps):
        out = model(encoder_outputs=(h_ref,), attention_mask=mask, decoder_input_ids=ids,
                    past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        ref_logits.append(out.logits[:, -1].numpy())
        ids = out.logits[:, -1].argmax(-1, keepdim=True)
        ref_tokens.append(int(ids))

    # --- ONNX loop
    h = enc.run(None, {xname: x.numpy(), "attention_mask": mask.numpy()})[0]
    d_enc = np.abs(h - h_ref.numpy()).max()
    ids = np.full((1, 1), cfg.decoder_start_token_id, dtype=np.int64)
    o = dec0.run(None, {"input_ids": ids, "encoder_hidden_states": h, "encoder_attention_mask": mask.numpy()})
    names0 = [x_.name for x_ in dec0.get_outputs()]
    o = dict(zip(names0, o))
    onnx_logits, onnx_tokens = [o["logits"][:, -1]], [int(o["logits"][:, -1].argmax())]
    past = {n.replace("present.", "past_key_values."): v for n, v in o.items() if n.startswith("present.")}
    for s in range(1, steps):
        ids = np.array([[onnx_tokens[-1]]], dtype=np.int64)
        feed = {"input_ids": ids, "encoder_attention_mask": mask.numpy(), **past}
        r = dec1.run(None, feed)
        r = dict(zip([x_.name for x_ in dec1.get_outputs()], r))
        onnx_logits.append(r["logits"][:, -1]); onnx_tokens.append(int(r["logits"][:, -1].argmax()))
        for i in range(L):
            for k in ("key", "value"):
                past[f"past_key_values.{i}.decoder.{k}"] = r[f"present.{i}.decoder.{k}"]
        assert past["past_key_values.0.decoder.key"].shape[2] == s + 1, "cache length did not grow by 1"

    d_logits = [float(np.abs(a - b).max()) for a, b in zip(ref_logits, onnx_logits)]
    print(f"[verify] T={T} steps={steps}  encoder max|diff|={d_enc:.2e}")
    print(f"[verify] per-step logits max|diff|: {['%.2e' % v for v in d_logits]}")
    print(f"[verify] tokens torch={ref_tokens}\n[verify] tokens onnx ={onnx_tokens}")
    # note: tokens are compared with the same-source reference (torch greedy), so a random-init
    # model with near-tied logits can flip a token while logits still agree within atol.
    ok = d_enc < atol and max(d_logits) < atol
    print("[verify]", "PASS" if ok else "FAIL (see [V1]-[V3] in the header)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/mt5-base")
    ap.add_argument("--out", default="models/mt5_onnx_manual")
    ap.add_argument("--encoder-input", choices=["embeds", "ids"], default="embeds")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true", help="onnxruntime vs PyTorch greedy decode")
    ap.add_argument("--verify-T", type=int, default=21)
    ap.add_argument("--verify-steps", type=int, default=6)
    ap.add_argument("--tiny", action="store_true", help="random tiny mT5 config for a fast self-test")
    args = ap.parse_args()

    from transformers import MT5Config, MT5ForConditionalGeneration
    if _tf_version() < EXPECTED_MIN_TF:
        print(f"[warn] transformers {_tf_version()} < {EXPECTED_MIN_TF}: cache API differs, see [V1]")
    if args.tiny:
        cfg = MT5Config(vocab_size=128, d_model=32, d_kv=8, d_ff=64, num_layers=2, num_decoder_layers=2,
                        num_heads=4, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
        torch.manual_seed(0)
        model = MT5ForConditionalGeneration(cfg)
    else:
        model = MT5ForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float32,
                                                            attn_implementation="eager")  # [V4]
    model.config._attn_implementation = "eager"
    model.eval()
    export_all(model, args.out, args.opset, args.encoder_input)
    if args.verify:
        ok = verify(model, args.out, args.encoder_input, args.verify_T, args.verify_steps)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
