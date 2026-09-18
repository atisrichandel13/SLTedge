#!/usr/bin/env python3
"""mT5 -> ONNX with KV cache via HuggingFace optimum (route A).

    pip3 install "optimum[exporters]>=1.20" onnx onnxruntime
    python task3_mt5_onnx/01_export_optimum.py --model google/mt5-base --out models/mt5_onnx_optimum

optimum supports model_type "mt5" for task `text2text-generation-with-past`.
With --no-post-process it leaves three graphs:
    encoder_model.onnx            input_ids, attention_mask -> last_hidden_state
    decoder_model.onnx            input_ids, encoder_attention_mask, encoder_hidden_states
                                  -> logits, present.{i}.decoder.{key,value}, present.{i}.encoder.{key,value}
    decoder_with_past_model.onnx  input_ids[B,1], encoder_attention_mask,
                                  past_key_values.{i}.decoder.{key,value}, past_key_values.{i}.encoder.{key,value}
                                  -> logits, present.{i}.decoder.{key,value}
(without --no-post-process optimum also merges the two decoders into
decoder_model_merged.onnx behind an `If` node on `use_cache_branch`; TensorRT
handles that poorly, so we keep them separate = two decoder engines.)

KNOWN LIMITATION FOR UNI-SIGN: the optimum encoder takes `input_ids`, but
Uni-Sign feeds pose features as `inputs_embeds`. Use the decoders from here
(well-tested cache plumbing) and the encoder from 02_export_manual.py
(--encoder-input embeds), or use 02 for everything.

This script prints every graph's I/O so you can confirm the names above before
building engines with 03_build_engines.py.
"""
import argparse
import os
import subprocess
import sys


def print_io(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    print(f"\n== {os.path.basename(path)}")
    for kind, seq in (("in ", m.graph.input), ("out", m.graph.output)):
        for t in seq:
            dims = [d.dim_param or d.dim_value for d in t.type.tensor_type.shape.dim]
            print(f"  {kind} {t.name:40s} {dims}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/mt5-base")
    ap.add_argument("--out", default="models/mt5_onnx_optimum")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-validate", action="store_true", help="skip optimum's ORT validation")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        from optimum.exporters.onnx import main_export
    except ImportError:
        print("[optimum] python API not importable, falling back to optimum-cli")
        cmd = [sys.executable, "-m", "optimum.exporters.onnx", "--model", args.model,
               "--task", "text2text-generation-with-past", "--opset", str(args.opset),
               "--no-post-process", args.out]
        print(" ".join(cmd))
        subprocess.check_call(cmd)
    else:
        # VERIFY: main_export kwargs shift between optimum versions; if this raises a TypeError
        # run the CLI line above instead.
        main_export(model_name_or_path=args.model, output=args.out,
                    task="text2text-generation-with-past", opset=args.opset,
                    no_post_process=True, do_validation=not args.no_validate, device="cpu")

    for f in sorted(os.listdir(args.out)):
        if f.endswith(".onnx"):
            print_io(os.path.join(args.out, f))
    print("\n[optimum] done. Note: files >2GB use external data (*.onnx_data); keep them together.")


if __name__ == "__main__":
    main()
