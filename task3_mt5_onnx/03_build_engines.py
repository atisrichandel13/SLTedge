#!/usr/bin/env python3
"""Build TensorRT engines for the mT5 ONNX graphs with dynamic-shape profiles.

    python task3_mt5_onnx/03_build_engines.py --onnx-dir models/mt5_onnx_manual --which all \
        --enc-len 1,64,128 --dec-len 1,1,128 --batch 1,1,1

Profiles are assigned per symbolic dimension name found in the ONNX graph:
    batch_size                      -> --batch  min,opt,max
    encoder_sequence_length         -> --enc-len
    decoder_sequence_length         -> --dec-len            (decoder_init: prompt length; 1 for greedy)
    past_decoder_sequence_length    -> (1, opt, max_dec-1)  (decoder_step past length)
    decoder_total_sequence_length   -> output only, ignored
Both optimum's names (decoder_model.onnx / decoder_with_past_model.onnx) and
02_export_manual.py's names are handled. Any other symbolic dim must be given
with --dim NAME=min,opt,max or the script stops and lists it.

Memory: mT5-base fp32 decoder engine ~1.6 GB serialized; the builder wants a
few GB more while optimising. Build ONE engine per process on the 8 GB Orin
Nano (--which encoder|decoder_init|decoder_step) if `all` gets OOM-killed, and
close other GPU users (X11, the mT5 PyTorch baseline) first.

Equivalent trtexec for the step decoder (names abbreviated):
    trtexec --onnx=decoder_step.onnx --saveEngine=decoder_step_fp32.engine \
      --minShapes=input_ids:1x1,encoder_attention_mask:1x1,past_key_values.0.decoder.key:1x12x1x64,... \
      --optShapes=... --maxShapes=...
"""
import argparse
import gc
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.trt_runner import build_engine_from_onnx  # noqa: E402

CANDIDATES = {
    "encoder": ["encoder.onnx", "encoder_model.onnx"],
    "decoder_init": ["decoder_init.onnx", "decoder_model.onnx"],
    "decoder_step": ["decoder_step.onnx", "decoder_with_past_model.onnx"],
}


def triple(s):
    a, b, c = (int(v) for v in s.split(","))
    assert a <= b <= c, s
    return a, b, c


def profile_for(onnx_path, dim_ranges):
    import onnx
    m = onnx.load(onnx_path, load_external_data=False)
    prof, unknown = {}, set()
    for inp in m.graph.input:
        mn, opt, mx = [], [], []
        for d in inp.type.tensor_type.shape.dim:
            if d.dim_param:
                if d.dim_param not in dim_ranges:
                    unknown.add(d.dim_param)
                    continue
                a, b, c = dim_ranges[d.dim_param]
            else:
                a = b = c = d.dim_value
            mn.append(a); opt.append(b); mx.append(c)
        prof[inp.name] = (tuple(mn), tuple(opt), tuple(mx))
    if unknown:
        raise SystemExit(f"unmapped symbolic dims {sorted(unknown)} in {onnx_path}; "
                         f"add --dim NAME=min,opt,max")
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--which", default="all", choices=["all", *CANDIDATES])
    ap.add_argument("--batch", type=triple, default=(1, 1, 1))
    ap.add_argument("--enc-len", type=triple, default=(1, 64, 128), help="encoder length min,opt,max")
    ap.add_argument("--dec-len", type=triple, default=(1, 1, 128), help="decoder length; max = decode budget")
    ap.add_argument("--dim", action="append", default=[], help="extra NAME=min,opt,max")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--workspace-gb", type=float, default=3.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    max_dec = args.dec_len[2]
    dim_ranges = {
        "batch_size": args.batch,
        "encoder_sequence_length": args.enc_len,
        "decoder_sequence_length": args.dec_len,
        "past_decoder_sequence_length": (1, max(1, min(max_dec // 2, max_dec - 1)), max_dec - 1),
        # optimum's merged/legacy exports sometimes use these:
        "past_sequence_length": (1, max(1, min(max_dec // 2, max_dec - 1)), max_dec - 1),
        "sequence_length": args.dec_len,
    }
    for s in args.dim:
        k, v = s.split("=")
        dim_ranges[k] = triple(v)

    targets = list(CANDIDATES) if args.which == "all" else [args.which]
    for t in targets:
        path = next((os.path.join(args.onnx_dir, c) for c in CANDIDATES[t]
                     if os.path.exists(os.path.join(args.onnx_dir, c))), None)
        if path is None:
            print(f"[build] skip {t}: none of {CANDIDATES[t]} in {args.onnx_dir}")
            continue
        prof = profile_for(path, dim_ranges)
        print(f"[build] {t} <- {os.path.basename(path)}")
        for n, (a, b, c) in list(prof.items())[:6]:
            print(f"        {n:40s} min={a} opt={b} max={c}")
        if len(prof) > 6:
            print(f"        ... {len(prof) - 6} more inputs with the same pattern")
        engine = os.path.join(args.onnx_dir, f"{t}_{'fp16' if args.fp16 else 'fp32'}.engine")
        build_engine_from_onnx(path, engine, fp16=args.fp16, workspace_gb=args.workspace_gb,
                               profiles=[prof], verbose=args.verbose)
        gc.collect()


if __name__ == "__main__":
    main()
