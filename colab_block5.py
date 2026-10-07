#!/usr/bin/env python3
"""Block 5: the pose track's three outstanding asks, plus the corrected test-split number.

    !python colab_setup.py && python colab_block5.py

Re-runnable: every step is skipped when its output exists.

WHAT THIS PRODUCES AND WHY EACH ONE IS OWED

  1. TEST-SPLIT EVAL OF THE LEAK-FREE CHECKPOINT (their ask 3).
     Every test-split BLEU in both tracks' documents -- including the 22.87 they restate in their
     section 9.1 -- comes from a keep set built over train+dev+TEST. The 53 test-only tokens occur
     55 times across 46 of 976 test sentences, so the corrected number should sit slightly below
     22.87 and IS the honest one. Run at both caps: cap 100 matches the original L5.3 protocol,
     cap 64 is the deployment default and L3.2 measured them bit-identical on this model.

  2. PRE-PRUNED HF DIRECTORY AT 26,025 (their ask 2).
     Their attach_pruned_tokenizer fast path avoids materialising the 250k-vocab mT5 before slicing
     it -- the load peak they measured at ~4.2 GB against ~5.3 GB available, which is why their
     end-to-end sustained run has never completed. The existing weights/mt5-base-openasl-pruned is
     at the superseded 26,078 keep set, and their merged cross-check compares keep ids ELEMENTWISE,
     so a stale directory is now rejected rather than silently mistranslating.
     keep_ids.json is written alongside, because the directory's spiece.model is the ORIGINAL 250k
     sentencepiece (byte-identical to google/mt5-base's) and without the remap every id is wrong.

  3. VERIFY THE FAST PATH REPRODUCES KNOWN-GOOD TEXT (their ask 1).
     They cannot run this: their Mac has no torch and the board is unreachable. The failure mode is
     fluent, WRONG text with no error raised, so a sentence-level check against a known-good
     prediction is the only acceptable evidence. This compares the fast path's output against the
     ordinary load path's on the same clips and requires EXACT string equality.

Outputs under /content/runs/block5/.
"""
import json
import os
import subprocess
import sys

# getattr, not a bare call: under IPython sys.stdout is an ipykernel OutStream with NO reconfigure,
# so importing this module inside a Colab cell dies with AttributeError before anything runs. Hit for
# real 2026-10-07 importing colab_setup to skip its unused archive step. Under `!python file.py`
# stdout is a real TextIOWrapper and this behaves exactly as before; IPython already line-buffers.
_rc = getattr(sys.stdout, "reconfigure", None)
if _rc:
    _rc(line_buffering=True)

R = "/content/runs/block5"
CKPT = "/content/pruned_traindev.pth"
MT5 = "/content/weights/mt5-base"
POSES = "/content/poses_train"
PRUNED_DIR = "/content/weights/mt5-base-openasl-pruned-26025"
P = sys.executable

os.makedirs(R, exist_ok=True)
for p in (CKPT, MT5, POSES):
    if not os.path.exists(p):
        sys.exit(f"[block5] missing {p} -- run colab_setup.py first")
sys.path.insert(0, "/content/SLTedge")


def run(cmd, what):
    print(f"\n[block5] {what}")
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit(f"[block5] FAILED: {what}")


# ---------------------------------------------------------------- 1. corrected test-split number
for cap in (100, 64):
    out = f"{R}/eval_test_pruned_traindev_cap{cap}.json"
    if os.path.exists(out):
        print(f"[block5] test cap{cap}: already done")
        continue
    run(f"{P} -u -m unisign.eval_openasl --ckpt {CKPT} --mt5 {MT5} --poses {POSES} "
        f"--labels data/openasl_labels/labels.test --num-beams 4 --max-new-tokens {cap} "
        f"--batch-size 8 --out {out} >> {R}/block5.log 2>&1", f"test-split eval, cap {cap}")

# ---------------------------------------------------------------- 2. pre-pruned directory at 26,025
if os.path.exists(f"{PRUNED_DIR}/keep_ids.json"):
    print("[block5] pruned dir: already built")
else:
    print("\n[block5] building pre-pruned mT5 directory at 26,025")
    import shutil

    import torch
    from unisign.model import PoseOnlyUniSign
    keep = json.load(open("/content/SLTedge/results/openasl_vocab_keep_ids_traindev.json"))
    m = PoseOnlyUniSign(MT5, keep_ids=keep)          # prunes weights AND wraps the tokenizer
    os.makedirs(PRUNED_DIR, exist_ok=True)
    m.mt5_model.save_pretrained(PRUNED_DIR)
    # save_pretrained writes the WEIGHTS only. The tokenizer files must come from the unpruned
    # source -- PrunedTokenizer is a runtime wrapper with no save_pretrained -- so the directory
    # carries the original 250k spiece.model and the remap lives in keep_ids.json beside it.
    for f in ("spiece.model", "special_tokens_map.json", "tokenizer_config.json"):
        src = os.path.join(MT5, f)
        if os.path.exists(src):
            shutil.copy(src, PRUNED_DIR)
    json.dump(keep, open(f"{PRUNED_DIR}/keep_ids.json", "w"))
    cfg = json.load(open(f"{PRUNED_DIR}/config.json"))
    print(f"[block5] wrote {PRUNED_DIR}: config vocab_size={cfg['vocab_size']}, "
          f"keep_ids {len(keep)}, "
          f"{sum(os.path.getsize(os.path.join(PRUNED_DIR,f)) for f in os.listdir(PRUNED_DIR))/1e6:.0f} MB")
    del m
    torch.cuda.empty_cache()

# ---------------------------------------------------------------- 3. verify the fast path
print("\n[block5] verifying the fast path against the ordinary load path")
import torch  # noqa: E402

from common.pose_to_unisign import collate, load_pkl, to_model_inputs  # noqa: E402
from unisign.model import PoseOnlyUniSign, load_model  # noqa: E402

labels = json.loads("{}")
import gzip  # noqa: E402
import pickle  # noqa: E402
labels = pickle.load(gzip.open("/content/SLTedge/data/openasl_labels/labels.dev", "rb"))
names = [n for n in list(labels)[:8]
         if os.path.exists(os.path.join(POSES, n.replace(".mp4", ".pkl")))]


def sentences(model):
    batch, bn = [], []
    for n in names:
        kps, scs, _ = load_pkl(os.path.join(POSES, n.replace(".mp4", ".pkl")))
        inputs, _ = to_model_inputs(kps, scs, 256, fps_ratio=1.0)
        batch.append(inputs); bn.append(n)
    src = collate(batch, bn)
    dev = next(model.parameters()).device
    src = {k: (v.to(dev).float() if torch.is_tensor(v) and v.is_floating_point() else v)
           for k, v in src.items()}
    with torch.no_grad():
        emb, mask = model.build_encoder_inputs(src)
        out = model.mt5_model.generate(inputs_embeds=emb, attention_mask=mask,
                                       max_new_tokens=64, num_beams=4)
    return model.mt5_tokenizer.batch_decode(out, skip_special_tokens=True)


ref = load_model(CKPT, MT5, device="cuda")          # ordinary path: materialise 250k, then slice
ref_txt = sentences(ref)
del ref; torch.cuda.empty_cache()

fast = PoseOnlyUniSign(PRUNED_DIR, device="cuda")   # fast path: weights already pruned on disk
fast.attach_pruned_tokenizer()                      # reads keep_ids.json from the directory
sd = torch.load(CKPT, map_location="cpu")
missing, unexpected = fast.load_state_dict(sd.get("model", sd), strict=False)
fast = fast.to("cuda").eval()
fast_txt = sentences(fast)

ok = ref_txt == fast_txt
print(f"\n[block5] fast path matches ordinary path: {ok}")
for n, a, b in zip(names, ref_txt, fast_txt):
    mark = "  " if a == b else "**"
    print(f"{mark} {n}\n     ordinary: {a}\n     fast    : {b}")

json.dump({"match": ok, "names": names, "ordinary": ref_txt, "fast": fast_txt,
           "missing_keys": len(missing), "unexpected_keys": len(unexpected)},
          open(f"{R}/fastpath_verify.json", "w"), indent=1)

print("\n[block5] DONE")
for f in sorted(os.listdir(R)):
    print("   ", f)
if not ok:
    print("\n*** FAST PATH DOES NOT REPRODUCE THE REFERENCE TEXT -- do not use it. ***")
