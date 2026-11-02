#!/usr/bin/env python3
"""Block 6: the adaptation 2x2 at 24 fps -- the shipped operating point.

    !python colab_setup.py && python colab_block6.py

Block 4 ran this 2x2 at 16 fps. 24 fps is the rate we actually ship (3.8: beam 4 @ 24 fps, -17 %
energy for -0.07 BLEU-4), and no adaptation result exists there at all. This fills that cell.

THE DESIGN IS A 2x2, NOT A SINGLE RUN
    {un-adapted, adapted} x {source rate, 24 fps}
"Adapted" always means adapted AT THAT RATE, so the adapted/24 fps cell needs its own training run.
The difference-in-differences asks whether adaptation buys more under the 24 fps shift than it buys
in general; the within-rate delta asks whether it helps at 24 fps at all. The second is the weaker
question and the one more likely to come back established -- at 16 fps the DiD was +0.32 BLEU-4
[-0.55, +1.18], not established, while the within-rate ROUGE-L gain was +1.044 [+0.247, +1.802].

WHY BOTH SOURCE-RATE CELLS ARE RETRAINED HERE RATHER THAN REUSED FROM BLOCK 4
Block 4's four cells came from one Colab environment, on purpose: GPU reductions are
non-deterministic, so a retrained checkpoint is not bit-identical and mixing environments inside one
2x2 puts an uncontrolled difference on the diagonal. The un-adapted cells DO reproduce bit-exactly
across runtimes (L17: 0.00e+00 on both metrics -- no training is involved, beam search over a fixed
checkpoint is deterministic), and this script checks that as a validity gate. The ADAPTED source
cell involves training and does not, so it is retrained. Cost: one extra 13-minute run.

READ fps_coverage BEFORE READING THE RESULT
24 fps is a much weaker treatment than 16 fps on this data (results/fps_coverage_dev.json):
frames reaching the model fall 10.0 % at 24 fps against 32.8 % at 16 fps, and 25.4 % of dev clips
reach the model with an unchanged frame count because max_length 256 already caps them. A null DiD
here is therefore the expected outcome, and is evidence about a small treatment, not about
adaptation in general.

DRIVE
--out-dir points at Drive so last.pt and log.jsonl survive a recycled runtime; --full-out keeps the
570 MB adapted_full.pth on local disk, because it is regenerable from last.pt in seconds (re-running
this script with a complete last.pt skips the epoch loop and just rewrites it). Eval JSONs go
straight to Drive -- they are small, and they are the only artifact that cannot be recreated cheaply.
Two Colab runtimes were lost mid-project with checkpoints on /content; that is what this avoids.

Outputs under <drive>/:
    runs/<name>/last.pt, log.jsonl, train.log     training state, resumable
    evals/eval_dev_*.json                         the 2x2 cells, 967 predictions each
    gate_unadapt_src.json                         bit-exactness check against block 4
"""
import argparse
import json
import os
import subprocess
import sys

_rc = getattr(sys.stdout, "reconfigure", None)
if _rc:
    _rc(line_buffering=True)

ap = argparse.ArgumentParser()
ap.add_argument("--drive", default="/content/drive/MyDrive/SLTedge/block6")
ap.add_argument("--repo", default="/content/SLTedge")
ap.add_argument("--seeds", default="42", help="comma-separated; 42 gives the 2x2, add 43,44 for a seed band")
ap.add_argument("--limit", type=int, default=20000, help="train clips (block 4 and L17's 20000 rung used 20000)")
ap.add_argument("--fps", type=float, default=24.0)
ap.add_argument("--skip-cap64", action="store_true", help="skip the max_new_tokens 64 vs 100 check")
ap.add_argument("--no-tf32", action="store_true",
                help="force true FP32 via NVIDIA_TF32_OVERRIDE=0. Default is the PyTorch/driver "
                     "default, which is what block 4 ran at -- see the TF32 note in the header.")
a = ap.parse_args()

RATE = f"fps{a.fps:g}"
R, P = "/content/runs", sys.executable
CKPT, MT5, POSES = "/content/pruned_traindev.pth", "/content/weights/mt5-base", "/content/poses_train"
LABELS_TR, LABELS_DEV = "data/openasl_labels/labels.train", "data/openasl_labels/labels.dev"
E = f"{a.drive}/evals"
SEEDS = [int(s) for s in a.seeds.split(",")]

if not os.path.isdir("/content/drive/MyDrive"):
    sys.exit("[block6] Drive is not mounted -- run the drive.mount() cell first.\n"
             "         Outputs must persist to Drive; two runtimes were already lost without it.")
for p in (CKPT, MT5, POSES):
    if not os.path.exists(p):
        sys.exit(f"[block6] missing {p} -- run colab_setup.py first")
os.makedirs(E, exist_ok=True)
os.makedirs(R, exist_ok=True)
print(f"[block6] rate {RATE}, seeds {SEEDS}, TF32 "
      f"{'FORCED OFF (NVIDIA_TF32_OVERRIDE=0)' if a.no_tf32 else 'PyTorch default, as block 4'}")


# TF32: default to whatever PyTorch defaults to, because that is what block 4 ran at.
# Block 4 ran on a Colab A100 (LM-TRACK-REPORT-2026-09-30.md section 8.4). RESULTS.md's L15 header
# says "All runs: Colab Tesla T4", but that header covers the L15.1/L15.2 recipe sweep whose logs
# are in results/colab_runs/ -- none of block 4's four run names appear there, and those logs are at
# lr 1e-4, the rejected recipe. REPLY-POWER-CAVEAT-STANDS-2026-10-05.md reads that header as
# covering block 4; it does not. See the note this script prints if the gate fails.
#
# On Ampere, PyTorch leaves cudnn.allow_tf32 True and cuda.matmul.allow_tf32 False, so the ST-GCN
# convolutions ran at a 10-bit mantissa in block 4 too. Forcing FP32 here would therefore make this
# 2x2 LESS comparable, not more. --no-tf32 exists to settle it empirically: if the validity gate
# below is not bit-exact, re-run just the un-adapted source eval with it and see which matches.
ENV = "NVIDIA_TF32_OVERRIDE=0 " if a.no_tf32 else ""


def run(cmd, what):
    print(f"\n[block6] {what}", flush=True)
    if subprocess.run(ENV + cmd, shell=True).returncode != 0:
        sys.exit(f"[block6] FAILED: {what}")


def train(name, fps_arg, seed):
    """Returns the path to adapted_full.pth, training or regenerating it as needed."""
    d, full = f"{a.drive}/runs/{name}", f"{R}/{name}_adapted_full.pth"
    if os.path.exists(full):
        print(f"[block6] {name}: adapted_full.pth present, skipping")
        return full
    os.makedirs(d, exist_ok=True)
    # --resume covers all three states: nothing yet, died mid-epoch, or finished (epoch loop is then
    # empty and this only rewrites --full-out, which takes seconds).
    run(f"{P} -u -m unisign.train_adapt --ckpt {CKPT} --mt5 {MT5} --poses {POSES} "
        f"--labels {LABELS_TR} --limit {a.limit} {fps_arg} "
        f"--out-dir {d} --full-out {full} --epochs 1 --batch-size 4 --accum 2 "
        f"--num-workers 2 --freeze-bn --save-every 500 --log-every 200 --seed {seed} "
        f"--lr 1e-5 --label-smoothing 0.0 --warmup-epochs 0.1 --resume "
        f">> {d}/train.log 2>&1", f"train {name} (tail {d}/train.log)")
    return full


def evaluate(tag, ckpt, fps_arg, cap=100):
    out = f"{E}/eval_dev_{tag}_cap{cap}.json"
    if os.path.exists(out):
        print(f"[block6] {tag} cap{cap}: already evaluated, skipping")
        return out
    run(f"{P} -u -m unisign.eval_openasl --ckpt {ckpt} --mt5 {MT5} --poses {POSES} "
        f"--labels {LABELS_DEV} --num-beams 4 --max-new-tokens {cap} --batch-size 8 {fps_arg} "
        f"--out {out} >> {a.drive}/block6.log 2>&1", f"eval {tag} cap{cap}")
    return out


def gate(fresh):
    """Un-adapted cells involve no training, so they must reproduce block 4 exactly. Report, do not
    abort: a mismatch invalidates the comparison to block 4, not this run's internally-consistent
    2x2, and the size of the mismatch is the diagnostic worth keeping."""
    ref_p = f"{a.repo}/results/block4/eval_dev_unadapt_src_cap100.json"
    if not os.path.exists(ref_p):
        print(f"[block6] gate: no {ref_p}, skipping")
        return
    ref, new = json.load(open(ref_p)), json.load(open(fresh))
    db = new["bleu"]["bleu4"] - ref["bleu"]["bleu4"]
    dr = new["rouge_l"] - ref["rouge_l"]
    same = sum(x == y for x, y in zip(ref["preds"], new["preds"]))
    g = {"ref": ref_p, "d_bleu4": db, "d_rouge_l": dr,
         "preds_identical": same, "n": len(ref["preds"]),
         "bit_exact": db == 0.0 and dr == 0.0 and same == len(ref["preds"])}
    json.dump(g, open(f"{a.drive}/gate_unadapt_src.json", "w"), indent=1)
    print("\n" + "=" * 72)
    print(f"[block6] VALIDITY GATE vs block 4 un-adapted/source ({len(ref['preds'])} clips)")
    print(f"           d BLEU-4 {db:+.6e}   d ROUGE-L {dr:+.6e}   identical preds {same}/{len(ref['preds'])}")
    print("           " + ("BIT-EXACT -- this 2x2 is comparable to block 4's."
                           if g["bit_exact"] else
                           "NOT bit-exact. This run's 2x2 is still internally consistent;\n"
                           "           comparisons to block 4 are not. Cheapest next test (~2 min): "
                           "delete\n           evals/eval_dev_unadapt_src_cap100.json and re-run this "
                           "script with --no-tf32.\n           If that one matches, block 4 ran at "
                           "true FP32 and the whole 2x2 wants --no-tf32."))
    print("=" * 72)


# --- training ----------------------------------------------------------------------------------
ctrl = train(f"ctrl_src_s{SEEDS[0]}", "", SEEDS[0])
adapted = {s: train(f"{RATE}_s{s}", f"--fps {a.fps:g}", s) for s in SEEDS}

# --- the 2x2 -------------------------------------------------------------------------------------
fresh_src = evaluate("unadapt_src", CKPT, "")
evaluate(f"unadapt_{RATE}", CKPT, f"--fps {a.fps:g}")
evaluate("adapt_src", ctrl, "")
evaluate(f"adapt_{RATE}_s{SEEDS[0]}", adapted[SEEDS[0]], f"--fps {a.fps:g}")

# Block 4 found cap 64 == cap 100 on the adapted 16 fps model; 24 fps keeps longer sequences, so the
# equivalence is re-checked rather than assumed.
if not a.skip_cap64:
    evaluate(f"adapt_{RATE}_s{SEEDS[0]}", adapted[SEEDS[0]], f"--fps {a.fps:g}", cap=64)

for s in SEEDS[1:]:
    evaluate(f"adapt_{RATE}_s{s}", adapted[s], f"--fps {a.fps:g}")

gate(fresh_src)

print(f"\n[block6] DONE. eval JSONs in {E}:")
for f in sorted(os.listdir(E)):
    print("   ", f)
print(f"\nNext:  python -m unisign.adapt_ci --evals {E} -n 1000 --rate {RATE} \\\n"
      f"           --out {a.drive}/adapt_ci_dev_{RATE}.json")
