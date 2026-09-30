#!/usr/bin/env python3
"""Block 4 batch: seed variance (guide row 4.3) + adapted/un-adapted evals with CIs (row 4.4).

    !python colab_setup.py && python colab_block4.py

Self-healing and re-runnable: every step is skipped when its output already exists, so a Colab
disconnect costs only the step that was in flight. Run colab_setup.py first (or let this fail
loudly on a missing input) -- it rebuilds poses, weights and the pruned checkpoint.

WHY EVERYTHING IS RETRAINED HERE RATHER THAN REUSING RESCUED CHECKPOINTS
L15.3's four cells came from two Colab runtimes that were later wiped; only the 16 fps checkpoint
survived, on the Mac. Mixing one rescued checkpoint with freshly trained ones would mean the four
cells of a 2x2 no longer come from one environment, and GPU reductions are non-deterministic so a
retrain is not bit-identical. Everything here is retrained in one batch so every number in the
final table traces to a checkpoint produced by this script.

WHY cap 100 AND cap 64
L3.2 measured max_new_tokens 64 as bit-identical to 100 on the *un-adapted* pruned model (0 of 976
sentences changed). The adapted model visibly hits the cap -- one dev sample ends mid-phrase at
"where they can establish a" -- so that equivalence is not established for adapted checkpoints, and
truncation depresses both metrics. Each adapted model is scored at both caps to measure it.

Outputs (all under /content/runs):
    ctrl_src_s42/, fps16_s{42,43,44}/   training runs, each with last.pt + adapted_full.pth
    evals/eval_dev_*.json               full predictions for paired bootstrapping
"""
import os
import subprocess
import sys

R = "/content/runs"
E = f"{R}/evals"
P = sys.executable
CKPT = "/content/pruned_traindev.pth"
MT5 = "/content/weights/mt5-base"
POSES = "/content/poses_train"
LABELS_TR = "data/openasl_labels/labels.train"
LABELS_DEV = "data/openasl_labels/labels.dev"
LIMIT = 20000
SEEDS = (42, 43, 44)

os.makedirs(E, exist_ok=True)
for p in (CKPT, MT5, POSES):
    if not os.path.exists(p):
        sys.exit(f"[block4] missing {p} -- run colab_setup.py first")


def run(cmd, what):
    print(f"\n[block4] {what}", flush=True)
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit(f"[block4] FAILED: {what}")


def train(name, fps, seed):
    d = f"{R}/{name}"
    if os.path.exists(f"{d}/adapted_full.pth"):
        print(f"[block4] {name}: already trained, skipping")
        return d
    os.makedirs(d, exist_ok=True)
    run(f"{P} -u -m unisign.train_adapt --ckpt {CKPT} --mt5 {MT5} --poses {POSES} "
        f"--labels {LABELS_TR} --limit {LIMIT} {fps} "
        f"--out-dir {d} --epochs 1 --batch-size 4 --accum 2 "
        f"--num-workers 2 --freeze-bn --save-every 500 --log-every 200 --seed {seed} "
        f"--lr 1e-5 --label-smoothing 0.0 --warmup-epochs 0.1 --resume "
        f"> {d}/train.log 2>&1", f"train {name}")
    return d


def evaluate(tag, ckpt, fps, cap):
    out = f"{E}/eval_dev_{tag}_cap{cap}.json"
    if os.path.exists(out):
        print(f"[block4] {tag} cap{cap}: already evaluated, skipping")
        return
    run(f"{P} -u -m unisign.eval_openasl --ckpt {ckpt} --mt5 {MT5} --poses {POSES} "
        f"--labels {LABELS_DEV} --num-beams 4 --max-new-tokens {cap} --batch-size 8 {fps} "
        f"--out {out} >> {R}/block4.log 2>&1", f"eval {tag} cap{cap}")


# --- training: one source-rate control, three 16 fps seeds -------------------------------------
ctrl = train("ctrl_src_s42", "", 42)
fps16 = {s: train(f"fps16_s{s}", "--fps 16", s) for s in SEEDS}

# --- evals: the 2x2, plus the cap comparison on the adapted models ------------------------------
# Un-adapted cells use the L5.4 pruned checkpoint directly.
evaluate("unadapt_src", CKPT, "", 100)
evaluate("unadapt_fps16", CKPT, "--fps 16", 100)
evaluate("adapt_src", f"{ctrl}/adapted_full.pth", "", 100)
evaluate("adapt_fps16_s42", f"{fps16[42]}/adapted_full.pth", "--fps 16", 100)

# cap 64 on one adapted model, to measure whether L3.2's equivalence survives adaptation
evaluate("adapt_fps16_s42", f"{fps16[42]}/adapted_full.pth", "--fps 16", 64)

# remaining seeds, for the seed band (row 4.3)
for s in (43, 44):
    evaluate(f"adapt_fps16_s{s}", f"{fps16[s]}/adapted_full.pth", "--fps 16", 100)

print("\n[block4] DONE. eval JSONs:")
for f in sorted(os.listdir(E)):
    print("   ", f)
print("\nDownload these plus every runs/*/log.jsonl before the runtime goes away.")
