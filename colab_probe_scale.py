#!/usr/bin/env python3
"""Scaling probe: how much adaptation data does this harness actually need?

    !python colab_setup.py && python colab_probe_scale.py

WHY THIS EXISTS
J9 adapts the pose encoder to our own extractor's keypoints, and the only training data available
for it is the ~920-clip dev split -- the train split would need 20,000 YouTube videos through the
board extractor (~140 GB, days of board time), which is out of reach. L15's adaptation runs, the
ones that worked, used 20,000 clips. Nobody knows whether ~920 is enough, and the two points we
have bracket it uselessly: 300 clips moved nothing (by design), 20,000 recovered +1.04 ROUGE-L.

So measure it, on the axis where the answer at 20,000 is already known. This trains the SAME
frame-rate adaptation as L15.3 at a ladder of training-set sizes and scores every one against the
same un-adapted baseline. If the curve is already flat near zero at ~1,000, J9 at ~920 is
predictably null and the board extraction buys nothing. If it is climbing, J9 is worth running.

WHAT IT DOES NOT TELL US
This measures scale on the FRAME-RATE shift using the authors' poses. J9's shift is POSE SOURCE,
which is a different distribution change. A positive curve makes J9 worth running; it does not
predict J9's result. Said plainly here because it is the easy mistake to make with this number.

WHY THE SIZES ARE NESTED
train_adapt.py --limit N takes the FIRST N available train clips, so every smaller set is a subset
of every larger one. The ladder is therefore a true scaling curve and not five different samples,
and a difference between two rungs cannot be a draw effect.

WHY 20,000 IS IN THE LADDER RATHER THAN QUOTED FROM BLOCK 4
Mixing a number from a wiped runtime with numbers from this one would mean the curve's anchor and
its body came from different environments, and GPU reductions are non-deterministic. Retraining it
here costs ~15 minutes and makes the whole curve one environment. It also independently replicates
L15.3's headline, which nothing has done yet.

WHAT GOES TO DRIVE, AND WHAT DELIBERATELY DOES NOT
Colab loses /content when the runtime is recycled, so anything expensive to recreate goes to Drive.
Here that is ONLY the eval JSONs, the CIs and the curve -- about 1 MB for the whole ladder. They are
the results; everything else is a means of producing them.

last.pt does NOT, despite being "only" the trainable params: with optimizer and scheduler state it
is 70 MB per rung, 350 MB across the ladder, and this quota was exceeded mid-run on 2026-10-05 with
two rungs' worth written. It buys mid-epoch resume, which is worth little here because the rungs are
individually cheap -- the largest is ~15 minutes and the smallest under a minute. Trading 350 MB of a
full quota for that is a bad deal, and a failed Drive write aborts the run outright, which is strictly
worse than retraining a rung.

`adapted_full.pth` does NOT. It is the FULL state dict at ~570 MB, not the trainable slice -- an
earlier draft of this file claimed otherwise and would have put ~2.9 GB on Drive across the ladder,
which is how the project lost a Drive folder once already. It goes to local disk via --full-out, and
the per-rung skip is keyed on the EVAL JSON rather than on the checkpoint, so a recycled runtime that
loses /content skips any rung already scored instead of retraining it. A rung that was trained but
not yet scored resumes from last.pt on Drive, which is the case worth protecting.

The 32 GB pose archive also stays local: it re-downloads in minutes and Drive handles tens of
thousands of small files badly.

Re-runnable: every step is skipped when its output exists, so a disconnect costs only the step that
was in flight. Training additionally resumes mid-epoch from last.pt.
"""
import json
import os
import subprocess
import sys

# Line-buffer stdout: under nohup with output redirected, Python block-buffers and a healthy run
# looks hung for minutes. This has caused three false "it died" diagnoses on this project.
sys.stdout.reconfigure(line_buffering=True)

P = sys.executable
CKPT = "/content/pruned_traindev.pth"
MT5 = "/content/weights/mt5-base"
POSES = "/content/poses_train"
LABELS_TR = "data/openasl_labels/labels.train"
LABELS_DEV = "data/openasl_labels/labels.dev"

# The ladder. 920 is in it because that is J9's actual dev-split size -- the rung the decision
# turns on -- rather than a round number near it.
SIZES = (500, 920, 2000, 5000, 20000)
SEED = 42
FPS = "--fps 16"          # L15.3's axis: the shift the 20,000-clip answer was measured on
CAP = 100                 # matches results/adapt_ci_dev.json, so the anchor is comparable

DRIVE = "/content/drive/MyDrive"
OUT = f"{DRIVE}/sltedge/probe_scale" if os.path.isdir(DRIVE) else "/content/runs/probe_scale"
E = f"{OUT}/evals"

if not os.path.isdir(DRIVE):
    print("[probe] WARNING: Drive is not mounted, writing to /content, which is lost when the "
          "runtime is recycled. Mount with:\n"
          "    from google.colab import drive; drive.mount('/content/drive')\n")
os.makedirs(E, exist_ok=True)
for p in (CKPT, MT5, POSES):
    if not os.path.exists(p):
        sys.exit(f"[probe] missing {p} -- run colab_setup.py first")

# Refuse to run twice. A `pkill` that silently misses leaves the old run's children alive, and the
# restart then races them over the same output paths -- this happened on 2026-10-05 with two
# bootstrap_ci processes writing one ci_n500.json, and earlier in the project with five trainers on
# one GPU. The lock holds a pid; a lock whose pid is gone is stale and reclaimed.
LOCK = "/content/probe.lock"
if os.path.exists(LOCK):
    try:
        old = int(open(LOCK).read().strip())
        os.kill(old, 0)                       # raises unless that pid is alive
        sys.exit(f"[probe] ABORT: already running as pid {old}. Stop it first:\n"
                 f"    !kill {old}; pkill -f unisign.bootstrap_ci; pkill -f unisign.train_adapt")
    except (ValueError, ProcessLookupError, PermissionError):
        print(f"[probe] stale lock from a dead pid, reclaiming {LOCK}")
open(LOCK, "w").write(str(os.getpid()))
import atexit
atexit.register(lambda: os.path.exists(LOCK) and os.remove(LOCK))


def run(cmd, what):
    print(f"\n[probe] {what}", flush=True)
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit(f"[probe] FAILED: {what}")


FULL = "/content/full"      # local: the 570 MB checkpoints, deliberately not on Drive
WORK = "/content/work"      # local: per-rung out-dir (last.pt, 70 MB, and the training log)
os.makedirs(FULL, exist_ok=True)
os.makedirs(WORK, exist_ok=True)


def train(n):
    """Train rung n. Returns the path to its full checkpoint, on LOCAL disk."""
    d, full = f"{WORK}/fps16_n{n}", f"{FULL}/fps16_n{n}.pth"
    if os.path.exists(full):
        print(f"[probe] n={n}: full checkpoint present locally, skipping training")
        return full
    os.makedirs(d, exist_ok=True)
    run(f"{P} -u -m unisign.train_adapt --ckpt {CKPT} --mt5 {MT5} --poses {POSES} "
        f"--labels {LABELS_TR} --limit {n} {FPS} "
        f"--out-dir {d} --full-out {full} --epochs 1 --batch-size 4 --accum 2 "
        f"--num-workers 2 --freeze-bn --save-every 200 --log-every 100 --seed {SEED} "
        f"--lr 1e-5 --label-smoothing 0.0 --warmup-epochs 0.1 --resume "
        f"> {d}/train.log 2>&1", f"train n={n}")
    return full


def evaluate(tag, ckpt_fn):
    """ckpt_fn is a callable so training is not triggered when the eval already exists."""
    out = f"{E}/eval_dev_{tag}.json"
    if os.path.exists(out):
        print(f"[probe] {tag}: already evaluated, skipping")
        return out
    run(f"{P} -u -m unisign.eval_openasl --ckpt {ckpt_fn()} --mt5 {MT5} --poses {POSES} "
        f"--labels {LABELS_DEV} --num-beams 4 --max-new-tokens {CAP} --batch-size 8 {FPS} "
        f"--expect-n 967 --out {out} >> {OUT}/probe.log 2>&1", f"eval {tag}")
    return out


# --- the un-adapted 16 fps baseline, scored in THIS environment ---------------------------------
# Every rung is compared against this one file, so the comparison is paired and the baseline is not
# imported from a different runtime.
base = evaluate("unadapt_fps16", lambda: CKPT)

# --- phase 1: all GPU work, serially ------------------------------------------------------------
# Training and eval need the GPU and must not overlap -- two CUDA processes on one device is how
# this project once ended up with five trainers fighting over a T4.
rows = []
for n in SIZES:
    # The eval JSON is the skip key: a recycled runtime that lost /content will not retrain a rung
    # it has already scored, because train() is only reached through the lambda.
    ev = evaluate(f"adapt_fps16_n{n}", lambda n=n: train(n))
    rows.append((n, ev, f"{OUT}/ci_n{n}.json"))

# --- phase 2: all bootstraps, in parallel ---------------------------------------------------------
# Deliberately after phase 1 rather than interleaved. bootstrap_ci is pure-Python ROUGE-L over 967
# clips x 1000 resamples -- CPU-bound and minutes long -- so interleaving leaves the GPU idle waiting
# on it. The rungs are independent and share no state but the baseline file, which is read-only here.
todo = [(n, ev, ci) for n, ev, ci in rows if not os.path.exists(ci)]
if todo:
    import concurrent.futures as cf
    workers = max(1, min(len(todo), (os.cpu_count() or 2)))
    print(f"\n[probe] {len(todo)} bootstrap(s) over {workers} worker(s)", flush=True)

    def boot(job):
        n, ev, ci = job
        # Each writes its own log: concurrent appends to one file interleave into nonsense.
        return subprocess.run(f"{P} -u -m unisign.bootstrap_ci {base} {ev} --out {ci} "
                              f"> {OUT}/ci_n{n}.log 2>&1", shell=True).returncode, n

    with cf.ThreadPoolExecutor(workers) as pool:
        for rc, n in pool.map(boot, todo):
            if rc != 0:
                sys.exit(f"[probe] FAILED: bootstrap n={n} (see {OUT}/ci_n{n}.log)")
            print(f"[probe] bootstrap n={n} done", flush=True)

# --- the curve ----------------------------------------------------------------------------------
print("\n[probe] SCALING CURVE -- adaptation gain vs training-set size")
print("All rows: 16 fps, seed 42, lr 1e-5, ls 0.0, warmup 0.1, 1 epoch, scored on 967 dev clips,")
print("paired against the same un-adapted 16 fps baseline.\n")
b = json.load(open(base))
print(f"{'n_train':>8}  {'BLEU-4':>7}  {'d BLEU-4':>22}  {'ROUGE-L':>8}  {'d ROUGE-L':>22}")
print(f"{'un-adapt':>8}  {b['bleu']:>7.2f}  {'(baseline)':>22}  {b['rouge_l']:>8.2f}  {'(baseline)':>22}")
summary = {"baseline": {"bleu4": b["bleu"], "rouge_l": b["rouge_l"]}, "rungs": {}}
for n, ev, ci in rows:
    e, c = json.load(open(ev)), json.load(open(ci))
    # bootstrap_ci.py writes {"bleu4": {"delta":…, "ci":[lo,hi]}, "rouge_l": {…}} -- verified
    # against results/ci_n400_posesub_fps24_lm.json, not assumed.
    def fmt(m):
        d = c[m]
        return f"{d['delta']:+.2f} [{d['ci'][0]:+.2f}, {d['ci'][1]:+.2f}]"
    print(f"{n:>8}  {e['bleu']:>7.2f}  {fmt('bleu4'):>22}  {e['rouge_l']:>8.2f}  {fmt('rouge_l'):>22}")
    summary["rungs"][n] = {"bleu4": e["bleu"], "rouge_l": e["rouge_l"], "ci": c}
json.dump(summary, open(f"{OUT}/scaling_curve.json", "w"), indent=1)

print(f"\n[probe] wrote {OUT}/scaling_curve.json")
print("\nHOW TO READ IT. The 20,000 rung should land near L15.3's +1.04 ROUGE-L [+0.25, +1.80]; if it")
print("does not, this environment differs from Block 4's and the curve is not comparable to it.")
print("The decision rung is n=920. A gain there whose interval excludes zero makes J9 worth the")
print("board extraction. A flat or zero rung at 920 with a climbing curve above it means J9 at dev")
print("scale is predictably null, and a null would then be about data volume, not about adaptation.")
