#!/usr/bin/env python3
"""J9 step 2: adapt the pose encoder to OUR keypoints, with the control arm. Colab.

    !python /content/SLTedge/colab_j9_step2.py

Runs three arms and three paired bootstraps, all in ONE checkpoint lineage, then prints the
difference-in-differences that is the contrast of record.

WHY ALL THREE ARMS ARE SCORED HERE AND NOT AGAINST THE BOARD'S n=931 NUMBERS
    The board arm (results/eval_n931_pruned_ours_fps24.json, 21.73 BLEU-4) uses
    weights/openasl_pose_only_slt_pruned.pth with weights/mt5-base-openasl-pruned -- keep set
    **26,078**, built over train+dev+TEST. colab_setup.py builds a different, leak-free checkpoint:
    /content/pruned_traindev.pth over results/openasl_vocab_keep_ids_traindev.json, keep set
    **26,025**. Those two vocabularies are not interchangeable, so an adapted checkpoint trained
    here CANNOT be scored against the board's arms: the delta would conflate adaptation with a
    53-token vocabulary change and with the test-set leak. RESULTS.md L18 already measured the
    mt5 directory as a decode-protocol axis that changes outputs.
    So every number this script compares is produced in this lineage. Absolute BLEU-4 will NOT
    equal 21.73 and is not meant to. The deliverable is a delta, and a lineage offset shared by all
    three arms cancels in every delta taken below.

THE CONTRAST OF RECORD IS adapted-on-OURS minus adapted-on-THEIRS, not adapted minus un-adapted.
    ASK-ADAPT-TO-OUR-POSES is explicit and L15.2 is why: fine-tuning this harness on dev clips
    improves the model by +0.18 BLEU-4 / +0.70 ROUGE-L with NO distribution shift to adapt to. So
    "adapted beats un-adapted" does not isolate the thing J9 is testing. Arm 3 trains the identical
    recipe on the AUTHORS' dev poses and is subtracted.

THE BAR, AGREED IN WRITING BEFORE THIS RUNS (RESULTS.md L21)
    At n=931 the measured half-width on the pose axis is +-0.778, so an effect must recover
    **55.3 %** of the -1.4077 deficit to come back established. L17's scaling probe sizes the
    expected recovery at ~39 %, which is BELOW that bar. A null here is therefore the predicted
    outcome and does not license "adaptation does not work" -- it was agreed as the expectation
    before the run, and L21 declined a one-sided test that would have moved the bar to 48.5 %.

RECIPE, each flag for a measured reason (ASK-ADAPT-TO-OUR-POSES)
    --label-smoothing 0.0   NOT the 0.2 default. L15.1: at 0.2 over a 26 K vocabulary the loss parks
                            on a ~3.35 smoothing floor that reads as a training curve while the
                            model gets WORSE. The single most important flag here.
    --lr 1e-5               the 1e-4 default is the one that degraded in L15.2.
    --fps 24                adapt at the rate we deploy and the rate the test arm is scored at.
                            NOTE this differs from L17's probe, which measured scaling at 16 fps;
                            L17 already records the transfer across that axis as unestablished, and
                            matching the EVAL rate matters more than matching the probe.
    --save-every 200        Colab kills sessions mid-epoch and L15 lost a run exactly that way.
                            save_every checkpoints mid-epoch and --resume continues inside it.
    --freeze-bn, batch 4
    x accum 2               same as colab_probe_scale.py, so L17's curve and this run share the
                            optimiser path.

WHAT GOES TO DRIVE AND WHAT DOES NOT
    Drive carries only what cannot be recreated cheaply: the eval JSONs, the CIs, the summary. The
    ~570 MB full checkpoints go to /content/full via --full-out and the 70 MB last.pt to
    /content/work -- a Drive quota failure killed a run on 2026-10-05 and is why --full-out exists.
    The authors' dev poses are re-fetchable from a public source, so they stay on local disk too.

RE-RUNNABLE. Every stage's output file is its own skip key, and training is reached only through a
lambda, so a recycled runtime that lost /content will not retrain an arm it has already scored.
"""
import json
import os
import subprocess
import sys

# Line-buffer stdout: under nohup with output redirected Python block-buffers and a healthy run
# looks hung for minutes. That has caused three false "it died" diagnoses on this project.
# getattr, not a bare call: under IPython sys.stdout is an ipykernel OutStream with NO reconfigure,
# so importing this module inside a Colab cell dies with AttributeError before anything runs. Hit for
# real 2026-10-07 importing colab_setup to skip its unused archive step. Under `!python file.py`
# stdout is a real TextIOWrapper and this behaves exactly as before; IPython already line-buffers.
_rc = getattr(sys.stdout, "reconfigure", None)
if _rc:
    _rc(line_buffering=True)

P = sys.executable

# Colab defaults, each overridable by env so the guards can be exercised off-Colab against fixtures
# (which is how the short-extract abort below was actually tested rather than eyeballed).
def _p(var, default):
    return os.environ.get(var, default)


REPO = _p("J9_REPO", "/content/SLTedge")
CKPT = _p("J9_CKPT", "/content/pruned_traindev.pth")
MT5 = _p("J9_MT5", "/content/weights/mt5-base")

POSES_DEV_OURS = _p("J9_DEV_OURS", "/content/poses_dev_ours")      # pkl_dev_rtmw_fp16.tar   (918)
POSES_DEV_THEIRS = _p("J9_DEV_THEIRS", "/content/poses_dev_theirs")  # authors' dev, HTTP fetch
POSES_TEST_OURS = _p("J9_TEST_OURS", "/content/poses_test_ours")   # pkl_split_rtmw_fp16.tar (931)
OUT_ROOT = _p("J9_OUT", "")                   # set to force an output root, else Drive-or-/content
LOCAL = _p("J9_LOCAL", "/content")            # where full/ and work/ live

LABELS_DEV = f"{REPO}/data/openasl_labels/labels.dev"
LABELS_TEST = f"{REPO}/data/openasl_labels/labels.test"
MAN_DEV = _p("J9_MAN_DEV", f"{REPO}/results/pkl_dev_rtmw_fp16.manifest")
MAN_TEST = _p("J9_MAN_TEST", f"{REPO}/results/pkl_split_rtmw_fp16.manifest")


# DERIVED from the manifest, not hardcoded. These were 918 and 931 literals, which made the preflight
# assert "the extract is complete" when the invariant that actually matters is "the pose directory is
# the set we are about to train and score against". The two came apart on 2026-10-07: one dev clip
# (rlUUw27_6kM-00:17:09.233-00:17:15.433) holds 7 frames of all-NaN confidence scores, was quarantined,
# and the arm legitimately trains on 917 -- whereupon the literal 918 aborted a correct run with "a
# short extract looks exactly like this", which is precisely what it was not.
# Deriving the count keeps the guard's strength: a genuinely short extract still fails, because the
# manifest it is compared against is the one the training step then checks member-by-member.
def _manifest_n(path, fallback):
    try:
        with open(path) as fh:
            return sum(1 for ln in fh if ln.strip())
    except OSError:
        return fallback          # reported as a missing input by the preflight below


N_DEV = _manifest_n(MAN_DEV, 918)
N_TEST = _manifest_n(MAN_TEST, 931)
SEED = 42
FPS = 24

# Eval protocol. batch 1 and beam 4 match the n=931 board arm field for field, so the absolute
# numbers stay as comparable as the lineage allows. L18 measured batching as changing ~30 % of
# sentences -- a shared setting cancels in the deltas, but batch 1 costs ~25 min per arm and there
# is no reason to spend the comparability.
BEAMS, CAP, BATCH = 4, 64, 1

DRIVE = "/content/drive/MyDrive"
OUT = OUT_ROOT or (f"{DRIVE}/sltedge/j9_step2" if os.path.isdir(DRIVE) else "/content/runs/j9_step2")
E = f"{OUT}/evals"
FULL = f"{LOCAL}/full"   # local: the ~570 MB adapted checkpoints
WORK = f"{LOCAL}/work"   # local: per-arm out-dir (last.pt + training log)

if not (OUT_ROOT or os.path.isdir(DRIVE)):
    print("[j9] WARNING: Drive is not mounted, so results go to /content and are lost when the\n"
          "     runtime is recycled. Mount with:\n"
          "       from google.colab import drive; drive.mount('/content/drive')\n")
# ---------------------------------------------------------------- preflight, before any GPU work
# Deliberately BEFORE os.makedirs. An earlier draft created the output dirs first, so running this
# anywhere without /content died in makedirs with a traceback instead of printing which inputs were
# missing -- the check that exists to be readable was unreachable.
# Every missing input is reported at once. Discovering the third one an hour into a run is how this
# project lost board time repeatedly.
missing = []
for p, what in ((CKPT, "pruned checkpoint -- run colab_setup.py"),
                (MT5, "mt5-base -- run colab_setup.py"),
                (LABELS_DEV, "dev labels (in the repo)"),
                (LABELS_TEST, "test labels (in the repo)"),
                (MAN_DEV, "dev manifest (in the repo)"),
                (MAN_TEST, "test manifest (in the repo)"),
                (POSES_DEV_OURS, "OUR dev poses -- extract pkl_dev_rtmw_fp16.tar here"),
                (POSES_TEST_OURS, "OUR test poses -- extract pkl_split_rtmw_fp16.tar here")):
    if not os.path.exists(p):
        missing.append(f"  {p}  <- {what}")
if missing:
    sys.exit("[j9] ABORT: missing inputs:\n" + "\n".join(missing) +
             "\n\nThe two pose tars are the Q10 handoff. Verify each against its .sha256 BEFORE\n"
             "extracting -- a partial transfer that extracts cleanly is the RESULTS.md 2.5i class:\n"
             "    shasum -a 256 -c pkl_dev_rtmw_fp16.sha256\n"
             "    tar -xf pkl_dev_rtmw_fp16.tar -C /content/poses_dev_ours\n")


def count_pkls(d):
    return sum(1 for f in os.listdir(d) if f.endswith(".pkl"))


for d, n, tar in ((POSES_DEV_OURS, N_DEV, "pkl_dev_rtmw_fp16.tar"),
                  (POSES_TEST_OURS, N_TEST, "pkl_split_rtmw_fp16.tar")):
    got = count_pkls(d)
    if got != n:
        sys.exit(f"[j9] ABORT: {d} holds {got} pkl(s), but the manifest it will be checked against\n"
                 f"      names {n}. Either the extract is short -- re-check {tar} against its\n"
                 f"      .sha256 and extract again -- or the directory and the manifest are not the\n"
                 f"      same set, e.g. a clip was quarantined without pointing J9_MAN_DEV at the\n"
                 f"      matching manifest.")
print(f"[j9] preflight OK: {N_DEV} dev + {N_TEST} test pkls, checkpoint and labels present")

for d in (E, FULL, WORK):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------------------------- the pid lock
# Refuse to run twice. A pkill that silently misses leaves the old run's children alive and the
# restart races them over one output path -- two bootstrap_ci processes did exactly that to one
# ci_n500.json on 2026-10-05, and earlier five trainers shared one GPU.
LOCK = f"{LOCAL}/j9.lock"
if os.path.exists(LOCK):
    try:
        old = int(open(LOCK).read().strip())
        os.kill(old, 0)                        # raises unless that pid is alive
        sys.exit(f"[j9] ABORT: already running as pid {old}. Stop it first:\n"
                 f"    !kill {old}; pkill -f unisign.train_adapt; pkill -f unisign.eval_openasl")
    except (ValueError, ProcessLookupError, PermissionError):
        print(f"[j9] stale lock from a dead pid, reclaiming {LOCK}")
open(LOCK, "w").write(str(os.getpid()))
import atexit  # noqa: E402

atexit.register(lambda: os.path.exists(LOCK) and os.remove(LOCK))

os.chdir(REPO)


def bleu4_of(ev_json):
    """eval_openasl writes bleu as a DICT {bleu1..bleu4}, not a float.

    Verified against every eval vintage in results/: the 931-clip board arms, the 976-clip ceiling
    and all five probe_scale rungs. Formatting ev["bleu"] directly raises
    "unsupported format string passed to dict.__format__" -- which is exactly the bug
    colab_probe_scale.py shipped with, and it fires in the SUMMARY, after all the GPU work is done.
    Tolerates a bare float too, in case an older artifact is ever fed in.
    """
    b = ev_json["bleu"]
    return b["bleu4"] if isinstance(b, dict) else b


def run(cmd, what):
    print(f"\n[j9] {what}", flush=True)
    if subprocess.run(cmd, shell=True).returncode != 0:
        sys.exit(f"[j9] FAILED: {what}")


# ------------------------------------------------------------------- arm 3's training input only
def fetch_authors_dev():
    """Authors' dev poses, ~680 MB by HTTP range. Public source, so LOCAL disk, never Drive."""
    if os.path.isdir(POSES_DEV_THEIRS) and count_pkls(POSES_DEV_THEIRS) >= N_DEV:
        print(f"[j9] authors' dev poses present ({count_pkls(POSES_DEV_THEIRS)} pkls)")
        return
    os.makedirs(POSES_DEV_THEIRS, exist_ok=True)
    run(f"{P} -u data/openasl_pose_fetch.py --split dev "
        f"--labels-dir data/openasl_labels --out {POSES_DEV_THEIRS} "
        f"> {WORK}/fetch_theirs.log 2>&1", "fetch the authors' dev poses (~680 MB)")
    got = count_pkls(POSES_DEV_THEIRS)
    print(f"[j9] authors' dev poses: {got} pkls")
    if got == 0:
        sys.exit("[j9] ABORT: the authors' dev fetch produced 0 pkls and exited 0 -- the silent-zero\n"
                 f"      class. See {WORK}/fetch_theirs.log")


def train(tag, poses, manifest=None, expect_n=None):
    """Train one arm. Returns the full checkpoint path, on LOCAL disk.

    manifest/expect_n are guards for OUR pose set, whose contents we assert. The control arm omits
    both: the authors' dev set is whatever the public fetch yields and its count is not ours to
    pin. What must match between the arms is the RECIPE, which is identical below.
    """
    d, full = f"{WORK}/{tag}", f"{FULL}/{tag}.pth"
    if os.path.exists(full):
        print(f"[j9] {tag}: full checkpoint already on local disk, skipping training")
        return full
    os.makedirs(d, exist_ok=True)
    guard = ""
    if manifest:
        guard += f" --manifest {manifest}"
    if expect_n:
        guard += f" --expect-n {expect_n}"
    run(f"{P} -u -m unisign.train_adapt --ckpt {CKPT} --mt5 {MT5} --poses {poses} "
        f"--labels {LABELS_DEV}{guard} --fps {FPS} "
        f"--out-dir {d} --full-out {full} --epochs 1 --batch-size 4 --accum 2 --num-workers 2 "
        f"--freeze-bn --save-every 200 --log-every 100 --seed {SEED} "
        f"--lr 1e-5 --label-smoothing 0.0 --warmup-epochs 0.1 --amp --resume "
        f"> {d}/train.log 2>&1", f"train {tag}")
    return full


def evaluate(tag, ckpt_fn):
    """Score an arm on OUR test poses. ckpt_fn is a callable so an existing eval never trains."""
    out = f"{E}/eval_test_{tag}.json"
    if os.path.exists(out):
        print(f"[j9] {tag}: already evaluated, skipping")
        return out
    run(f"{P} -u -m unisign.eval_openasl --ckpt {ckpt_fn()} --mt5 {MT5} --poses {POSES_TEST_OURS} "
        f"--labels {LABELS_TEST} --num-beams {BEAMS} --max-new-tokens {CAP} "
        f"--batch-size {BATCH} --fps {FPS} --expect-n {N_TEST} "
        f"--out {out} >> {OUT}/j9.log 2>&1", f"eval {tag} on {N_TEST} test clips")
    return out


# ---------------------------------------------------------- phase 1: GPU work, strictly serial
# Training and eval must not overlap: two CUDA processes on one device is how this project once had
# five trainers fighting over a T4.
base = evaluate("unadapted", lambda: CKPT)

ours = evaluate("adapt_ours", lambda: train("adapt_ours", POSES_DEV_OURS, MAN_DEV, N_DEV))

# The control arm. fetch_authors_dev() is called INSIDE the lambda, not before it, so a recycled
# runtime that already has this arm's eval does not re-download 680 MB to score nothing.
def train_control():
    fetch_authors_dev()
    return train("adapt_theirs", POSES_DEV_THEIRS)


theirs = evaluate("adapt_theirs", train_control)


# ------------------------------------------------------- phase 2: the bootstraps, in parallel
# Deliberately after phase 1. bootstrap_ci is pure-Python ROUGE-L over 931 clips x 2000 resamples,
# CPU-bound and minutes long, so interleaving leaves the GPU idle waiting on it.
PAIRS = [
    ("ours_vs_unadapt", base, ours, "adaptation on OUR poses vs un-adapted"),
    ("theirs_vs_unadapt", base, theirs, "the GENERIC gain: adaptation on the authors' poses"),
    ("ours_vs_theirs", theirs, ours, "*** THE CONTRAST OF RECORD: ours - theirs ***"),
]
todo = [(t, a, b, w, f"{OUT}/ci_{t}.json") for t, a, b, w in PAIRS
        if not os.path.exists(f"{OUT}/ci_{t}.json")]
if todo:
    import concurrent.futures as cf
    workers = max(1, min(len(todo), os.cpu_count() or 2))
    print(f"\n[j9] {len(todo)} bootstrap(s) over {workers} worker(s)", flush=True)

    def boot(job):
        tag, a, b, _what, ci = job
        # Each writes its own log: concurrent appends to one file interleave into nonsense.
        return subprocess.run(f"{P} -u -m unisign.bootstrap_ci {a} {b} -n 2000 --out {ci} "
                              f"> {OUT}/ci_{tag}.log 2>&1", shell=True).returncode, tag

    with cf.ThreadPoolExecutor(workers) as pool:
        for rc, tag in pool.map(boot, todo):
            if rc != 0:
                sys.exit(f"[j9] FAILED: bootstrap {tag} (see {OUT}/ci_{tag}.log)")
            print(f"[j9] bootstrap {tag} done", flush=True)

# ------------------------------------------------------------------------------- the result
GAP = 1.4077       # RESULTS.md 2.5k / L19, BLEU-4, n=931
HALFWIDTH = 0.778  # same artifact; L21's bar is derived from it
BAR = HALFWIDTH / GAP

print("\n" + "=" * 78)
print("J9 step 2 -- adaptation to our own keypoints, with the control arm")
print("=" * 78)
print(f"All arms: pruned_traindev lineage (keep set 26,025), fps {FPS}, lr 1e-5, ls 0.0, 1 epoch,")
print(f"seed {SEED}; scored on {N_TEST} OUR test clips, beam {BEAMS}, batch {BATCH}, cap {CAP}.")
print("Absolute BLEU-4 is NOT comparable with the board's 21.73 -- different vocabulary. The")
print("deltas are, because the lineage offset is shared by every arm and cancels.\n")

b = json.load(open(base))
print(f"  un-adapted          BLEU-4 {bleu4_of(b):6.2f}   ROUGE-L {b['rouge_l']:6.2f}")
for tag, ev in (("adapted on OURS", ours), ("adapted on THEIRS", theirs)):
    e = json.load(open(ev))
    print(f"  {tag:<18}  BLEU-4 {bleu4_of(e):6.2f}   ROUGE-L {e['rouge_l']:6.2f}")

print(f"\n{'comparison':<22} {'BLEU-4':>8} {'95% CI':>20} {'ROUGE-L':>9} {'95% CI':>20}")
summary = {"lineage": "pruned_traindev keep 26025", "fps": FPS, "n_test": N_TEST,
           "seed": SEED, "bar_frac_of_gap": BAR, "arms": {}, "pairs": {}}
for tag, ev in (("unadapted", base), ("adapt_ours", ours), ("adapt_theirs", theirs)):
    e = json.load(open(ev))
    summary["arms"][tag] = {"bleu4": bleu4_of(e), "rouge_l": e["rouge_l"]}
for tag, _a, _b, what in PAIRS:
    c = json.load(open(f"{OUT}/ci_{tag}.json"))
    bl, rg = c["bleu4"], c["rouge_l"]
    print(f"{tag:<22} {bl['delta']:>+8.2f} [{bl['ci'][0]:>+7.2f},{bl['ci'][1]:>+7.2f}] "
          f"{rg['delta']:>+9.2f} [{rg['ci'][0]:>+7.2f},{rg['ci'][1]:>+7.2f}]")
    summary["pairs"][tag] = {"what": what,
                             "bleu4": {"delta": bl["delta"], "ci": bl["ci"],
                                       "established": not bl["straddles_zero"]},
                             "rouge_l": {"delta": rg["delta"], "ci": rg["ci"],
                                         "established": not rg["straddles_zero"]}}

rec = json.load(open(f"{OUT}/ci_ours_vs_theirs.json"))
d, ci = rec["bleu4"]["delta"], rec["bleu4"]["ci"]
est = not rec["bleu4"]["straddles_zero"]
print("\n--- the contrast of record: adapted-on-ours minus adapted-on-theirs ---")
print(f"  BLEU-4 {d:+.4f}  95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]  -> "
      f"{'ESTABLISHED' if est else 'NOT established'}")
print(f"  as a share of the -{GAP:.4f} pose deficit: {100 * d / GAP:+.1f}%")
print(f"  the bar agreed before this ran (L21): {100 * BAR:.1f}% of the gap, i.e. +{HALFWIDTH:.3f}")
if est:
    print("  -> clears the bar. Record it, and note the bar was set BEFORE the run.")
else:
    print("  -> does NOT clear the bar. This was the PREDICTED outcome: L17 sizes the expected")
    print("     recovery at ~39 % against a 55.3 % bar, and L21 recorded that before the run.")
    print("     A null here does NOT license 'adaptation does not work' -- the test set is")
    print("     exhausted at 931 clips (L20), so 0.71 detectability is a CAP, not a waypoint.")

summary["contrast_of_record"] = {"bleu4_delta": d, "bleu4_ci": ci, "established": est,
                                "share_of_gap": d / GAP}
json.dump(summary, open(f"{OUT}/j9_step2_summary.json", "w"), indent=1)
print(f"\n[j9] wrote {OUT}/j9_step2_summary.json")
print("[j9] Drive holds the evals, the CIs and this summary. The ~570 MB checkpoints are on local")
print(f"[j9] disk at {FULL} and are NOT backed up -- re-running retrains only what is missing.")
