# Block 6 runbook — the adaptation 2×2 at 24 fps

Fills the gap in `DECISIONS-2026-10-22.md` §3.9: adaptation was only ever measured at **16 fps**,
while **24 fps is the rate we ship** (§3.8 — beam 4 @ 24 fps, −17 % energy for −0.07 BLEU-4).

**The 2×2.** "Adapted" means *adapted at that rate*, so the adapted/24 fps cell needs its own
training run. Both source-rate cells are retrained here so all four cells come from one environment.

|  | source rate | 24 fps |
|---|---|---|
| **un-adapted** | `unadapt_src_cap100` — pruned ckpt | `unadapt_fps24_cap100` — pruned ckpt, `--fps 24` |
| **adapted** | `adapt_src_cap100` — train `ctrl_src_s42` | `adapt_fps24_s42_cap100` — train `fps24_s42`, `--fps 24` |

Two questions come out of it:

* **Within-rate:** does adaptation help at 24 fps? (At 16 fps: **+1.044 ROUGE-L [+0.247, +1.802]**, established.)
* **Difference-in-differences:** does it help *more* at 24 fps than at source rate? (At 16 fps: **+0.32 BLEU-4 [−0.55, +1.18]**, P(>0) = 0.772 — **not** established.)

## Read this before reading the result

`results/fps_coverage_dev.json`, measured on all 967 dev pose files:

| | 16 fps | **24 fps** |
|---|---|---|
| frames reaching the model | **−32.8 %** | **−10.0 %** |
| median frames per clip | 158 → 86 | 158 → **129** |
| clips whose frame count is unchanged (`max_length 256` absorbs the thinning) | 8.1 % | **25.4 %** |
| clips already at or below the target rate, passed through untouched | 0.0 % | 1.9 % |

**24 fps is roughly a third of the treatment 16 fps is.** The 16 fps DiD was already null at n=967.
A null DiD at 24 fps is therefore the expected outcome and is evidence about a small treatment, not
about adaptation in general. The within-rate delta is the question worth running this for.

## Cost

Measured from `results/colab_runs/*__train.log`: **753 s** for one 20,000-clip epoch at 16 fps,
**756 s** at source rate (`max_length 256` makes the rate barely matter). Eval is ~135 s per cell.

**These are T4 numbers** — those logs are the L15.1/L15.2 recipe sweep, which did run on a T4. Block
4 and this run are on an A100, so treat the table below as an upper bound.

| step | time |
|---|---|
| `colab_setup.py` (32 GB archive, 98,419 pose pkls) | ~20 min |
| 2 training runs | ~26 min |
| 5 evals | ~12 min |
| `adapt_ci.py` at n=1000 | ~15 min |
| **total** | **~1 h 15 m** |

Needs ~80 GB of Colab disk (32 GB of zip parts plus the extracted poses) and ~150 MB of Drive.

---

## Cells — fresh runtime, nothing assumed to exist

### 1. GPU and disk

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv && df -h /content | tail -1
```

### 2. Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/SLTedge/block6
```

### 3. Clone the repo

Public clone first; if the repo is private it will fail and the second cell prompts for a token.
**Type the token into the Colab prompt, not into chat.** The remote is rewritten without it
afterwards so it is never stored in the clone.

```python
!git clone --depth 1 https://github.com/atisrichandel13/SLTedge.git /content/SLTedge
```

```python
# only if the clone above failed
import getpass, subprocess
tok = getpass.getpass('GitHub token: ')
r = subprocess.run(f'git clone --depth 1 https://{tok}@github.com/atisrichandel13/SLTedge.git /content/SLTedge',
                   shell=True, capture_output=True, text=True)
print((r.stdout + r.stderr).replace(tok, '***'))
!cd /content/SLTedge && git remote set-url origin https://github.com/atisrichandel13/SLTedge.git
```

### 4. Build the environment (~20 min)

```python
%cd /content/SLTedge
!python colab_setup.py
```

Ends with `[setup] ready:` and three paths. Re-runnable: every step skips when its output is
already present and complete, so a disconnect costs only the step in flight.

### 5. Train and evaluate the 2×2 (~40 min)

`nohup` so a browser disconnect does not kill it; re-run the cell to resume.

```python
%cd /content/SLTedge
!nohup python -u colab_block6.py > /content/drive/MyDrive/SLTedge/block6/driver.log 2>&1 &
```

```python
!tail -40 /content/drive/MyDrive/SLTedge/block6/driver.log
```

Watch for the **VALIDITY GATE** block near the end. The un-adapted source cell involves no training,
so it should reproduce block 4 bit-exactly (L17 measured 0.00e+00 across two runtimes). If it does
not, this run's 2×2 is still internally consistent but is not comparable to block 4's — say so in
whatever the result gets written into.

### 6. Confidence intervals and the DiD (~15 min)

```python
%cd /content/SLTedge
!python -m unisign.adapt_ci \
    --evals /content/drive/MyDrive/SLTedge/block6/evals \
    -n 1000 --rate fps24 \
    --out /content/drive/MyDrive/SLTedge/block6/adapt_ci_dev_fps24.json
```

### 7. Check what survived

```python
!ls -la /content/drive/MyDrive/SLTedge/block6/evals/
!python -c "import json;d=json.load(open('/content/drive/MyDrive/SLTedge/block6/adapt_ci_dev_fps24.json'));print(json.dumps(d['did'],indent=1))"
```

Everything needed afterwards is on Drive: the five eval JSONs, `adapt_ci_dev_fps24.json`,
`gate_unadapt_src.json`, and each run's `last.pt` + `log.jsonl`. The 570 MB `adapted_full.pth`
files stay on `/content` on purpose — re-running cell 5 against a complete `last.pt` rebuilds them
in seconds without retraining.

## Options

```bash
python colab_block6.py --seeds 42,43,44   # + seed band (2 more trainings, ~30 min)
python colab_block6.py --skip-cap64       # drop the max_new_tokens 64 vs 100 check
python colab_block6.py --fps 16           # reproduce block 4's rate in this environment
```

## TF32 on Ampere — left at the PyTorch default on purpose

Block 4 ran on a **Colab A100** (`LM-TRACK-REPORT-2026-09-30.md` §8.4, describing exactly block 4's
2×2). On Ampere, PyTorch leaves `cudnn.allow_tf32` **True** and `cuda.matmul.allow_tf32` **False**,
so the ST-GCN convolutions already ran at a 10-bit mantissa there. Forcing true FP32 now would make
this 2×2 *less* comparable to block 4, not more, so `colab_block6.py` changes nothing by default.

`--no-tf32` sets `NVIDIA_TF32_OVERRIDE=0` and exists to settle it empirically if the validity gate
comes back not bit-exact.

> **Repo defect worth fixing.** `results/RESULTS.md` §L15's header reads *"All runs: Colab Tesla T4,
> FP32 (no AMP — T4 is Turing, no bf16…)"*, and `REPLY-POWER-CAVEAT-STANDS-2026-10-05.md` reads
> "All runs" as covering Block 4. It does not: that header covers the L15.1/L15.2 recipe sweep whose
> logs are in `results/colab_runs/` — none of block 4's four run names (`ctrl_src_s42`,
> `fps16_s42/43/44`) appear there, and every one of those logs is at lr 1e-4, the rejected recipe.
> §8.4 and `SYNC-REPLY-LM-2026-09-30.md:239` both say A100 for the runs that shipped.
