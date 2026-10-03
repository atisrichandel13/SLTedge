# The training and adaptation work, in plain words

*2026-10-03. Expands §4 of `RESEARCH-STORY-2026-10-03.md`. Every training run we ever launched is
listed, including the ones that failed and the ones that were killed by infrastructure. Numbers from
`results/RESULTS.md` §C8.1, §C8.2, §L15 and `results/adapt_ci_dev.json`.*

---

## 1. What we were trying to do, and why

The frontier says running the camera at a lower frame rate saves a lot of energy. But feeding the
model fewer frames costs accuracy, because the model was **trained on 30 fps** and has never seen
anything slower.

**The question: if we show the model low-frame-rate data during training, can it learn to cope, and
get the accuracy back?**

That is all "adaptation" means here. Not a new model — the same model, shown the kind of input it
will actually meet at deployment.

## 2. What we train, and what we deliberately don't

The model has two parts, and we freeze most of it:

| part | size | trained? |
|---|---|---|
| mT5 (the language model that writes English) | 238 M | **frozen** |
| pose stack + `pose_proj` + `part_para` (reads skeletons) | **5.35 M** | **trained** |

So we train **5.35 M of 243.5 M parameters — about 2%.**

**Why freeze the language model?** Two reasons. The problem is on the *input* side — the skeletons
changed, English didn't. And with only 20,000 training clips, fine-tuning a 238 M-parameter language
model would overfit badly and probably damage its fluency. Freezing it means the only thing that can
change is how we read poses, which is exactly the thing that needs to change.

A practical bonus: a checkpoint is only the trainable parameters, so it's tens of megabytes rather
than 570 MB. That mattered a lot given how often Colab died on us.

---

## 3. Every training run we did, in order

### Round 1 — Mac smoke tests: "does the script work at all?"

Not experiments. The point was to prove the code runs and the checkpoints are valid *before* spending
Colab GPU time.

**Run 1 — `C8.1`, no shift, 300 clips, 2 epochs, Mac CPU.**

| | BLEU-4 (200 held-out clips) |
|---|---|
| before | 14.71 |
| after epoch 0 | 14.08 |
| after epoch 1 | 14.49 |

Wobbles ±0.6 and goes nowhere, which is **exactly what should happen** — 300 clips teach a model
nothing. What this proved: the script trains, saves, and the saved checkpoint reloads and scores
within **0.06 BLEU-4** of the training log's own eval. The plumbing is correct.

**Run 2 — `C8.2`, same but at 16 fps, 300 clips, 2 epochs, Mac CPU.**

| | BLEU-4 |
|---|---|
| full rate baseline | 14.71 |
| 16 fps, before training | 13.79 |
| 16 fps, after epoch 0 | 13.78 |
| 16 fps, after epoch 1 | 13.55 |

Recovers nothing (−0.24, inside the ±0.6 noise). Also expected. This existed to exercise the `--fps`
code path end to end before Colab, not to measure anything.

### Round 2 — the first real attempt, on Colab. **It failed.**

Recipe: the authors' own settings — **label smoothing 0.2, learning rate 1e-4, no warmup.** 20,000
clips, 1 epoch, evaluated on 300 dev clips.

**Run 3 — `fps16_seed42`**, trained and evaluated at 16 fps:

| before | after | change |
|---|---|---|
| 18.27 | 16.57 | **−1.70 BLEU-4** |

Training made the model **substantially worse.**

The obvious conclusion is "adaptation doesn't work for this problem." **That conclusion would have
been wrong, and one more run is what proved it.**

### Round 3 — the control run that saved the project

**Run 4 — `nofps_seed42`**, the identical recipe but trained on **normal-rate** data — where, by
construction, **there is nothing to adapt to.**

| before | after | change |
|---|---|---|
| 17.17 | 16.13 | **−1.04 BLEU-4** |

**The control degraded too.** With no distribution shift at all, training still cost a point.

That single run reframes everything: the −1.70 was never evidence about adaptation. It was evidence
about a **broken training recipe**. If we hadn't run the control, we'd have written down "adaptation
fails here" — a false negative, published.

> **The general lesson:** when a treatment makes things worse, run the treatment with the active
> ingredient removed before you conclude anything about the treatment.

### Round 4 — finding the broken ingredient

**Runs 5 and 6 — a recipe sweep**, all on the *control* condition (no frame-rate shift), now on the
**full 967-clip dev split**. Un-adapted baseline: **23.13 BLEU-4 / 42.93 ROUGE-L**.

| run | recipe | Δ BLEU-4 | Δ ROUGE-L |
|---|---|---|---|
| (round 2) | ls 0.2, lr 1e-4, no warmup | −1.04 *(at n=300)* | not measured |
| **Run 5** | ls 0.0, lr 1e-4, warmup 0.1 | −0.38 | +0.58 |
| **Run 6** | **ls 0.0, lr 1e-5, warmup 0.1** | **+0.18** | **+0.70** |

**Run 6 is the working recipe.** Training now *improves* the model instead of damaging it. This was
the first real evidence that the harness **learns**, as opposed to merely executing.

**What label smoothing was doing.** Label smoothing tells the model "don't be too confident in the
right answer — spread some probability over everything else." At 0.2 over a 26,000-word vocabulary,
that spreading dominates. The reported training loss sat at **~3.35 and barely moved**, which looked
like a model that couldn't learn. With smoothing off, the true cross-entropy is **~0.36**. The 3.35
was mostly a **constant floor imposed by the smoothing**, and we had been reading that constant as a
training curve.

**An honest gap, and we flag it every time.** The clean way to prove label smoothing was the culprit
is to change *only* label smoothing: run ls 0.2 at lr 1e-5. **We queued that run and a Colab
disconnect killed it after 10 log lines. It has never been run.** The working recipe changed
smoothing, learning rate *and* warmup together. So "label smoothing was the problem" is **the best
available reading, not an isolated result.** It is written that way in RESULTS.md.

### Round 5 — the real adaptation runs, with statistics

**Runs 7–10 (Block 4)** — the working recipe, 967 dev clips, with eval JSONs saved so every number
can get a confidence interval, and **three training seeds** so we know the noise floor:

- `ctrl_src_s42` — control, source rate
- `fps16_s42`, `fps16_s43`, `fps16_s44` — 16 fps, three seeds

**The results** (967 dev clips, paired bootstrap, 1000 resamples):

| comparison | BLEU-4 | ROUGE-L | verdict |
|---|---|---|---|
| cost of 16 fps, un-adapted | −0.35 [−0.97, +0.52] | **−1.34 [−2.35, −0.33]** | **real loss** (ROUGE-L) |
| **adaptation at 16 fps** | +0.46 [−0.17, +1.06] | **+1.04 [+0.25, +1.80]** | **real gain** (ROUGE-L) |
| adaptation at source rate | +0.12 [−0.54, +0.78] | +0.60 [−0.10, +1.37] | not established |
| *does adaptation help MORE under shift?* | — | +0.42 [−0.58, +1.53] | **not established** |
| seed-to-seed spread (3 seeds) | 0.05 | 0.15 | the noise floor |

**Reading it in plain terms:**

1. **Running at 16 fps really does cost accuracy** — about 1.3 ROUGE-L.
2. **Adaptation really does recover it** — about 1.0 ROUGE-L back, so roughly **75% of the loss.**
3. The gain is **~7× the seed-to-seed spread** (1.04 against 0.15), so it isn't a lucky seed.
4. **But we cannot claim adaptation specifically fixes frame-rate shift.** That's the last row — the
   interval includes zero. It might just be that *any* fine-tuning on this data helps a bit.

We claim (1), (2) and (3). We explicitly refuse to claim (4).

---

## 4. Everything that went wrong, and the fix

### Scientific problems

**1. The training recipe destroyed the model.** Label smoothing 0.2 at lr 1e-4.
→ **Fixed:** ls 0.0, lr 1e-5, warmup 0.1. Caught by running a no-shift control.
→ **Still imperfect:** the isolating run was never completed, so the attribution to label smoothing
alone is the best reading rather than a proven one.

**2. We evaluated on 300 clips and got a sign error.** At n=300, 16 fps scored **+1.10 higher** than
source rate — the *opposite* of the truth we'd already measured at n=976.
→ **Fixed:** every later run uses the full 967-clip dev split. This became the seed of the project's
most interesting methodological finding: BLEU-4 is too noisy to resolve effects this small.

**3. The first adaptation results had no confidence intervals.** `train_adapt.py`'s built-in eval
logged summary numbers only, not the 967 individual predictions, so nothing could be bootstrapped —
while every other comparison in the project carried a CI.
→ **Fixed:** re-ran `eval_openasl.py` against the saved checkpoints to produce real eval JSONs, then
bootstrapped those.

**4. We overclaimed, and retracted it.** L15.3 originally said *"adaptation helps more under
frame-rate shift than without it"*, based on a difference of +0.22 with no interval. With intervals:
**+0.42 [−0.58, +1.53]** — not established. **Retracted.**

**5. A plausible excuse that turned out to be false.** I suggested the adaptation gain might be
*understated* because evaluation truncated outputs at 64 tokens. Measured it: cap 64 vs cap 100 gives
**exactly 0.00 difference** — bit-identical. **Retracted.**

**6. One seed is not a result.** "+0.40 vs +0.18" is uninterpretable without knowing how much two
identical retrains differ.
→ **Fixed:** three seeds. Spread is 0.05 BLEU-4 / 0.15 ROUGE-L, now `results/l10_seeds.json`.

### Infrastructure problems (Colab ate a lot of our time)

**7. `ModuleNotFoundError: portalocker`, three separate times.** Colab loses pip state on a runtime
restart, so a package installed in one cell is gone in the next.
→ **Fixed:** install *and verify the import in a subprocess* before launching any job. (It then bit a
fourth time on the Mac.)

**8. Google Drive quota exceeded — and then the entire `unisign/` Drive folder was deleted.**
→ **Fixed by removing Drive from the critical path entirely**: everything re-fetches from public
HuggingFace. The lasting rule is the inverse — small, expensive-to-recreate things (checkpoints,
eval JSONs, logs) go to Drive; large, cheaply re-downloadable things (the 32 GB pose archive) do not.

**9. `EOFError: Ran out of input` at training step 2250.** A pose file had been truncated when an
earlier download was killed. Our extraction code checked only whether the file *existed*.
→ **Fixed:** validate every extracted file against its size in the archive (`ZipInfo.file_size`), not
its existence.

**10. Jobs that looked hung but were working, three times.** `nohup` plus output redirection buffers
stdout, so a perfectly healthy job prints nothing for a long time.
→ **Fixed:** `sys.stdout.reconfigure(line_buffering=True)` in every long-running script.

**11. About five copies of the same job running on one GPU.** Re-launching a cell that looked stuck
started another trainer alongside the first; `fps16_s43` trained three times and `fps16_s44` twice.
→ **Fixed:** checked every checkpoint by actually loading it, rather than trusting that files existed.
All four were valid.

**12. A 2.3 GB download.** A cell that zipped "the results" swept in four 570 MB checkpoints.
→ **Fixed:** filter to `.json` / `.jsonl` / `.log`.

**13. Mid-epoch checkpointing.** Colab kills sessions without warning, and a run that only saved at
the end of an epoch lost everything.
→ **Fixed:** `--save-every` writes atomically mid-epoch (temp file, then `os.replace`), records the
batch index, and resumption re-enters a partial epoch at the right place with the dataloader reseeded
per epoch so the ordering is reproducible.

---

## 5. What this adds up to

**Adaptation works, and it is worth about 1.0 ROUGE-L at 16 fps — roughly three quarters of what the
frame-rate cut costs.**

But the most valuable things we produced here are arguably not that number:

- **The control run.** It is the only reason we didn't record a false negative about adaptation.
- **The n=300 sign flip.** A known effect reversed direction at small n, which is what started the
  metric-power investigation that is now the project's most publishable finding.
- **The retractions.** Three claims we made and withdrew on measurement — the difference-in-differences,
  the truncation excuse, and the original label-smoothing-free reading of the first failure.

**What is still not done:** 12 fps and 8 fps adaptation, the face-group drop, more than one epoch, and
more than 20,000 of the 96,477 available training clips. All deliberately deferred.
