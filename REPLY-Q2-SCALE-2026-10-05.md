# Q2: the two "20,000" are different pipelines, and the probe needs neither of them

*LM track → pose track, 2026-10-05. Answers the feasibility objection in `OPEN-QUESTIONS-POSE` Q2.*

---

## You are right that 20,000 **of our own extractions** is infeasible. That is not what L15 did.

L15's 20,000 were the **authors' released pose pkls**, pulled by HTTP range out of the HuggingFace
archive with `data/openasl_pose_fetch.py`. No video was fetched, no frames were written, and the board
was never involved. Measured from the 300 train pkls on the LM-track Mac (0.80 MiB each):

| | what it needs | size | board time |
|---|---|---:|---|
| **L15's 20,000** (authors' poses) | HTTP range reads of an existing archive | **~15.5 GiB** of pkls | **none** |
| **J9 at 20,000** (our extractor) | 20,000 YouTube videos → frames → RTMW FP16 TRT | ~140 GB + the pkls | days |

So your 40 hours and ~140 GB are the right numbers for the thing you costed, and the comparison that
produced "~20× less data" was never a comparison of like pipelines. **J9 at L15's scale is out of
reach, and we are not asking for it.**

## The probe sidesteps both

The question Q2 actually needs answered is *"does ~920 training clips do anything at all with this
harness?"* — and that is answerable on the **frame-rate** axis, where we already know the answer at
20,000, using the cheap pipeline:

* **Authors' poses, not ours.** `data/openasl_pose_fetch.py --split train --limit 1000` is a
  **~0.78 GiB** range-read. No video, no frames, no board.
* **Everything else already fixed.** Same recipe (`ls 0.0, lr 1e-5, warmup 0.1`), same eval set (967
  dev), same un-adapted baseline (23.13 BLEU-4 / 42.93 ROUGE-L), same bootstrap. The only variable is
  the training-set size.
* **Known reference point.** At 20,000 this setup recovers **+1.04 ROUGE-L [+0.25, +1.80]** of the
  −1.34 the 16 fps cut costs.

If ~1,000 clips recovers a decent share of that, J9 at ~920 is worth your board extraction. If it
recovers nothing, J9 at this scale is predictably null and the extraction buys a result the agreed §3
framing already says licenses nothing. Either way the project gains a data-scale number it currently
does not have, and the report needs one regardless of what J9 does.

**It costs Colab time on the LM track and zero board time on yours.** Atisri decides whether to spend
it; this document is the costing, not a commitment.

## What this does not rescue

The probe tells us about **scale**, on an axis where adaptation is known to work. It does not tell us
that pose-source adaptation behaves like frame-rate adaptation at the same scale — those are different
shifts, as your own L15-vs-J9 note sets out. A positive probe makes J9 worth running; it does not
predict J9's result.

## One correction while we were in these files

`TRAINING-STORY` and `RESULTS.md` §L15.2 both said the isolating control (ls 0.2 at `lr 1e-5`) *"has
never been run"* / died *"after 10 log lines"*. The log is tracked —
`results/colab_runs/ctrl_ls02_lr1e5__train.log` — and reaches **step 400, 3,200 of 20,000 clips**,
about 16 % of the epoch. Partial, not absent.

The caveat stands, because it never completed and has no eval. But it leaves evidence we were
discarding: across those 400 steps the loss is flat at **3.3865–3.3993** — the smoothing floor,
observed at the *working* learning rate. Weak, being a partial curve with no eval, but it points the
same way as the diagnosis. Both files corrected.

If you are citing L15's recipe diagnosis anywhere in `REPORT.md`, that is the line to take the wording
from now.
