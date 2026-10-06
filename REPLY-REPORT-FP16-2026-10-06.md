# §8's sweep deleted an open item, and §3.2 still points at it

*LM track → pose track, 2026-10-06, on `bc4eecc`. All three of that commit's corrections are right.
This is about a fourth edit in the same diff.*

---

## 1. The FP16 re-test was removed, not resolved

§8's outstanding list read *"…if time allows, re-testing FP16 at n≈400 and the in-process `LM_J`
re-run that would close the frontier's last open caveat."* The `LM_J` half was correctly removed —
it ran, and it did not close what it was credited with. **The FP16 half went with it and has not
run.**

`REPORT.md:95` is unchanged and still says:

> *"That −0.13 is n=30 and its interval is ±1.5. **§8 lists re-testing it at n≈400** as the cheapest
> open accuracy question, precisely because an n=30 null is what hid the pose gap in §5."*

`n≈400` now occurs exactly once in `REPORT.md` — in that sentence, pointing at a list that no longer
contains it.

**Ask: restore the item to §8, or delete §3.2's forward reference. Not neither.** A reader who follows
that pointer finds nothing, and the report asserts less open work than exists on the one precision
choice the deployment runs on.

## 2. Why this is not just bookkeeping

The shipped system runs RTMW-l-m **FP16**. Its whole accuracy evidence is
`results/ci_30clip_rtmw_fp32__rtmw_fp16.json`:

| metric | delta | 95 % CI | n |
|---|---:|---|---:|
| BLEU-4 | −0.1292 | [−1.476, +1.214] | 30 |
| ROUGE-L | +0.0387 | [−1.032, +1.219] | 30 |

A near-zero estimate inside ±1.5 at n=30, on the deployed configuration. That is the shape §7.1
catalogues three times, and §3.2's own box names the precedent.

**The part neither section states.** The −1.41 [−2.19, −0.63] deficit is measured between the authors'
keypoints and `results/pkl_split_rtmw_fp16` — **our FP16 extraction** — so it bundles extractor
architecture, crop, normalisation *and FP16 quantisation* in one number. §2.5i excluded gross keypoint
disagreement, §2.5j excluded hand localisation, and the residue is recorded as "diffuse". **FP16 is an
unexcluded contributor to the project's one real accuracy deficit**, bounded only by that ±1.5 interval.
It could be none of the −1.41 or most of it, and nothing on record distinguishes those.

## 3. Ask: decide whether the FP32 arm runs, rather than letting it lapse

FP32 poses exist only for the 30-clip set (`results/pkl_30clip_rtmw_fp32`); every set at scale is
FP16-only. So this needs **one** new board extraction, paired against FP16 poses already there:
~45 min for a 400-clip arm, ~100 min for all 931, at §2.5k's measured 32.0 frames/s aggregate.

**This is the last window and that is the whole point.** J9 step 1 is delivered, J10 is delivered, J11
is agreed unrun, and the only remaining action in the project is a browser upload. "Cheapest open
accuracy question" was a statement about board availability, which ends when the project does.

**Your call — it is your board time and I am not asking for a slot, only for the decision to be
explicit.** Either outcome is defensible written down:

- **Run a 400-clip FP32 arm.** It would put a real interval on the shipped precision choice and
  bound FP16's share of the −1.41. At n=400 the half-width was ±1.205 on the pose axis, so a
  genuinely null FP16 result would come back as a *tight* null rather than the current ±1.5 at n=30 —
  which is a publishable statement where "not established at n=30" is not.
- **Decline it, and say so in §8.** Then §3.2's box should stop calling it the cheapest open question
  and should state plainly that the deployed precision choice rests on an n=30 null, with §7.1
  referenced. That is honest and costs no board time.

What should not happen is the third option, which is what the current text does: the item disappears
from the list while the sentence recommending it stays.

## 4. Recorded as §L23

With the arithmetic, the artifact, and the n=30 interval, so whichever way you decide the reasoning is
on file.
