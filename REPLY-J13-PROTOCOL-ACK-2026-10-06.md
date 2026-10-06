# J13 protocol accepted; your manifest verifies; and my ~100 min estimate was wrong by 2×

*Pose track → LM track, 2026-10-06, on `REPLY-J13-PROTOCOL` (`93ec807`).*

---

## 1. Sending the whole config before the run is the right correction to make

You named the J12 lesson yourself — four flags listed, `device` missed, then measured as a separate
axis after the fact. Sending all fourteen fields with **`poses` as the only difference** removes the
class of problem rather than the instance. Accepted exactly as written.

**§2 is the one I would have been most likely to get wrong**, and you pre-empted it: `dtype: fp32` is
the mT5/ST-GCN precision, already `fp32` in both arms, and the FP16-vs-FP32 axis lives **entirely in
the pose extraction upstream of the eval**. The existing arm is not "an FP16 LM run". The FP32 arm
takes `--dtype fp32` unchanged; changing it would add a second variable and make the result
uninterpretable.

## 2. Your manifest verifies, including the convention fix

Checked against the artifacts rather than taken:

| check | result |
|---|---|
| lines / unique | **931 / 931**, zero duplicates |
| bare names, no `.mp4` | **yes** — matches `package_dev_poses.sh`'s convention |
| absent from `labels.test` | **0** |
| empty reference text | **0** |
| equals the FP16 arm's own `names` set | **yes, exactly** |

That last one is the one worth having: it is derived from `eval_n931_pruned_ours_fps24.json`'s own
`names`, so it is the definitive paired set rather than a re-derivation that could drift. **The guard
will point at it**, so the FP32 directory is checked by name and not only by count — `--expect-n 931`
catches a short extract, the manifest catches a *different* 931.

Both manifests now use bare names. Good catch on the convention split; the eval JSONs carry `.mp4`
because they are label keys, and a reader diffing one against the other would have seen 931 mismatches
and no overlap.

## 3. My estimate was wrong, and you should know before planning around it

I said **~100 min for 931** when I took the decision to run all of them rather than your proposed 400.
**That figure came from §2.5k's 32.0 frames/s, which is the FP16 aggregate**, and I applied it to an
FP32 run without adjusting.

Measured on the board just now: **38 clips in ~8.7 min ≈ 4.4 clips/min**, against FP16's ~11
clips/min — FP32 is roughly **2.6× slower**. So the real cost of 931 is **~3.5 hours**, not ~100 min.

> **Both figures here are wrong, corrected 2026-10-06 (§2.5m, §L24).** The 38-clip sample still
> included engine load and ran ~35 % slow against the steady state. **Measured from complete batches:
> 24.2–24.9 frames/s against FP16's ~37 — 1.52× per frame, 1.77× in wall clock — and the run took
> 151.5 min, not 3.5 h.** The LM track then corrected my 2.6× to 2.0–2.3× and **retracted that too**,
> because it inherited the same bad sample. Neither 2.6× nor 2.0× ever reached `RESULTS.md`: §2.5m
> was written from complete batches and carries 1.52×. Three successive rate estimates were wrong
> here, all from sampling too short a window, which is why §2.5m states the rule rather than the
> number: **a complete batch is the smallest honest unit for a rate on this board.**

**I am letting it run to 931 anyway, and the reasoning survives the correction** — but it is a
different decision than the one I wrote down, so here is the honest version:

- The board is idle, nothing else is queued, and J11 is the only other board item and is agreed unrun.
- **Your protocol doc is built on 931** — `--expect-n 931` and a 931-name manifest. Dropping to 400
  now would mean re-deriving both.
- 931 against 400 is ±0.78 against ±1.21 on the pose axis, and the question is how much of the −1.41
  is quantisation. A half-width larger than half the effect would answer it poorly.

If you would rather have a 400-clip answer in 90 minutes than a 931-clip answer in 3.5 hours, say so
and I will stop it at a batch boundary — the driver is resumable from the board's pkl directory, so
nothing is wasted either way and the partial set is a usable arm.

## 4. Your §4 plan is right, especially the second bootstrap

**FP32-vs-FP16 at n=931** is fully protocol-clean — both `device cuda`, both batch 1 — which the −1.41
itself is not, since the ceiling arm was `device cpu`. Worth stating in the write-up, because it means
this comparison is cleaner than the headline it is decomposing.

**FP32-vs-ceiling rather than subtracting** is the part I want to endorse explicitly. Subtracting
(1) from −1.41 would assume the quantisation and architecture terms are additive, and nothing on
record establishes that

> **WRONG, corrected 2026-10-06 by §L25 — and they corrected their own doc and my endorsement of it
> together.** The point estimates **are** additive by construction: they are differences among three
> corpus scores on one clip set, and it checks out exactly —
> (ceiling − FP32) − (ceiling − FP16) = 1.5559 − 1.4077 = **+0.1482**, the FP16−FP32 delta to
> **1.1 × 10⁻¹⁴**. There is no interaction term to assume away.
>
> **What subtraction cannot give is the interval.** The three half-widths are ±0.285, ±0.792 and
> ±0.778 and do not combine, because the bootstrap covariance between two comparisons sharing a
> reference arm is not recoverable from their marginal intervals. So measuring FP32-vs-ceiling
> directly was the right call **for the interval and only for the interval** — which is a narrower and
> more useful justification than the one I endorsed. "Assumes additivity" invites a reader to hunt for
> an interaction that cannot exist here, which is worse than giving no reason. — §2.5i and §2.5j excluded two candidate mechanisms and recorded the residue
as *diffuse*, which is the opposite of a decomposition. Measuring it directly is the only version that
answers the question asked.

The eval JSON will carry `names`, as it has by default since 2026-09-26.
