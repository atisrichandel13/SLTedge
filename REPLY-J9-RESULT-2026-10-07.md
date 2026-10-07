# J9 step 2 is a null — what it changes for your track, and two asks still open

*LM track → pose track, 2026-10-07. §L26, artifacts at `results/j9_*.json`.*

---

## 1. The pose deficit is not recoverable by adaptation — stop treating it as a pending lever

Contrast of record, adapted-on-ours minus adapted-on-theirs, all three arms in one lineage on 931
clips: **−0.1263 BLEU-4, 95 % CI [−0.4973, +0.2488]. Not established.**

**The part that bears on you is not the null, it is the upper bound.** §L17's predicted recovery of
39 % of the −1.4077 deficit is **+0.5490**, and §L21's bar is **+0.7785**. Both sit **outside** the
interval. At 95 % this run excludes any adaptation benefit larger than **+17.67 %** of the gap.

So wherever your track's writing treats the −1.41 deficit as partly an adaptation shortfall pending a
run, that framing is now wrong. **The deficit stays attributed to architecture, crop and
normalisation** — the three things on your side of the split — with quantisation already excluded to
within 11 % by §L25 and adaptation now excluded to within ~18 %.

**§L21's bar was 2.09× too conservative**, which is worth knowing before you set a bar the same way
again. It projected 55.3 % from the pose-axis half-width of ±0.778; this contrast's own half-width is
**0.3730**, so real detectability was **26.50 %**. Same mechanism as §L25: arms sharing an extractor
and a test set cancel far more under paired resampling than a cross-axis projection assumes.

## 2. A measurement that touches your protocol rule

The un-adapted arm scored **21.728700** here against the board's **21.730953** — **0.002253** BLEU-4
apart, across *both* the 53-token vocabulary difference (26,025 vs 26,078) *and* different hardware
(A100 vs Orin Nano, different torch). ROUGE-L: 41.8178 vs 41.8239. §L18's device axis **alone** was
0.0514, so the combined offset is **23× smaller than that one axis**.

**This does not retire your one-environment rule and should not be read as doing so.** One pair of
numbers on one arm is not a characterisation of the offset, and the rule cost us nothing. It is
evidence that the 53 test-only tokens are essentially never generated — no more than that.

## 3. Both asks from this morning are still open

1. **The NaN-score clip.** `rlUUw27_6kM-00:17:09.233-00:17:15.433`, 7 whole frames of all-NaN
   confidences beside finite coordinates. Re-extract, or state that the NaN is expected and how it
   should be read. We have not imputed it and will not.
2. **The FP32 poses have still never been scanned by either side.** Same estimator as the archive that
   produced the bad clip. §L23/§8's FP16-vs-FP32 comparison rests on them, and `data/verify_handoff.sh`
   does not look inside pkls — a non-finite value passes every check we currently run on that archive.

Corroboration for ask 2 you may not have: the control arm trained on all **967** authors' dev clips
with the new non-finite guard never firing (`results/logs/j9_train_adapt_theirs.log`). Two instruments
— a direct scan and a training pass — agree the authors' poses are clean. Neither has been pointed at
FP32.
