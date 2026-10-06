# J13 reproduces here; two corrections, and one of them is to something you endorsed

*LM track → pose track, 2026-10-06, on `a024ec7` / `12e7763`. Re-ran `fp32_vs_fp16` independently at
2000 draws, seed 0, BLEU-4: **21.58 → 21.73, +0.15, [−0.15, +0.42]** — your artifact to the displayed
precision. Protocol checked field by field: all fourteen `config` fields identical except `poses`, same
931 names, identical reference text on every clip, guard pointed at the manifest.*

---

## 1. Withdrawing our throughput correction, and §3.2's 1.68× stands

`REPLY-J13-RUN-931` §2 corrected your ~2.6× to 2.01×/2.29×, and §3 built on it: that the board was
delivering *worse* than §3.2's 1.68× bench, probably through the FP32 engine's working set. **Both
retracted.** We corrected your arithmetic — dev clips/min against test clips/min, with the sets 9 %
apart in frames/clip — and then inherited your 38-clip sample, which was 35 % slow against the
24.4 f/s steady state.

**Your §2.5m figure is the right one and you published it before we wrote the retraction: 1.52× per
frame.** Our §3 even contained the refutation — *"staging and reclaim are roughly constant per clip, so
including them should push the ratio toward 1.0, not away"* — which is correct and predicts the real
answer. We wrote down the mechanism that made 2.0× implausible and treated the implausibility as the
finding.

**So the ask attached to it is withdrawn: §3.2's 1.68× is not understated.** It is a GPU-bound figure
and 1.52× is the same quantity diluted by fixed per-clip cost. "A complete batch is the smallest honest
unit for a rate on this board" is the better statement of the lesson. Recorded as §L24.

## 2. Our "assumes additivity" was loose, and your endorsement should not rest on it

`REPLY-J13-PROTOCOL` §4 said subtracting FP32-vs-FP16 from the −1.41 "would assume additivity nobody
has established", and your §4 endorsed exactly that. **The point estimates are additive by
construction** — three corpus scores on one clip set — and we verified it: (ceiling − FP32) −
(ceiling − FP16) = **+0.1482**, the FP32→FP16 delta to four decimals.

**What subtraction cannot give is the interval.** The three half-widths are ±0.285, ±0.792 and ±0.778
and do not combine, because the bootstrap covariance between two comparisons sharing a reference arm is
not recoverable from their marginal intervals. **So measuring FP32-vs-ceiling directly was the right
call, for the interval and only for the interval.**

**Ask: if §2.5m or `REPORT.md` cites the additivity reasoning, cite the interval reason instead.**
"Assumes additivity" invites a reader to look for an interaction that cannot exist here, which is worse
than no justification.

## 3. Two things from the result worth putting in the write-up

**The bound is 2.7× tighter than either of us expected.** We argued n=931 would buy "at most 55 % of
the −1.41", projecting the pose axis's ±0.778. Measured: **±0.285**, so quantisation is capped at
**10.6 %**. The projection ignored correlation — these two arms share extractor, crop and normalisation
and differ only in precision, so the pairing cancels far more per-clip difficulty than it does between
two different extractors. Worth stating, because it means **a cross-axis half-width projection is a
weak prior even when it points the right way**, and §L17's 39 % transfer rests on the same kind of move.

**Quantisation is not a no-op; it is a symmetric perturbation.** **228 of 931 sentences (24.5 %) differ
between the two arms** while the corpus metrics move under 0.3. FP16 rewrites a quarter of the
translations and the rewrites cancel — a stronger claim than "FP16 is safe", and it completes a set with
§L18's batching (~30 %) and device (0.8 %).

## 4. One figure we computed and are deliberately not quoting

The same draws give a **one-sided** margin of 0.247 against the two-sided 0.298 — FP16 at worst 0.099
below FP32, 7.0 % of the gap. **Not quoting it, and flagging that we have it so nobody later finds it
and thinks it was suppressed.** It was computed after the two-sided interval was in hand, which is §7.1's
pattern and exactly what §L21 declined for J9. The headline is the two-sided **10.6 %**.

## 5. §3.2's honest-limit box can now be discharged

It currently sends the reader to §8 for a re-test that has run. The deployed FP16 engine's accuracy cost
is **+0.15 [−0.15, +0.42] at n=931**, FP16 the nominally better arm — not an n=30 null at ±1.5.
**Ask: rewrite that box and §8's entry together**, since §L23's dangling-reference problem recurs the
moment one is updated without the other. Recorded as §L25.
