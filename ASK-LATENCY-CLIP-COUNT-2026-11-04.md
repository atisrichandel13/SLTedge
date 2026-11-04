# Ask: board latency and energy need ~30 clips, not 3 repeats of one

*2026-11-04, LM track → pose track.*

## The defect

Every board end-to-end result in the repo is `n_sentences: 3`, and those three are **repeats of one
clip**, not three sentences:

```
results/e2e_2x2/e2e_{16,24,src}_beam{1,4}.json      n_sentences = 3
results/e2e_pruned/e2e_{24,src}_beam{1,4}.json      n_sentences = 3
```

Table I of the Milestone 1 proposal reports **9052.8 ± 335.6 ms** and **RTF 1.064** from that design.
Three consequences, in descending severity:

**1. The P99 cannot be computed.** The proposal's Metrics section promises latency "as P50 and P99 in
ms". Three measurements cannot estimate a 99th percentile. The metric we committed to and the protocol
we used are inconsistent with each other.

**2. The ±335.6 ms is the wrong variance.** Three repeats of a fixed input measure run-to-run jitter.
They say nothing about variance *across sentences*, which is the variance that governs the result,
because both stages scale with per-clip quantities:

* the decoder dominates — **1062–1538 ms against 39–50 ms for the encoder** (§2.9C) — and its cost
  scales with the number of *output* tokens, which differs per sentence
* the pose stage charges per frame, and frame count differs per clip

**3. The decision rule is a claim about the system.** RTF is latency divided by clip duration, and
both terms vary per clip. On one clip we can only claim RTF ≤ 1.00x *for that clip*. The distribution
across clips may straddle 1.00 while the single measured point sits below it.

## The ask

Time **~30 clips sampled across the OpenASL duration range**, 3 repeats each, at the deployable
config. Not cherry-picked: the current 255-frame / 8.51 s clip is one point on a wide distribution.

**Re-measure the baseline on the same 30 clips.** §IV of the proposal compares everything "in the
same runtime, on the same board and clip", so a 30-clip compressed measurement against a 1-clip
baseline is not a comparison.

Then report **P50 and P95**, which 90 samples support. P99 needs ~100 clips (300 samples, ~45 min) —
either widen to that or we change the proposal's committed metric to P95.

## Why the existing cost argument does not block this

§II justifies 3 instead of the protocol's 1000 because "1000 would hold a shared board for over two
hours". That is correct — 1000 × ~9 s ≈ 2.5 h. It does not extend to 30:

| design | inferences | board time |
|---|---:|---:|
| protocol default | 1000 | ~2.5 h |
| **30 clips × 3 repeats** | **90** | **~13.5 min** |
| current | 3 | ~30 s |

Against the 6–8 board-hours already budgeted for the remaining measurements, 13.5 minutes per
configuration is not the constraint. The reasoning that ruled out 1000 was never re-examined for 30.

## Blocking question

Does the 30-clip set need a fresh pose extraction pass, or do pkls already exist for enough test
clips that `jetson/full_split_pass.sh` covers it? That changes this from ~15 minutes to a longer job,
and it decides whether this lands before or after Milestone 2.
