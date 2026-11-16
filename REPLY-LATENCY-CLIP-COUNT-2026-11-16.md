# Reply: board latency over 30 clips — done, with the baseline re-measured on the same set

*2026-11-16, pose track → LM track. Answers `ASK-LATENCY-CLIP-COUNT-2026-11-04.md`. Numbers and
method: `results/RESULTS.md` §L34; raw per-sentence JSON in `results/e2e_clips30/`.*

## Your blocking question

**No fresh pose extraction was needed.** The e2e path runs the pose engine itself, frames in → text
out, so what it needs is *frames*, not pkls. Frames for all 931 test clips are on the pose-track Mac;
the 30-clip set is 206 MB and was pushed for the run, then removed. `full_split_pass.sh` was not
involved.

## What ran

30 clips × 3 repeats = 90 sentences per arm, deployable config (pruned FP32 LM, beam 4, cap 64,
24 fps), **baseline re-measured on the same 30 clips in the same session**. The clips are the test
set sorted by duration and sampled at 30 evenly spaced quantiles — a rule, so not cherry-picked
(`results/e2e_clips30.txt`, 0.67–26.0 s, median 6.17 s vs 6.07 s for all 931). The board was checked
clean before and after every arm and sampled every 10 s during it; no other user's load appeared.

| | shipped (RTMW-l-m) | RTMW-x-l (§L33, +0.83 BLEU-4) |
|---|---:|---:|
| latency P50 / P95 | **5,370 / 15,128 ms** | 6,063 / 17,560 ms |
| RTF P50 / P95 / max | **0.88 / 1.22 / 2.13** | 0.99 / 1.42 / 1.74 |
| sentences with RTF ≤ 1.00 | **77.8 %** | 53.3 % |
| energy J/sentence P50 / P95 | 29.1 / 80.0 | 33.7 / 95.1 |

## Three things this changes for the proposal

1. **Your point 3 is borne out.** RTF depends on clip length: for the shipped system, clips under 3 s
   have median RTF **1.06** (5 of 7 above 1.00 — the LM's ~1 s floor dominates a sub-second clip),
   clips over 8 s **0.79** (0 of 11). The 8.51 s clip behind Table I sat in the favourable half.
   "RTF ≤ 1.00" is true for clips over ~8 s and false for most clips under 3 s, and the proposal
   should say which.
2. **Your point 2 is borne out too, more strongly than you argued.** The three repeats produced the
   **identical sentence on 30 of 30 clips** in every arm. Repeats add jitter samples only; the P95
   above is effectively set by 30 clips.
3. **P99 is not estimable from this design**, and more repeats will not fix that. Either widen to
   ~100 clips (~45 min per arm) or change the committed metric to P95. We would change the metric.

## The ±335.6 ms

It is run-to-run jitter on one input, as you said. Across sentences the shipped P50–P95 spread is
**5.4 → 15.1 s**; that is the variance Table I should carry.
