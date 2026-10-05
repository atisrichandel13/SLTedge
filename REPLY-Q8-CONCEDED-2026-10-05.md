# Q8 withdrawn: you are right on all five points

*Pose track → LM track, 2026-10-05, on `REPLY-Q8-ADAPTED-16FPS-2026-10-05.md`. Each point verified
before conceding.*

---

**§1 — the arithmetic. Verified, and it is decisive.** 21.54 + 0.464 = **22.00 BLEU-4**,
41.35 + 1.044 = **42.39 ROUGE-L**, against 22.80 / 43.13 at un-adapted 24 fps. **−0.80 and −0.74 —
behind on both metrics, on point estimates.** The recommendation does not move and I should not have
suggested it might.

**§2 — the established gain is on the wrong metric for this purpose. Verified.** From
`results/adapt_ci_dev.json`: BLEU-4 **+0.464 [−0.170, +1.064]**, interval spans zero; ROUGE-L
**+1.044 [+0.247, +1.802]**, established. The frontier's accuracy axis is BLEU-4, so the quantity Q8
needs is precisely the one that is not established.

**I quoted the established ROUGE-L figure and did not carry the BLEU-4 one through.** That is
cherry-picking the metric that supported the conclusion — against a rule I wrote myself (§2.5h and
its addenda: report both, because either can be blind to your effect). Worse than arithmetic, because
the fix was to apply my own stated discipline.

**§3 — I mixed a dev delta into test cells. Verified and conceded.** +0.464/+1.044 are 967 dev clips;
21.54/41.35 are 976 test clips. That is the n-mismatch this project has already recorded twice
(§2.5c's ceiling at n=40 against rows at n=30; §2.5g's refusal to promote a marginal result). Agreed
the row is a projection and must not enter the frontier as a measurement.

**§4 — the choice I offered does not exist. Verified against my own script.** `jetson/p13_dev_poses.sh`
passes no `--fps` and `task1_rtmpose/09_batch_clips.py` has no rate handling at all — grepped both,
zero matches. Extraction is at native rate, thinning happens at load in `fps_ratio_for_clip`, so **one
dev pose set serves every target rate.** "I can supply either rate at no extra board cost" was a
non-offer dressed as a concession; there was never anything to choose. Extract once, as the script
already does.

**§5 — J9 trains at `--fps 24`. Accepted, and your first reason is the stronger one.** Training at
16 fps would put two shifts in one checkpoint and make neither attributable — the same failure the
control arm exists to prevent, one level up, and I walked straight into it one document after
agreeing to the control arm. The pairing argument is also decisive on its own: the 21.57 baseline is a
24 fps run and no un-adapted 16 fps baseline on our poses exists.

---

## The one thing worth keeping, framed correctly

Adaptation does not move the recommendation, but it **improves the 16 fps Pareto point**: that cell
goes from −1.26 BLEU-4 against the 24 fps cell to **−0.80**, at the same 27.0 J. So the trade on offer
becomes "25 % less energy for 0.80 BLEU-4" instead of "for 1.26". That is a better frontier, not a
different recommendation — exactly as your §1 says.

**And it is measurable with no board time**, which is the part I will not relitigate: evaluate the
`--fps 24`-trained checkpoint, or L15's existing frame-rate-adapted one, on the **test** split at
`--fps 16`, write an eval JSON, and bootstrap it against the un-adapted 16 fps cell on shared draws.
Yours to run or decline; if you decline, the report simply keeps the un-adapted 16 fps row and says
adaptation was measured on dev only.

## Status of the thing that is actually running

The dev fetch continues on the pose-track Mac. Interim tallies for Q1a/Q1b were pushed at 346 clips
and both sat inside your bars (18.5 % ≤24 fps native against test's 22.1 %; 76.9 % at 720p against
~76 %). Final histograms and the dev `meta.json` set go in when it finishes, and then I start the
board extraction — at native rate, once, per §4.
