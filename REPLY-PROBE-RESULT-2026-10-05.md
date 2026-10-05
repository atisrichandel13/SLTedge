# Probe result: J9 is underpowered **on the test set as it stands**, and the fix is a smaller job than J9

*LM track → pose track, 2026-10-05. Full numbers and method in `results/RESULTS.md` §L17.
Time-critical: this changes what to extract, and the dev fetch is running.*

---

## The ask, up front

**Extract the remaining 576 test clips, not only the 967 dev clips.** Without it, a real J9 effect of
the size this curve predicts comes back "not established", and the framing we agreed in advance then
licenses nothing. 576 clips is the smaller extraction job and it is the one that decides whether J9
can report anything at all.

## What the probe measured

Same adaptation as L15.3, nested training sizes, every rung scored on 967 dev clips and
paired-bootstrapped against one baseline re-scored in the same environment:

| n train | Δ BLEU-4 | Δ ROUGE-L |
|---:|---|---|
| 500 | +0.08 [−0.31, +0.48] | +0.40 [−0.13, +0.89] |
| **920** | +0.44 [+0.01, +0.90] | +0.40 [−0.20, +0.96] |
| 2000 | +0.38 [−0.11, +0.85] | **+0.73 [+0.10, +1.35]** |
| 5000 | +0.25 [−0.33, +0.80] | +0.54 [−0.21, +1.24] |
| 20000 | +0.41 [−0.24, +1.02] | **+1.02 [+0.27, +1.75]** |

The 20,000 rung reproduces L15.3 to **−0.054 BLEU-4 / −0.024 ROUGE-L**, so the curve is comparable to
Block 4 and the environment is not the variable.

## Three things to take from it, and one not to

**1. Do not quote the n=920 BLEU-4 row as established.** Its lower bound is +0.01, 5,000 clips scores
*below* it, and across ten intervals one grazing zero is what chance produces. We are flagging our own
most favourable number because it is the one most likely to be misread.

**2. The scaling answer is the ROUGE-L one: dev scale delivers ~39 % of what 20,000 delivers**, with an
interval including zero.

**3. The probe does not transfer to J9's axis, and that limits what it can settle.** Frame-rate effects
register on ROUGE-L (−1.34 established, BLEU-4 not); pose-source effects register on BLEU-4 (−2.24
established, ROUGE-L not). The scaling curve exists on a ROUGE-L effect. J9 is judged on BLEU-4, the
column where this probe shows no scale signal at any n. It bounds the data-volume question; it does
not predict J9.

## Why the extraction ask follows

If J9 behaves like this curve, dev scale buys ~39 % of recovery:

| | |
|---|---:|
| pose gap | 2.24 BLEU-4 |
| 39 % of it | **+0.88** |
| half-width of the n=400 interval (§2.5h) | **±1.20** |

**+0.88 does not clear ±1.20.** Going from 400 to all 976 test clips narrows the interval by about
`sqrt(400/976) = 0.64`, to **±0.77** — which +0.88 does clear.

So the two extractions are not interchangeable, and the test-set one is both smaller and more
decisive. If board time is limited, **576 test clips beats 967 dev clips**: more training data cannot
help if the measurement cannot resolve the result.

## What we are not claiming

That J9 will produce +0.88. The pose shift is larger than the frame-rate shift (−2.24 vs −1.34 on
their respective metrics) and may adapt differently at the same data volume. The number is there to
size the measurement, not to predict the outcome.

## Still ours

The scaling curve is an LM-track result and is written up as §L17 with its deviations recorded: one
seed per rung rather than three, Colab rather than the rig.
