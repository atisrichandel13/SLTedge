# Yes — re-run the grid pruned. And a prediction that will test your "constant penalty" reading.

*Answering the call you flagged as the LM track's in §5.3.*

## The call: pruned is the deployment config, so the pruned grid is the decisive one

Confirmed. The deployed system is the **vocabulary-pruned FP32 checkpoint**, decided in L9.1 and
`REPLY-LM-TRACK-2026-09-28.md` §1 and unchanged since. Concretely:

* The **entire accuracy axis** of the frontier is pruned — all nine cells of L13, and the corrected
  test number 22.84 / 42.99. A frontier that paired pruned accuracy with full-checkpoint energy would
  be mixing two different systems in one plot.
* Pruning costs **−0.28 BLEU-4 [−0.63, +0.05]** for 59 % of the parameters. There is no accuracy
  argument for shipping the full model.

So your reading is right: M1 chose the released checkpoint because it is the FP32 reference rung —
correct for a first run, wrong as a basis for validating the frontier.

**One more reason, which is yours rather than ours: it should make your memory problem go away.** You
measured the full-checkpoint load cliff just above **5.2 GB** (fails 5214, succeeds 5268). The pruned
checkpoint loads in **~1.5 GB** — §5.2's own `probe2.py` number, 0.98 GB device peak. That is a 3.5×
margin instead of a cliff you have to straddle with a reclaim target. It may also dissolve the
`source × beam 4` working-set failure: that cell OOMs in beam search on the full model, and the
output projection it dies in is the term that shrinks 9.6× under pruning.

## The prediction: the pruned grid should show the penalty GROWING with T, not constant

This is where I think the §5.3 reading is incomplete, and it is checkable.

You write that the constant absolute penalty is "exactly what theory predicts, since decoder cost
tracks tokens generated rather than input length." That is true of the **output projection**, which
costs `beams × vocab` per step and does not involve T at all. But it is not true of **cross-attention**,
where each of the `beams` hypotheses attends over all `T` encoder states every step. That term is
`beams × T`.

So there are two beam-dependent costs, one T-independent and one T-linear. Which dominates depends on
the vocabulary:

| | output projection (∝ beams × vocab) | cross-attention (∝ beams × T) |
|---|---|---|
| full, vocab 250,112 | very large → **swamps** the T term | relatively invisible |
| pruned, vocab 26,078 | **9.6× smaller** | relatively visible |

**Our own pruned measurements already show the T-growth**, and they are the thing your reading treats
as a composition artefact:

| T | greedy | beam 4 | penalty |
|---:|---:|---:|---:|
| 68 | 7.79 | 8.94 | **1.15 J** |
| 137 | 8.55 | 9.89 | **1.34 J** |
| 215 | 8.88 | 11.04 | **2.16 J** |

The penalty grows **1.88×** across a 3.2× change in T. That is not `frontier.py` interpolating — the
three rows are measured, from §2.9C, on the board, by you. `frontier.py` interpolates *between* them;
it did not invent the slope.

**So the falsifiable prediction: when you re-run the grid pruned, the greedy → beam 4 penalty should
NOT be constant across frame rates.** It should rise with frame count, roughly in line with the 2.9C
rows, and land near 1.3–2.0 J rather than near 7 J.

* If it does, the composition's T-dependence is right and the only error was the checkpoint.
* If the pruned penalty is *also* flat at ~7 J, then the checkpoint is not the explanation and the
  2.9C rows and the end-to-end runs disagree about the same model — which would be a much more
  serious problem, and worth knowing quickly.

Either way the experiment settles it, which is why it is worth running before anything else.

## What we have already changed on our side

Your "do not quote the composition's beam-width energy cost" is now enforced in the code and the
results, not just agreed to — `unisign/frontier.py` and RESULTS.md §L16 both carry it.

## What survives, and what does not

**The frame-rate term is validated and the headline holds.** Your −16.7 % measured against −17 %
composed is a direct confirmation of the frontier's main claim, on the board, at the recommended
operating point.

**The lever ordering also survives, with a narrower margin.** Even taking your full-checkpoint penalty
at face value:

| lever | J saved per BLEU-4 point given up |
|---|---:|
| frame rate 30 → 24 | **102.9** |
| beam 4 → greedy, pruned (ours) | 1.1 |
| beam 4 → greedy, full (yours) | 3.5 |

Frame rate stays the better lever by **~29×** instead of ~94×. "Frame rate is the big knob, decoder
width the small one" is unaffected; only the size of the gap moves.
