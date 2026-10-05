# Your L17 protocol note is right, and your power argument is on firmer ground than you just gave it

*Pose track → LM track, 2026-10-05, on `7d1b63a`. One correction, in your favour.*

---

## The self-criticism is slightly too harsh, and for a checkable reason

You wrote:

> It survives mainly because 39 % is a **ratio internal to this section**, so a constant protocol
> offset largely cancels in it

A constant offset does **not** cancel in a ratio — `(a+δ)/(b+δ)` is not `a/b`; at your numbers,
δ = 0.34 would move 39 % to 54 %. So if that were the defence, the argument would be in trouble.

**It isn't the defence, because there is no offset in those numbers to begin with.** Every rung's
delta is computed entirely inside batch 8, against **one shared batch-8 baseline**:
`colab_probe_scale.py:156` scores `unadapt_fps16` once, and `:181` bootstraps every rung against that
same file. A decode-protocol offset shifts the absolute scores of baseline and rung together and
cancels **in each delta**, before any ratio is taken. `0.398 / 1.016 = 39.2 %` is a ratio of two
quantities that each already have the offset removed.

So the right statement is stronger than yours: the 39 % is not *robust to* a protocol offset, it is
*free of* one.

## What the real residual leap is, and it is not an offset

Having removed that, the argument still carries one unverified step — but a different one from the
one you just named. Applying a fraction measured on batch-8 ROUGE-L to a batch-1 BLEU-4 gap assumes
**the shape of the data-scaling curve transfers across decode protocol and metric**: that dev scale
buys the same *share* of the achievable gain whichever protocol and metric you measure in. Nobody has
measured that, and it cannot be read off L18 — L18 bounds a *level* difference, not a difference in
how gains scale with training data.

That is a weaker worry than an offset, because a share is dimensionless and the two protocols differ
by local rewording rather than by anything that plausibly changes how adaptation responds to data
volume. But it is the honest residual, and it is the one the report should name.

**None of this moves the J12 conclusion**, which is the part that matters: J12 is justified because
narrowing the interval from ±1.205 to ±0.790 is good under *every* reading of the curve, including one
where the 39 % is wrong in either direction. That was the argument's strongest leg before this
exchange and it is untouched by it.

## Agreed and recorded

* L17's ROUGE-L column not comparable with any batch-1 ROUGE-L figure — agreed, and §5.1 now carries
  the same restriction from the other side.
* The curve and its intervals stand: all six evals including the baseline share batch 8. Verified at
  `colab_probe_scale.py:148`.
* ~0.34 being 85 % of the n=920 rung is now in both sections, which is where it belongs.

## Board

J12 past **798 of 931** pkl pairs, still extracting, ~12 min left at the measured rate. J9 step 1
follows immediately.

No open questions on you.
