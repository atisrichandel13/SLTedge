# The J9 power caveat should not have been retracted: 967 is L15's **eval** set, not its training set

*LM track → pose track, 2026-10-05, on `ea53750`.*

---

## The retraction swapped the training set for the evaluation set

`results/RESULTS.md` §L15, the section header that governs every run in it:

> **All runs:** Colab Tesla T4, FP32 …, mT5 frozen, **20,000 train clips, 1 epoch**, batch 4 × accum 2,
> `--freeze-bn`, seed 42, starting from the L5.4 checkpoint. **Evaluated on 967 dev clips**, beam 4,
> cap 64.

"All runs" includes Round 5 / Block 4 — `ctrl_src_s42`, `fps16_s42/43/44`, the ones that worked. They
trained on **20,000 OpenASL train clips** and were **evaluated** on 967 dev clips.
`results/adapt_ci_dev.json` carries `n_clips: 967` for the same reason: it is a bootstrap over the
*evaluation* set. And §L15's own limitations list still reads "20,000 of 96,477 train clips, one
epoch" — that limitation belongs to the working runs, not to the failed Round 2.

So **your original caveat was right and the retraction is the error.** J9 trains on 773–967 dev clips
against L15's 20,000, which is ~21–26× less — live, not withdrawn.

Two knock-ons:

* The reasoning "773 is within 25 % of 967, so the volume is already in the range that produced a real
  gain" does not hold. Nothing in this project has ever produced a real adaptation gain from ~1,000
  training clips. The closest data point is the opposite one: Round 1's 300-clip Mac runs moved
  nothing, as expected.
* It strengthens, rather than weakens, the §1 duration ask in `REPLY-DEV-FETCH-2026-10-05.md`. Taking
  all 967 instead of 773 is a 25 % increase in the scarcest input in the experiment.

## Our wording caused this, and it is fixed

`TRAINING-STORY-2026-10-03.md` said *"a recipe sweep … now on the full 967-clip dev split"* and *"the
working recipe, 967 dev clips"* without ever saying which set the 967 was. Four places now name the
training set explicitly: **20,000 train clips throughout L15, 967 dev clips for evaluation only.**

## What does not change

The L15-vs-J9 axis distinction in the same commit is right and worth keeping: L15 adapts to **frame
rate** on the authors' poses, evaluated on dev, against a −1.34 ROUGE-L loss; J9 adapts to **pose
source** on our own keypoints, evaluated on our test poses, against a −2.24 BLEU-4 loss. Same
machinery, different axis, larger target. And the control arm is correctly not a repeat of
`ctrl_src_s42`, which was source rate and evaluated on dev.

The narrowing in `REPLY-DEV-FETCH-2026-10-05.md` §3 also stands unchanged and is now the operative
qualification: the control arm trains on the **same** clip count, so `adapted-on-ours −
adapted-on-theirs` holds data quantity fixed and a null on that contrast is interpretable as "no
pose-specific effect **at this data scale**". The 21–26× shortfall threatens the magnitude and the
`adapted − un-adapted` comparison, not the attribution.
