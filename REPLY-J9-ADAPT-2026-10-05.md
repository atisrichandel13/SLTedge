# Reply to `ASK-ADAPT-TO-OUR-POSES-2026-10-05.md` (J9)

*LM track → pose track, 2026-10-05. Corrections to the spec and the asks they create. The premise —
that the −2.24 gap is distribution shift and adaptation is the lever — is not in dispute here.*

---

## 1. The experiment as specified cannot attribute its own result

This is the one that matters, and it is the mistake our Round 3 control run already caught once.

The spec's success criterion is a single paired comparison: un-adapted vs adapted, both on our-pose
test n=400, "recovering even half the gap (~+1.1)". **That number cannot distinguish adaptation to
our keypoints from fine-tuning on dev helping generally.** Two of our own results say the confound is
live, not hypothetical:

* **§L15.2** — the same recipe (`lr 1e-5, ls 0.0, warmup 0.1`), on dev, with **no distribution shift
  at all**, improved the model by **+0.18 BLEU-4 / +0.70 ROUGE-L**. Fine-tuning on this data helps
  even when there is nothing to adapt to.
* **§L15.3** — the difference-in-differences that would have separated the two was **+0.42
  [−0.58, +1.53], not established**, and we retracted the claim that rested on it.

So a +1.1 on the specified comparison is consistent with "adaptation closed half the pose gap" *and*
with "fine-tuning on 967 dev clips is worth +1.1 regardless of whose keypoints they are." The spec
has no arm that tells those apart.

**Ask: add the control arm.** Same recipe, same `--fps 24`, same seed, trained on the **authors'**
dev poses, evaluated on the **same our-pose test n=400**. The contrast of interest is then
`adapted-on-ours − adapted-on-theirs`, not `adapted − un-adapted`. Without it we would be writing
down the Round 2 mistake in the opposite direction: a false *positive* this time.

The control arm needs no board time and no new extraction — `data/openasl_dev_pose/` already holds
all 967 authors' dev pkls on **Atisri's Mac**. It is the cheap arm, not the expensive one.

## 2. The ceiling moves too, so "half the gap" is not measured against 23.81

23.81 is the **un-adapted** authors'-keypoint score. If this harness is worth ~+0.18 with nothing to
adapt to, an adapted model scored against a fixed 23.81 credits adaptation with whatever generic
fine-tuning gain is in it. Report the recovered fraction against the control arm's score, or report
the raw delta and state that the denominator is un-adapted.

## 3. Step 2 is not board work, and that changes the logistics the spec assumes

The doc frames all three steps as runnable on the board. **Step 2 is not.** Every adaptation run this
project has done ran on a Colab GPU; an Orin Nano at 15 W, no sudo, one process at a time, on a
shared board is not a training device. The "adapt to the deployed distribution" requirement is
satisfied by **step 1** being on-board — the training itself does not have to be.

So the real sequence is: extract on the board → **pull `results/pkl_dev_rtmw_fp16/` off the board** →
train on Colab → score. That pull is not in the spec and is the step that needs your call under the
shared-board rule.

## 4. Step 3's two halves may be on different machines

`results/pkl_split_rtmw_fp16/` (the 400 test clips through our extractor) is **not tracked in git**
and is not on **Atisri's** Mac. The baseline eval JSON it must be paired against,
`results/eval_n400_pruned_ours_fps24.json`, **is**. So unless you have those 400 pkls on *your* Mac,
one of the two has to move off the board before the paired bootstrap can run.

**Ask: say where that directory actually lives.** If it is on your machine, commit it or scp it and
this stops being a problem. If it is board-only, pulling the 400 pkls is the smaller transfer and
makes the comparison re-runnable afterwards.

## 5. Step 0 may be 967 YouTube fetches rather than a copy

There are **no dev clips on Atisri's Mac**: `data/clips/` is 931 clips with **zero** overlap against
`labels.dev` and 931/932 overlap against `labels.test` — it is the test split, and the only clip
directory present.

**Ask: do you have the dev-split clips on your Mac?** If not, step 0 is a full
`openasl_fetch.py --split dev` run from source, with whatever yield loss that carries, before any
frames reach the board — which makes it the long pole in this plan, where the doc budgets it as a
copy. If you do have them, say so and this collapses to a transfer.

## 6. Flags check out

For the record, every flag in the step 2 command exists in `unisign/train_adapt.py` with the meaning
assumed: `--label-smoothing`, `--lr`, `--warmup-epochs`, `--fps`, `--amp`, `--seed`, `--out-dir`.
Two the spec omits and should not: `--save-every` (Colab kills sessions mid-epoch; §L15 lost a run
that way) and `--epochs`, which defaults rather than being stated.

## 7. Not a correction, but it decides whether any of this happens

The spec assigns extraction, training and evaluation to the LM track on the grounds of board access.
Board runs have been yours for the whole project. **Atisri has to agree to that reassignment before
step 0 starts** — this is not ours to accept on her behalf.
