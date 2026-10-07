# One dev pkl carries 7 frames of all-NaN confidence scores — and it stopped a training arm dead

*LM track → pose track, 2026-10-07. A defect in `pkl_dev_rtmw_fp16.tar`, found by training on it.*

---

## 1. The clip, and the exact signature

**`rlUUw27_6kM-00:17:09.233-00:17:15.433`**, in `pkl_dev_rtmw_fp16.tar`.

| | |
|---|---|
| `scores` NaN | **931** values |
| `keypoints` NaN / inf | **0** |
| `max abs(keypoint)` | 1.046 — in range, nothing overflowed |
| frames in clip | 149 |

931 = **7 × 133** exactly. So **seven complete frames have every one of their 133 confidence scores
NaN**, while the keypoint coordinates for those same frames are finite and in range. This is the
confidence channel alone, in whole frames — not scattered values, and not an fp16 overflow, which
would have shown in the coordinates first.

**It is the only one.** Scanned every pkl in both archives:

| archive | pkls with non-finite values |
|---|---|
| `pkl_dev_rtmw_fp16.tar` | **1 of 918** |
| `pkl_split_rtmw_fp16.tar` | **0 of 931** |

## 2. What it did, and why nothing caught it before

The NaN scores reach the encoder, the loss comes back NaN, and `.backward()` writes NaN into all
5.35 M trainable parameters in one step. The arm then **trained to completion, saved a 570 MB
checkpoint, and exited 0**. It scored **BLEU-4 0.00, ROUGE-L 2.97, every one of 931 clips decoded as
`"a"`** — and `colab_j9_step2.py` treats that checkpoint's existence as the "already trained" skip
key, so a re-run would have declined to fix it.

That was our bug, not yours, and it is fixed in `0a1354f`: `unisign/train_adapt.py` now refuses to
backward or save on a non-finite loss and names the clips in the offending batch. The batch was 206 of
230, `ntok=71` — the targets were fine, which is what pointed at the inputs.

**Why no earlier run hit it:** the un-adapted and FP32 arms only ever *scored* poses, and the test
archive is clean. This clip is in dev, and dev is read only when something trains. §L17's probe
trained at 16 fps on a different subset.

## 3. Two asks

1. **Is the NaN from RTMW or from the fp16 quantisation step?** The keypoints being finite while the
   scores are NaN in whole frames suggests the pose estimator emitted them, rather than a
   quantisation overflow — but that is your pipeline and the distinction decides whether anything
   else is affected. If RTMW can emit NaN confidence on a whole frame, the FP32 arm's poses may carry
   it too and nobody has scanned those.
2. **Re-extract that one clip**, or tell us the NaN is expected and should be imputed. We will not
   impute it ourselves: replacing a NaN confidence with 0 is an edit to a measured artefact, and
   whether "no detection" is the right reading of it is yours to say, not ours to assume.

## 4. What we are doing meanwhile, so the number is not misread

Re-running arm 2 on **917** clips with this one excluded, via
`results/pkl_dev_rtmw_fp16_minus_nan_scores.manifest` (committed, 917 names, the original minus that
clip). The arm will be reported as trained on 917, not 918, wherever it appears.

**This does not break the comparison.** The control arm already trains on a different set — the
authors' dev poses, whatever the public fetch yields — so the arms were never matched on training-set
size. What must match is the recipe, and it does.
