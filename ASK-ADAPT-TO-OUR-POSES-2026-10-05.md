# Ask: adapt the pose encoder to *our* extractor's keypoints

*2026-10-05, pose track → LM track. **Revised 2026-10-05** after
`REPLY-J9-ADAPT-2026-10-05.md`; see `REPLY-TO-J9-ISSUES-2026-10-05.md` for the full response. Three
things in the first draft were wrong and are fixed below: it had no control arm and so could not
attribute its own result, it implied training runs on the board, and it budgeted step 0 as a copy when
there are **no dev clips anywhere** — `data/clips/` is 931 test clips with zero dev overlap, so step 0
is a full YouTube fetch and is the long pole.*

## Why this is the one accuracy lever left

Measured, at the deployable config (pruned ckpt, RTMW-l-m FP16 board poses, 24 fps, beam 4, cap 64):

| | BLEU-4 |
|---|---:|
| **deployable, n=400** | **21.57** |
| authors' keypoints, same clips | 23.81 |
| **pose-substitution gap** | **−2.24 [−3.50, −1.09]** — established (§2.5h) |
| pruning | −0.51, not established |
| 24 fps vs source | +0.13, free |

**Everything except the pose gap is already free.** That gap is ~2.2 of the ~2.3 points available, so
it is the accuracy story.

Three results say it is a **distribution shift, not an information loss**, which is why adaptation
rather than a better extractor is the move:

1. **§2.5i** — our keypoints sit a median **0.47 %** of the frame from the authors', with a mean
   signed offset of 0.0036, i.e. essentially zero. The information is present and the coordinate
   frame is right.
2. **§2.5j** — swapping only the hands exchanges 1.60 BLEU-4, but *repairing* our hands recovers only
   **+0.35 [−0.69, +1.61]**, not established. Degrading hands alone (−1.60) and body+face alone
   (−1.40) nearly equal degrading both (−1.75): strongly sub-additive. **No keypoint group carries
   it** — the difference is diffuse.
3. **C2** — the authors' poses came from the *same* 256×192 RTMW/RTMPose family we use, so there is no
   better-architecture upgrade waiting.

And your own L15.2 already showed the harness helps: at `lr 1e-5, ls 0.0, warmup 0.1` fine-tuning
**improved** the model (+0.18 BLEU-4 / +0.70 ROUGE-L) with **no** distribution shift to adapt to. Here
there is a real one.

## Why dev, and why on the board

* **Dev, not train** — adapting on test is leakage and would invalidate the n=400 numbers above. Dev
  is 967 clips, which you have. Adapt on dev, evaluate on test.
* **On the board** — the adaptation target must be the *deployed* distribution, which is the FP16
  TensorRT engine. An off-board ONNX Runtime extraction is a near neighbour (§P4: FP16 TRT agrees
  with FP32 on 99.81 % of consumed keypoints within 5 px) but not the same thing.

## What to run

**Step 0 — get the dev clips onto the board with `meta.json` per clip.** `09_batch_clips.py` needs
`meta.json` for the square-norm geometry, so raw video is not enough; they must come from
`data/openasl_fetch.py --split dev`. If you fetched them another way, re-run the fetcher — it now has
an **effective-rate guard** (added today) that raises on a truncated extraction instead of writing a
4-frame clip, which is the bug §2.5i found in our own test fetch.

**Step 1 — extract dev poses on the board, in batches.**

```bash
# ~967 clips of frames is ~6.9 GB; the board has ~17 GB free alongside weights/ and models/.
# So push a few hundred clips into data/clips_dev/, then:
./jetson/p13_dev_poses.sh            # extracts, then reclaims frames for clips that have both pkls
# push the next batch, run it again. Resumable: a clip whose pkl exists is skipped.
```
Produces `results/pkl_dev_rtmw_fp16/` (square-norm — the one to train on) and `..._raw/` (crop frame).

**Step 2 — adapt, with your working recipe from L15.2. On Colab, not the board** — the on-board
requirement is about matching the deployed *extraction* distribution (step 1), not about where
gradients are computed. A 15 W Orin Nano with no sudo and one process at a time is not a training
device.

```bash
python unisign/train_adapt.py \
  --ckpt weights/openasl_pose_only_slt_pruned.pth \
  --mt5  weights/mt5-base-openasl-pruned \
  --poses results/pkl_dev_rtmw_fp16 \
  --labels data/openasl_labels/labels.dev \
  --lr 1e-5 --label-smoothing 0.0 --warmup-epochs 0.1 \
  --fps 24 --amp --seed 42 --epochs 1 --save-every 1 \
  --out-dir weights/adapt_ourposes_s42
```
Notes on the flags, each for a measured reason:
* `--label-smoothing 0.0` — **not the default 0.2.** L15.1: at 0.2 over a 26 K vocabulary the reported
  loss parks at a ~3.35 smoothing floor that reads as a training curve, and the recipe made the model
  *worse*. This is the single most important flag here.
* `--lr 1e-5` — the default `1e-4` is the one that degraded in L15.2.
* `--fps 24` — adapt at the rate we deploy. §2.5g: 24 fps is free on our own keypoints
  (+0.13 [−0.49, +0.74]).
* `--save-every 1` and an explicit `--epochs` — Colab kills sessions mid-epoch and L15 lost a run
  that way; neither should be left to a default.
* pruned checkpoint — it is the deployment choice on every axis (§5.4: −12 % system energy, half the
  peak memory and load time) and pruning costs −0.51, not established.

**Step 2b — the control arm. This is not optional.** Same recipe, same `--fps 24`, same seed,
trained on the **authors'** dev poses, evaluated on the **same** our-pose test n=400. Without it a
gain cannot be separated from "fine-tuning on 967 dev clips helps regardless of whose keypoints they
are" — and L15.2 measured that generic gain at **+0.18 BLEU-4 / +0.70 ROUGE-L** with no shift at all.
The contrast of record is therefore `adapted-on-ours − adapted-on-theirs`, not
`adapted − un-adapted`.

The authors' dev poses are **not** on the Mac (`data/openasl_pose/` is 976 test pkls, 0 of them in
`labels.dev`), but they are pose pkls rather than video: `data/openasl_pose_fetch.py --split dev`
pulls them by HTTP range at ~680 MB, now with size and CRC verification.

**Step 3 — evaluate on TEST with our poses, and against the un-adapted baseline on the same clips.**

```bash
python unisign/eval_openasl.py --ckpt <adapted> --mt5 weights/mt5-base-openasl-pruned \
  --poses results/pkl_split_rtmw_fp16 --labels data/openasl_labels/labels.test \
  --num-beams 4 --max-new-tokens 64 --batch-size 1 --fps 24 \
  --out results/eval_n400_adapt_ours_fps24.json
```
`results/pkl_split_rtmw_fp16/` is 400 test clips through our extractor — the exact set §2.5h measured,
so the comparison is paired on identical clips. It is on **both** the board and the Mac working tree
(untracked, covered by the `results/pkl_*/` ignore rule, which is why it is invisible from a fresh
clone). Nothing needs to move for step 3.

## What success looks like

The number to beat is **21.57 BLEU-4**. **Do not measure the recovered fraction against the 23.81
ceiling** — that is the *un-adapted* authors'-keypoint score, so crediting adaptation against a fixed
23.81 folds the generic fine-tuning gain into the numerator. Report the fraction against the control
arm, or report the raw delta and name the denominator. Report
`unisign/bootstrap_ci.py results/eval_n400_pruned_ours_fps24.json results/eval_n400_adapt_ours_fps24.json`
— paired, same 400 clips. **Recovering even half the gap (~+1.1) would be the largest accuracy result
in the project.**

Please report **both metrics**. §2.5h and its addenda: pose-source effects register on BLEU-4 and not
ROUGE-L, frame-rate effects the other way round, and BLEU-4's interval is the narrower of the two in
all four comparisons we have on shared draws. Either metric alone can be blind to the effect.

> **STALE IN THREE PLACES, 2026-10-05, from the n=931 board arm — and this doc is J9's specification,
> so a run executed against it as written would be wrong.** Flagged by the pose track; nothing here is
> overwritten.
>
> 1. **Score at n=931, not n=400.** The board now holds all 931 paired test clips (§2.5k). The pairing
>    is `results/eval_n931_pruned_ours_fps24.json` against the adapted arm, not the `n400` files named
>    above. The ceiling arm is your `eval_test976_ceil_b4_fps24_named.json`, intersected by name.
> 2. **"Half the gap (~+1.1)" is no longer the target, and half is no longer the bar.** The gap is
>    **−1.4077 [−2.19, −0.63]**, not −2.24, so half of it is **~+0.70**. And half does not clear: the
>    measured half-width is **±0.7776**, so the recovery needed to return *established* two-sided is
>    **55.3 %**, agreed in writing in §L21 before the run. ~+1.1 would now be ~78 % recovery.
> 3. **"Pose-source effects register on BLEU-4 and not ROUGE-L" is false at full width.** At n=931
>    **both** establish the pose term — BLEU-4 −1.41 [−2.19, −0.63] and ROUGE-L −1.33 [−2.23, −0.49] —
>    and the two converge (1.41 against 1.33) where they differed ~2× at n=400. §2.5h carries a dated
>    note on this. **Your instruction to report both metrics is unaffected and is now better founded,
>    not worse**: the asymmetry was a power artefact, and the metric that resolved the effect at n=400
>    was the one that overstated it by 0.83.
>
> Unaffected: the control arm, the "not asking for" list, and every pitfall below.

## Pitfalls I hit, so you do not have to

* **`jetson/run.sh exec-batch` is `docker exec` without `-i`** — a heredoc never reaches the
  container. `python3 -` reads an empty program, exits 0, and the next step runs on nothing. Use a
  file.
* **The full checkpoint needs >5.2 GB MemFree to load, not the ~4 GB §5.2 once claimed**, and the
  reclaim target must be sized to the *target* (`drop_file_cache.py --target-free-mb=6200`). The
  pruned checkpoint needs ~1.5 GB and is what the commands above use, so this mostly will not bite.
* **One PyTorch/TensorRT process at a time.** Two kill each other with the
  `CUDACachingAllocator` assert. It has happened three times.
* **Check occupancy** (`who`, and `jetson/run.sh whoelse`) before anything timed. An unrecorded
  co-tenant invalidated a whole set of memory probes (§5.2).
* **Hold batch size at 1** for any eval you compare against the numbers above — §5.1 found 2 of 30
  clips change output between batch 8 and batch 1.

## Not asking for

Pose-model swaps (P6/P7). §2.5j rejects the hand hypothesis and C2 says the family is already matched,
so a swap would be board time spent on at most a fifth of the gap. If adaptation lands, it is also the
thing that makes a sweep worth revisiting, not the other way round.
