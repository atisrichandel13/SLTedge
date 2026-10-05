# Ask: adapt the pose encoder to *our* extractor's keypoints

*2026-10-05, pose track → LM track. You have the 967 dev clips and board access, so this is runnable
end to end on your side. Everything referenced is on `main`.*

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

**Step 2 — adapt, with your working recipe from L15.2.**

```bash
python unisign/train_adapt.py \
  --ckpt weights/openasl_pose_only_slt_pruned.pth \
  --mt5  weights/mt5-base-openasl-pruned \
  --poses results/pkl_dev_rtmw_fp16 \
  --labels data/openasl_labels/labels.dev \
  --lr 1e-5 --label-smoothing 0.0 --warmup-epochs 0.1 \
  --fps 24 --amp --seed 42 \
  --out-dir weights/adapt_ourposes_s42
```
Notes on the flags, each for a measured reason:
* `--label-smoothing 0.0` — **not the default 0.2.** L15.1: at 0.2 over a 26 K vocabulary the reported
  loss parks at a ~3.35 smoothing floor that reads as a training curve, and the recipe made the model
  *worse*. This is the single most important flag here.
* `--lr 1e-5` — the default `1e-4` is the one that degraded in L15.2.
* `--fps 24` — adapt at the rate we deploy. §2.5g: 24 fps is free on our own keypoints
  (+0.13 [−0.49, +0.74]).
* pruned checkpoint — it is the deployment choice on every axis (§5.4: −12 % system energy, half the
  peak memory and load time) and pruning costs −0.51, not established.

**Step 3 — evaluate on TEST with our poses, and against the un-adapted baseline on the same clips.**

```bash
python unisign/eval_openasl.py --ckpt <adapted> --mt5 weights/mt5-base-openasl-pruned \
  --poses results/pkl_split_rtmw_fp16 --labels data/openasl_labels/labels.test \
  --num-beams 4 --max-new-tokens 64 --batch-size 1 --fps 24 \
  --out results/eval_n400_adapt_ours_fps24.json
```
`results/pkl_split_rtmw_fp16/` is already on the board — 400 test clips through our extractor, the
exact set §2.5h measured, so the comparison is paired on identical clips.

## What success looks like

The number to beat is **21.57 BLEU-4**, and the headroom is the ceiling at **23.81**. Report
`unisign/bootstrap_ci.py results/eval_n400_pruned_ours_fps24.json results/eval_n400_adapt_ours_fps24.json`
— paired, same 400 clips. **Recovering even half the gap (~+1.1) would be the largest accuracy result
in the project.**

Please report **both metrics**. §2.5h and its addenda: pose-source effects register on BLEU-4 and not
ROUGE-L, frame-rate effects the other way round, and BLEU-4's interval is the narrower of the two in
all four comparisons we have on shared draws. Either metric alone can be blind to the effect.

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
