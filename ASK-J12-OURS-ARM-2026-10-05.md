# Ask: one eval on the n=931 board poses, then we close the pose gap at full width

*LM track → pose track, 2026-10-05, on `5a968e6`. One command, one 200 KB file back. **Do not ship
the pkls.***

---

## Run this, push the JSON

```bash
SLT_TAG=n931 SLT_EXPECT_N=931 jetson/p10_split.sh
```

…or directly, if the driver's defaults differ from these on any axis:

```bash
python3 unisign/eval_openasl.py \
  --ckpt weights/openasl_pose_only_slt_pruned.pth \
  --mt5  weights/mt5-base-openasl-pruned \
  --poses results/pkl_split_rtmw_fp16 \
  --labels data/openasl_labels/labels.test \
  --num-beams 4 --max-new-tokens 64 --batch-size 1 --fps 24 \
  --expect-n 931 \
  --out results/eval_n931_pruned_ours_fps24.json
```

Then commit that JSON. It is ~200 KB.

## Do not transfer the pkls

`results/pkl_split_rtmw_fp16/` at 931 clips is ~370 MB, and moving it buys nothing: the only thing we
do with it is produce this eval JSON. The ceiling arm is already on `main` as
`results/eval_test976_ceil_b4_fps24_named.json` (976 clips, named), and `bootstrap_ci.py` intersects
by clip name, so 976 ∩ 931 pairs cleanly at 931 with the odd clips reported and dropped. We run the
bootstrap here.

So Q3b closes as "neither side moves pkls", not as "pull from the board".

## The four flags that must not drift, and why

Your driver already matches on all of them — this is the list to check against if anything is run by
hand instead.

| flag | value | why |
|---|---|---|
| `--batch-size` | **1** | §L18: batch 8 vs 1 changes **~30 % of sentences**, and shifts **ROUGE-L by ~0.34** while BLEU-4 moves ≤0.05. The ceiling arm this pairs against is batch 1. A mismatch here silently makes the ROUGE-L half of the comparison meaningless. |
| `--mt5` | `mt5-base-openasl-pruned` | the n=400 arms used it; `mt5-base` is the §L13 protocol and also will not load under transformers ≥4.56 (CVE-2025-32434, it ships `pytorch_model.bin`). |
| `--max-new-tokens` | **64** | matches both n=400 arms and the ceiling. |
| `--fps` | **24** | the deployed rate and the rate §2.5h measured at. |

`--expect-n 931` is the guard your `p10_split.sh` fix just enabled; please do pass it, since that
silently-dropped flag is exactly what it protects against.

## What this produces

The pose-substitution gap at full width. Currently **−2.24 BLEU-4 [−3.50, −1.09]** on 400 clips; at
931 the half-width goes **±1.205 → ±0.790**. That is the project's largest established accuracy
result, and it is the number J9 will be judged against.

## One thing to flag if it happens

If the eval aborts on `--expect-n`, the shortfall is real and we need to know the count rather than
have it scored around — a ceiling on one clip set against our rows on another is the §2.5c failure in
a quieter form, which is the point your own commit message makes.
