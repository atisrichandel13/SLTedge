# J13: the exact eval protocol the FP32 arm has to match, sent before the eval rather than after

*LM track → pose track, 2026-10-06, while J13 extracts. This is the ask I got wrong on J12 — I listed
four flags and missed `device`, which then had to be measured as a separate axis (§L18 addendum 1).
Sending the whole config this time.*

---

## 1. Every axis, from the FP16 arm's own JSON

`results/eval_n931_pruned_ours_fps24.json`'s `config`, which is what the FP32 arm must reproduce with
**`poses` as the only difference**:

| field | value |
|---|---|
| `ckpt` | `weights/openasl_pose_only_slt_pruned.pth` |
| `mt5` | `weights/mt5-base-openasl-pruned` |
| `labels` | `data/openasl_labels/labels.test` |
| **`device`** | **`cuda`** |
| `dtype` | `fp32` |
| `w8_runtime` | `dequant` |
| **`batch_size`** | **`1`** |
| `max_length` | `256` |
| `fps` | `24.0` |
| `src_fps` | `30.0` |
| `max_new_tokens` | `64` |
| `num_beams` | `4` |
| `limit` | `null` |
| `expect_n` | `931` |
| `poses` | **the new FP32 directory** — the only field that changes |

`batch_size` and `device` are the two that matter most and the two easiest to let drift: §L18
measured batching as changing **~30 % of sentences** and §L18 addendum 1 measured the device axis at
3/400 sentences and **+0.0514 BLEU-4**. Neither is large, but both are larger than nothing and the
whole point of this arm is to attribute a difference to quantisation.

## 2. `dtype: fp32` is the LM, not the pose engine — do not change it

Worth stating because it is genuinely confusable: that `dtype` field is the **mT5/ST-GCN** precision
and it is already `fp32` in both existing arms. **The FP16-vs-FP32 axis in J13 lives entirely in the
pose extraction**, upstream of the eval. The existing arm is not "an FP16 LM run". So the FP32 arm
takes `--dtype fp32` exactly as before — changing it would introduce a second variable and make the
result uninterpretable.

## 3. A manifest for the test set, since none existed

`results/pkl_split_rtmw_fp16.manifest` — **931 bare clip names**, generated from the FP16 arm's own
`names` field, so it is the definitive list of the paired set rather than a re-derivation.

**Convention note, and I got this wrong first:** the eval JSONs record names **with** `.mp4` (they are
the label keys), while your `package_dev_poses.sh` manifest records them **bare** (`basename -s .pkl`).
I have written this one **bare, to match yours** — both manifests now use one convention. Verified:
931 unique names, all present in `labels.test`, none with empty reference text, no duplicates.

**Ask: point the FP32 arm's guard at it**, so the new directory is checked by name and not only by
count. `--expect-n 931` catches a short extract; the manifest catches a *different* 931.

## 4. What we will produce when it lands, so nothing is run twice

Two paired bootstraps at **2000 draws** (the count §L19 standardised on), both ours:

1. **FP32 poses vs FP16 poses, n=931.** The headline. Both arms at `device cuda`, batch 1 — so this
   comparison is **fully protocol-clean**, which the −1.41 pose gap itself is not: the ceiling arm was
   scored at `device cpu`, the one impurity §L19 records.
2. **FP32 poses vs the authors' ceiling, n=931.** This is the one that actually decomposes the deficit
   — it measures what remains of the −1.41 when quantisation is removed, directly, rather than by
   subtracting (1) from −1.41, which would assume additivity nobody has established.

So **send the eval JSON with `names` recorded** (the default since 2026-09-26) and we will do both.

**Nothing is blocked on us** while it extracts.
