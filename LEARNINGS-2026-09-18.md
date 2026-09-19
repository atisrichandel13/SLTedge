# What we learned — 2026-09-18

Session focus: the LM track (Atisri). Frame-rate reduction as an energy lever (L7.1), the fine-tuning
harness (C8 / PROJECT-GUIDE 4.2), and the setup for the adaptation runs (4.4 / L11).
Numbers are copied from `results/RESULTS.md`, and every one of them comes from a real run.

---

## 1. Frame rate is the biggest energy lever, and 24 fps is free

The L7.1 sweep caps the encoder input length with no retraining (pruned + W8A16 model, all 976 test
clips, beam 4):

| cap (≈ fps) | BLEU-4 | Δ vs 256 (95% CI) | GCN + encoder ms (Mac CPU, relative only — board numbers come from the Jetson) |
|---|---|---|---|
| 256 (30) | 22.79 | — | 220 |
| 205 (24) | 22.62 | −0.17 [−0.61, +0.26] | 182 |
| 137 (16) | 21.17 | −1.61 [−2.36, −0.89] | 136 |
| 103 (12) | 19.52 | −3.26 | 106 |
| 68 (8)   | 14.76 | −8.02 | 84 |

- **24 fps costs nothing measurable.** The confidence interval spans zero.
- **16 fps costs 1.6 BLEU-4 without retraining.** 12 and 8 fps can't be used unless the model is adapted.
- GCN and encoder time scale linearly with frame count. Decoder time doesn't depend on it at all.
- The pose stage's energy is linear in frames processed. So 24 fps cuts about 20 % of pose energy at
  no BLEU cost, and 16 fps cuts about 47 % for −1.6 BLEU. **That's why we're doing the adaptation
  run:** if fine-tuning at 16 fps gets back about 1 BLEU, 16 fps becomes the deployment rate.
- Test-clip lengths: mean 217 frames, median 171, p95 545. Even the default cap of 256 already
  subsamples 305 of 976 clips.

## 2. A frame-length cap is not the same as a lower frame rate

- `--max-length L` (L7.1) only shortens clips **longer** than L. Short clips pass through unchanged.
- A real lower-fps camera thins **every** clip. We added `fps_ratio` to
  `common/pose_to_unisign.py::subsample`, exposed as `--fps` / `--src-fps` in `unisign/eval_openasl.py`
  and `unisign/train_adapt.py`. It first thins each clip uniformly to `round(T × fps/src_fps)`, then
  applies the usual 256 cap. Check: T=152 gives 101 at 16/24, 76 at 12/24 and 51 at 8/24.
  A 869-frame clip still ends up capped at 256.
- ⚠️ **Open question: the source frame rate.** L7.1 assumes 30 fps source video. The new `--src-fps`
  flag defaults to 24. These can't both be right. OpenASL clips come from YouTube, so the true rate
  may differ per video. Settle this before the Colab L11 runs, because it changes what "16 fps"
  actually means (ratio 0.67 vs 0.53).

## 3. How the authors trained Uni-Sign (from their GitHub code)

AdamW (eps 1e-9, weight decay 1e-4), cosine schedule, no warm-up, gradient clipping 1.0, label
smoothing 0.2, target sentences truncated to 50 tokens, padding tokens set to −100 in the labels,
lr 3e-4 (stage 3), batch 8, 20 epochs, bf16. Clips longer than 256 frames are randomly subsampled.
They use no pose augmentation.

Our choices on top of theirs:
- **lr 1e-4 instead of 3e-4.** We're adapting a checkpoint that has already converged, not training
  from scratch.
- **mT5 frozen.** Only the pose side trains: `proj_linear`, `gcn_modules`, `fusion_gcn_modules`,
  `part_para`, `pose_proj`. That's **5.35 M of 243.6 M params**, so a checkpoint of just the trainable
  weights is 64 MB, and the approach fits in Colab memory. mT5 also stays in eval mode, so its
  dropout is off.

## 4. The training harness works (C8, PROJECT-GUIDE 4.2 ✅)

`unisign/train_adapt.py` was smoke-tested on the Mac: 300 training clips, 2 epochs, no input shift,
evaluated on 200 test clips.

| checkpoint | BLEU-4 |
|---|---|
| before training | 14.71 |
| after epoch 0 | 14.08 |
| after epoch 1 | 14.49 |
| epoch 1, reloaded through `unisign.eval_openasl` | 14.55 |

- **With no input shift, BLEU wobbles by about ±0.6.** That's the noise floor for this harness on a
  slice this small. Only an effect bigger than that counts.
- The saved `adapted_full.pth` reloads through the standard eval script and scores within 0.06 BLEU of
  the training log's own number, so saving and loading are correct.
- Wall time on Mac CPU: about 3 min per epoch of training and about 6 min to evaluate 200 clips.
- For Colab, add `--amp --batch-size 16`. `--amp` turns on bf16 autocast and only works on CUDA.

## 5. The first 200 test clips are harder than the rest

The pruned model scores **14.71 BLEU-4 on the first 200 test clips but 22.87 on all 976.** We checked
this against the saved full eval: the first 200 clips score exactly the same there, so it's not a bug.
Takeaway: **only compare numbers measured on the same clip subset.** For the 16 fps run, the baseline
is the run's own `--eval-before` score at `--fps 16` on those same 200 clips. It is not 21.17, which
was measured on all 976 clips with a length cap.

## 6. Bugs hit and fixed

| Symptom | Cause | Fix |
|---|---|---|
| Saved checkpoint was 975 MB instead of 571 MB | State dict was saved in fp32 | Cast each tensor back to the source checkpoint's dtype (bf16) |
| `Can't pickle local object 'make_collate.<locals>.fn'` with `--num-workers 2` | macOS starts DataLoader workers with *spawn*, and spawn can't pickle closures | Replaced the closure with a module-level `collate_batch` |
| Pre-training BLEU looked low (14.71) | Harder subset (§5) | No fix needed |

## 7. Where data and compute live

- **Mac:** only a 300-clip debug slice (`data/openasl_train_pose_smoke/`, 239 MB, mean 305 frames,
  137 clips over 256 frames). It's for proving the code works. It doesn't count as the PROJECT-GUIDE
  4.1 dataset.
- **Colab + Drive:** the full training set (96,476 labelled clips, about 30 GB of poses) and all real
  training runs (4.3 seed variance, 4.4 adaptation).
- **Jetson:** inference and power measurements only, run by the teammate. No training. No sudo, no
  pip-installing torch, and never `--rm` the `lpcv-work` container.
- `runs/` is in `.gitignore`, because checkpoints are too large to commit.

## 8. Still in progress / next

- **16 fps adaptation on the Mac slice** (`runs/c8_adapt16`) is still running. Expected result:
  probably nothing, since 300 clips is too few to show a real effect. Its job is to exercise the
  `--fps` path end to end before Colab.
  Baseline before any training, at 16 fps (ratio 0.667) on the same 200 clips: **13.79 BLEU-4**. At
  full frame rate those clips score 14.71, so thinning to 16 fps costs 0.92 BLEU-4 on this subset.
- **Colab:** fetch the full training poses into Drive (4.1) → seed-variance runs (4.3, L10) →
  adaptation at 16 fps and maybe 12 fps (4.4, L11).
- Other open items: L6 remainder, L13 LLM correction, frontier plot script (5.4), GitHub remote (C0),
  the Jetson half of C4, and recording the board rows (1.7). Teammate queue: J1–J5.
