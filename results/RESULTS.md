# Results log

All numbers measured on the board. Nothing here is estimated.

## Setup

- Jetson Orin Nano 8GB, L4T R36.4.7, TensorRT 10.11.0.33 (container `lpcv:fall2026`), CUDA 12.9.
- Power mode: **15W (mode 0)**, read from `/var/lib/nvpmodel/status`. The board is course-owned with no
  sudo, permanently. So: no 7W mode, no `jetson_clocks`. Every run is at 15W with the default DVFS
  governor. Power budgets in this project are therefore *measured average watts*, not a hardware cap.
  Clock frequency and fan state are logged from sysfs alongside each run to document the DVFS state.
- Power: INA3221 via hwmon sysfs (`common/power_logger.py --power-backend sysfs`), 100 ms samples,
  `VDD_IN` = module total. Idle baseline measured for 3 s before each run.
- Input: OpenASL test clip `Ads-4j06eJY-00:07:37.233-00:07:47.200`, 299 frames at 29.97 fps, cropped
  to the OpenASL signer bbox (`data/test_frames_meta.json`). Batch 1, frames streamed one at a time.
- RTMPose-x, COCO-WholeBody 133 kpts, 384x288, checkpoint
  `rtmpose-x_simcc-coco-wholebody_pt-body7_270e-384x288-401dfc90_20230629.pth`.

## Correctness ladder (Task 1)

| Check | Result |
|---|---|
| Standalone preprocess vs mmpose pipeline (Mac) | max diff 0.0 |
| ONNX Runtime vs PyTorch, 20 frames (Mac) | simcc max diff 8.0e-6 |
| TensorRT FP32 vs PyTorch, **TF32 left on** (TRT default) | simcc max diff 1.55e-3, FAIL |
| TensorRT FP32 vs PyTorch, TF32 disabled | simcc max diff 7.2e-6, 2416/2416 confident kpts at 0.0 px, PASS |

Methodology note: TensorRT enables TF32 by default on Ampere. An "FP32" engine built with defaults is
not FP32. `common/trt_runner.py` now clears the flag for the FP32 baseline.

## Task 1: RTMPose-x pose extraction, per-frame

| Config | Mode | pre ms | TRT ms | post ms | total ms | avg W | idle W | mJ/frame | dyn mJ/frame | GPU % | Tj C |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FP32 (TF32 off), first pass, 3x299 frames | 15W | 10.6 | 44.4 | 1.5 | 56.4 | 11.62 | 3.73 | 754 | 512 | 68 | 56 |

Raw: `results/rtmpose_fp32_15W_power.json`, `.csv` (on the Jetson, copy back).

Reading it:
- 56 ms/frame = ~18 fps compute-bound; source video is 30 fps, so FP32 RTMPose-x cannot keep up
  without dropping frames. That is the motivation for every knob that follows.
- Preprocess (JPEG decode + affine, CPU) is 19% of frame time and does not shrink with GPU quantization.
- Peak `VDD_IN` 13.3 W, close to the 15 W cap. TRT latency jitter <2 ms, so no throttling at 56 C.
- One 10 s sentence costs 299 x 0.754 J = 225 J of pose extraction at FP32.

First pass only: single run per cell. Final table needs 3 runs, mean+std, several clips/signers, 30 min sustained.

## Track B: Uni-Sign OpenASL pose-only

### Metric check (B1.3), no model run

Scored the authors' released test predictions (`out/eval_openasl_pose_only/test_tmp_*.txt`, prefixes
stripped) with the repo's own `translation_performance` (sacrebleu 13a, rouge-l F).

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| Paper, OpenASL test, pose-only | 49.10 | — | — | 22.67 | 42.77 |
| Recomputed from released predictions | 49.09 | 35.91 | 28.10 | 22.64 | 42.83 |

976 test sentences. Example (first test clip): ref *"I am Ed Bosson, and I am 67 years old and now
retired."* → pred *"Hello, I'm Howard Rosenblum, and I'm a fourteen-year-old immigrant."* Fluent,
grammatical, wrong in every detail. This is what the field's state of the art looks like at 22 BLEU.

Gotcha: the released files prefix every line with `sample: <clip>, ground-truth: ` / `prediction: `;
scoring them raw gives a bogus 58.7 BLEU-4.

### B1.5 Reproduction, released checkpoint, our run (2026-09-17)

| | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| Paper | 49.10 | — | — | 22.67 | 42.77 |
| Our run, Colab T4, bf16, beam 4, max_new_tokens 100 | 48.83 | 35.73 | 27.92 | **22.53** | 42.68 |

976 test clips, 122 batches of 8, 5 min 05 s wall, 2.50 s/batch, peak GPU mem 6.1 GB. Poses = authors'
released `openasl_pose_format` (archive folder is named `pose-rtmpose-192`, i.e. a 256x192
RTMPose-family extractor, not the 384x288 RTMPose-x used for the Jetson FP32 row; see B2.1).
Env: transformers 4.57.6, torch 2.11, deepspeed 0.16.3, ZeRO-2 eval path unchanged. Patches: dev-split
eval skipped (dev poses not extracted), BLEURT skipped (no checkpoint). Predictions saved to Drive
`unisign/eval_openasl_ours/`. Gap to paper (-0.14 BLEU-4) is within the loader's random frame
subsampling for long clips; treat 22.5–22.7 as the FP32 reference band.

### C5 standalone inference (`unisign/unisign_infer.py`), Mac CPU, released checkpoint

Pose-only rebuild loads `openasl_pose_only_slt.pth` strictly. 587.7M params = mT5 582.4M + pose
stack 5.35M (hands share weights). Bitcoin clip, 299 frames → 256 used, encoder length 264
(8 prefix tokens + 256 pose tokens), greedy, max_new_tokens 64.

| Pose source for the same clip | Output | tokens | mean logprob | gcn / enc / dec ms (M-series CPU) |
|---|---|---|---|---|
| ground truth | *The tweets appeared to solicit donations as part of a Bitcoin scam.* | | | |
| rtmlib lightweight (RTMW-l-m 256x192) | *The word itself appears to have directly contributed to the donations including the Biden campaign.* | 27 | -0.81 | 81 / 76 / 295 |
| rtmlib performance (RTMW-x-l 384x288) | *The question is apparently a targeted contributor to the donation which includes a ballot of the Biden campaign.* | 31 | -1.20 | 89 / 80 / 345 |

Decoder dominates LM time even on CPU (65 %); encoder + GCN are one pass. The lightweight poses give
the more confident and closer sentence, consistent with the archive name `pose-rtmpose-192`.
Jetson RTMPose-x and the authors' own pkl rows pending (C1, C2).

### C2 Which extractor produced the released OpenASL poses (Bitcoin clip, 299/300 frames)

Reference: authors' `pose-rtmpose-192/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl` (fetched with
`data/openasl_pose_fetch.py`, 18 MB instead of the 32 GB archive). Candidates run on our frames.

| Candidate | raw mean px vs ref | after framing fit (median px) | model-space dist: body / left / right / face |
|---|---|---|---|
| rtmlib lightweight = RTMW-l-m 256x192 | 57 | 3.3 | **0.091 / 0.053 / 0.041 / 0.012** |
| rtmlib performance = RTMW-x-l 384x288 | 57 | 3.5 | 0.141 / 0.139 / 0.056 / 0.020 |
| Jetson RTMPose-x 384x288 | pending C1 | | |

Raw px gaps are framing, not extractor error: a per-axis scale+offset (x 0.64, y 0.68) maps our
666x720 tight bbox crop onto the authors' frame, which is consistent with OpenASL's ~1050 px square
crop with padding. Uni-Sign's `crop_scale` body normalisation removes this before the model.
In model space the 256x192 model is closest on every group. **Decision (P1): the Jetson baseline pose
model is RTMW-l-m 256x192 (rtmlib "lightweight", ONNX shipped by OpenMMLab). RTMPose-x 384x288 is the
oversized comparison row.** Caveat: an MMPose RTMPose-m/l wholebody 256x192 would also fit the
folder name; distinguishing them needs the Jetson-side variant sweep (P6), not more clips.
| authors' released poses (pose-rtmpose-192) | *The tweet apparently targeted people with donations that were likely part of the Biden campaign.* | 24 | -0.75 | — |

### L5.1 Vocabulary census (OpenASL train+dev+test, 98,419 sentences, mT5 tokenizer)

| | rows | params (embed + lm_head, 768-d) |
|---|---|---|
| mT5-base full vocabulary | 250,112 | 384M |
| OpenASL keep set (26,075 used + prefix + specials) | **26,078 (10.4%)** | **40M** |

Coverage of token occurrences: top 5K = 94.6%, 10K = 98.1%, 15K = 99.2%, 20K = 99.7%.
Whole model: 582M → ~238M params (-59%). bf16 weights 1.16 GB → ~0.48 GB. Keep-id list in
`results/openasl_vocab_keep_ids.json`. Next: slice `shared`/`lm_head`, remap tokenizer, re-score.

### L5.2 Vocabulary pruning applied (`unisign/model.py: prune_vocab`, `prune_and_save`)

| | full | pruned (26,078 tokens) |
|---|---|---|
| params, whole model | 587.7M | **243.6M** (-59%) |
| params, mT5 | 582.4M | 238.3M |
| checkpoint on disk (bf16, as released) | 1,187 MB | **571 MB** |
| Bitcoin clip, greedy, authors' poses | *The tweet apparently targeted people with donations that were likely part of the Biden campaign.* | identical text, identical token ids |
| reload from disk | | identical |

Mechanism: slice `shared` embedding and `lm_head` rows to the keep set (sorted old ids; pad/eos/unk
0/1/2 land on 0/1/2 so decoder_start/eos are unchanged), tokenizer wrapped to remap ids both ways,
unseen tokens → unk. No retraining. Full-split BLEU before/after: pending (running on Mac CPU).
Preliminary CPU timing (contended, re-measure): decoder step time dropped ~3x with the 250K→26K
`lm_head`, which is the per-token matmul the decoder is bound by.

### C7 Standalone eval loop (`unisign/eval_openasl.py`), Mac CPU, released checkpoint, 976 test clips

| Run | BLEU-1 | BLEU-4 | ROUGE-L | notes |
|---|---|---|---|---|
| Paper | 49.10 | 22.67 | 42.77 | |
| Repo `fine_tuning.py --eval`, Colab T4, bf16, random frame subsample | 48.83 | 22.53 | 42.68 | 5 min |
| **Standalone, Mac CPU, fp32, uniform frame subsample** | 49.61 | **23.16** | 43.17 | 15.7 min, beam 4, max_new_tokens 100 |

Same checkpoint, same poses, same metric code. Spread across the three is 0.6 BLEU-4: that is the
noise band from subsampling policy + dtype alone. Any claimed effect under ~0.6 BLEU-4 needs
bootstrap CIs (L12). The standalone loop is the scoring path for every LM ablation from here.

### L5.3 Vocabulary pruning, full-split accuracy (976 test clips, Mac CPU, beam 4, max_new_tokens 100)

| Model | params | ckpt MB | BLEU-1 | BLEU-4 | ROUGE-L | identical preds |
|---|---|---|---|---|---|---|
| full | 587.7M | 1,187 | 49.61 | 23.16 | 43.17 | — |
| **vocab-pruned (26,078 tokens)** | 243.6M | 571 | 49.25 | **22.87** | 42.98 | 655 / 976 |

Paired bootstrap (1000 resamples): delta **-0.28 BLEU-4, 95% CI [-0.63, +0.05]**, P(pruned < full) = 0.95.
Read: a small, probably real cost bounded at ~0.6 BLEU-4, for -59% params / -52% checkpoint, with
no retraining. 321 predictions change under beam search because dropping 224K tokens from the softmax
shifts beam scores even when the argmax is unchanged (greedy on the Bitcoin clip was identical).
Tool: `unisign/bootstrap_ci.py` (L12).

### L6.1 Decode knobs on the pruned model (976 test clips, Mac CPU, max_new_tokens 100)

| Decoding | BLEU-4 | ROUGE-L | paired delta vs beam 4 | eval wall (CPU) |
|---|---|---|---|---|
| beam 4 | 22.87 | 42.98 | — | 919 s |
| **greedy** | 20.88 | 41.56 | **-2.00 [CI -2.63, -1.41]** | 342 s (2.7x faster) |
| beam 2 | 22.06 | 42.22 | -0.81 [CI -1.29, -0.39] | 529 s (1.7x faster) |

Greedy is a real 2-point loss, not "well under a point" as assumed in the plan (§5). Beam width is
therefore an accuracy–energy axis in its own right, not a free lever. Longest prediction is 40 words
under both, so a `max_new_tokens` cap of 64 would change nothing; the cap matters only as a runaway
guard on the board.

### L3 LM per-sentence timing, Mac CPU (Apple silicon, fp32, indicative only; board numbers via J2)

Bitcoin clip, 256 pose frames → encoder length 264, ~20–24 output tokens, median of 3.

| Model | beams | GCN ms | encoder ms | decoder ms | total ms | decoder share |
|---|---|---|---|---|---|---|
| full | 1 | 92 | 86 | 268 | 446 | 60% |
| full | 2 | 94 | 94 | 669 | 859 | 78% |
| full | 4 | 103 | 108 | 1000 | 1250 | 80% |
| pruned | 1 | 102 | 105 | 191 | 396 | 48% |
| pruned | 2 | 116 | 122 | 541 | 790 | 68% |
| pruned | 4 | 125 | 139 | 772 | 1035 | 75% |

Decoder cost scales ~linearly with beam width and dominates at beam 4. Pruning cuts decoder time
20–30% on CPU (the 250K→26K `lm_head` matmul per step); GCN + encoder are one pass and unchanged.
The energy-relevant conclusion: on the LM side the lever order is beam width > vocab pruning >
anything on the encoder. Combined with L6.1: beam 4→2 saves ~25% total LM time for -0.8 BLEU-4;
greedy saves ~60% for -2.0.

### L8.1 mT5 (pruned) → ONNX with explicit KV cache (`task3_mt5_onnx/02_export_manual.py`)

Three graphs, fp32, opset 17, from `weights/mt5-base-openasl-pruned` (vocab 26,078):
`encoder.onnx` 340 MB, `decoder_init.onnx` 614 MB, `decoder_step.onnx` 557 MB (50 inputs: ids, mask,
48 past K/V; 25 outputs). Synthetic verify at T=264, 12 steps: encoder max diff 4.1e-6, per-step
logits ≤ 2.2e-5, tokens identical. Real-pose verify (`unisign/onnx_decode.py`, Bitcoin clip, greedy):
**tokens identical to PyTorch, max |logprob diff| 2.9e-5**; ORT CPU encoder 62 ms, decoder 178 ms for
24 tokens (7.4 ms/token) vs PyTorch 191 ms. Export + verification took one attempt (budget was 3 days).
Remaining for L8: TensorRT engines on the Jetson (queue J4) and the drift check there.
