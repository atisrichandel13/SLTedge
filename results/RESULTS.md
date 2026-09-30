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
48 past K/V; 25 outputs). Verified on real poses only (`unisign/onnx_decode.py`, Bitcoin clip, greedy):
**tokens identical to PyTorch, max |logprob diff| 2.9e-5**; ORT CPU encoder 62 ms, decoder 178 ms for
24 tokens (7.4 ms/token) vs PyTorch 191 ms. Export + verification took one attempt (budget was 3 days).
Remaining for L8: TensorRT engines on the Jetson (queue J4) and the drift check there.

### L3.2 `max_new_tokens` cap sweep (pruned model, 976 test clips, Mac CPU, beam 4)

| cap | BLEU-4 | ROUGE-L | Δ BLEU-4 vs cap 100 (paired bootstrap, 1000) | preds at/over cap | sentences changed |
|---|---|---|---|---|---|
| 100 (reference) | 22.87 | 42.98 | — | 0 | — |
| 64 | 22.87 | 42.98 | 0.00 (identical output) | 0 / 976 | 0 |
| 48 | 22.73 | 42.96 | −0.14 [−0.30, −0.03], P(B<A)=0.994 | 39 / 976 | 28 |

Files: `results/eval_test_pruned_mac.json`, `results/eval_test_pruned_mnt64_mac.json`, `results/eval_test_pruned_mnt48_mac.json`.
Prediction length (mT5 tokens): mean 21.2, median 19, p95 44, max 62. References: mean 21.4, max 136.
Reading: the longest prediction is 62 tokens, so cap 64 is free and is the deployment default
(`unisign_infer.py` already defaults to 64). Cap 48 truncates 4 % of sentences for −0.14 BLEU-4,
statistically real but small. The cap only bounds the worst case; the mean sentence (21 tokens) is
unaffected, so the energy saving is in tail latency, not the average. Beam width (L6.1) remains the
lever that moves the average decoder cost.

### L9.1 Weight-only INT8 (W8A32) on the pruned mT5 (`unisign/quant.py`), 976 test clips, Mac CPU, beam 4, cap 64

**Precision, stated explicitly because the name has caused a misreading:** weights are int8, the
per-row scale is *stored* fp16 and cast up before use, and activations / matmul / accumulation are
**fp32** (`compute_dtype=torch.float32`). No activation is ever fp16. This is the distinction that
matters given L8.2's finding that fp16 breaks mT5 outright (uniform logits at −ln(26078)): a genuine
W8A16 scheme would inherit that failure, this one cannot, and the working 22.79 BLEU-4 below is the
proof. Labelled "W8A16" until 2026-09-28; renamed W8A32.

Per-row symmetric int8 (absmax/127) + fp16 scale on every mT5 2-D weight ≥ 1e5 elements: 220 tensors,
278.3 M of 285.2 M stored values (the shared embedding is counted once per copy in the state dict).
Pose stack, layer norms and relative-attention bias stay fp32. Max per-tensor relative error 4.2e-3.

| Checkpoint | file MB | BLEU-4 | ROUGE-L | Δ BLEU-4 (paired bootstrap, 1000) | sentences changed |
|---|---|---|---|---|---|
| released, full vocab (fp32 ref) | 1187 | 23.16 | 43.17 | — | — |
| pruned vocab, bf16 | 571 | 22.87 | 42.98 | −0.28 [−0.63, +0.05] vs released | — |
| **pruned + W8A32** | **293** | 22.79 | 43.00 | **−0.09 [−0.34, +0.14]** vs pruned, P=0.77; −0.37 [−0.76, −0.02] vs released | 182 / 976 |

Files: `results/eval_test_pruned_w8_mac.json`, checkpoint `weights/openasl_pose_only_slt_pruned_w8.pth`.
Clip check (Bitcoin): tokens identical to pruned fp32 in both load modes, max |logprob diff| 0.078
(`results/c5_w8_{dequant,int8}_Ads-4j06eJY.json` vs `results/c5_pruned_fp32_Ads-4j06eJY.json`).
Reading: W8 alone is within noise (CI spans zero). Stacked with pruning the total cost vs the released
model is −0.37 BLEU-4 for a 4.05× smaller file (1187 → 293 MB). 19 % of sentences change wording, so
the quantisation is not invisible per sentence, only in aggregate. Runtime modes: `--w8-runtime dequant`
(float weights in RAM, used for these numbers) and `int8` (int8 in RAM, dequantised per forward;
10× slower decoder on CPU, 2139 vs 197 ms, because 217 layers dequantise per token). Board memory and
speed for both modes come from a J-request; a fused W8A32 kernel (TensorRT INT8 weights) is the
version that saves both memory and energy.

### L7.1 Encoder input length (frame cap) vs BLEU and LM time, no retraining (pruned + W8A32, 976 clips, Mac CPU, beam 4)

`--max-length L` subsamples any clip longer than L frames uniformly to L; shorter clips are untouched.
Test clips: mean 217 frames, median 171, p95 545 (30 fps), so a cap of 256 already touches 305/976 clips.

| cap L (≈ fps) | clips subsampled | BLEU-4 | ROUGE-L | Δ BLEU-4 vs 256 (paired bootstrap) | GCN ms | encoder ms | decoder ms |
|---|---|---|---|---|---|---|---|
| 256 (30) | 305 / 976 | 22.79 | 43.00 | — | 107 | 113 | 189 |
| 205 (24) | 410 | 22.62 | 42.87 | −0.17 [−0.61, +0.26], P=0.78 | 94 | 88 | 178 |
| 137 (16) | 602 | 21.17 | 41.92 | −1.61 [−2.36, −0.89] | 77 | 59 | 162 |
| 103 (12) | 716 | 19.52 | 40.74 | −3.26 [−4.10, −2.41] | 62 | 44 | 181 |
| 68 (8) | 819 | 14.76 | 37.28 | −8.02 [−9.20, −6.86] | 49 | 35 | 134 |

Files: `results/eval_test_w8_len{205,137,103,68}_mac.json` (256 = `results/eval_test_pruned_w8_mac.json`).
Timing: Bitcoin clip (300 frames), median of 3, Mac CPU; GCN + encoder scale linearly with L
(220 → 84 ms from 256 to 68), decoder does not depend on L.
Reading: 24 fps-equivalent is free (CI spans zero); 16 fps costs 1.6 BLEU-4 without retraining; 12 and
8 fps are not usable un-adapted. The pose stage's energy is linear in frames processed, so 24 fps
is a 20 % pose-energy cut for nothing and 16 fps a 47 % cut for −1.6 BLEU-4, which is the case for the
C8 adaptation run at 16 fps (L11): if adaptation recovers ~1 BLEU, 16 fps becomes the deployment rate.
⚠️ Correction (2026-09-26): the "≈ fps" labels in this table are approximate. A cap only shortens clips
longer than L, and OpenASL is not one frame rate — measured over 400 test clips, 73 % are 30 fps, 21 %
are 24 fps, the rest 25/31/60 (frames / duration from the clip-name timestamps). True per-clip rate
emulation is now in `common/pose_to_unisign.py::fps_ratio_for_clip` (`--fps` in `eval_openasl.py` and
`train_adapt.py`); see L7.2 for the honest numbers. Likewise C8.2's 0.667 ratio emulated ~20 fps on the
30 fps clips and 16 fps on the 24 fps ones, so its 0.92 BLEU-4 drop is a mixture, not a 16 fps cost.
Caveat: this models frame-dropping by uniform subsampling of the released 30 fps poses; the pose owner's
P8 rows drop frames before extraction, which is the same input to the LM.

### C8.1 Training harness smoke test (`unisign/train_adapt.py`), Mac CPU, 300-clip train slice, no input shift

Purpose: prove the script and checkpoint format are correct, not to produce a real result — 300 clips
teach the model nothing. mT5 frozen; trainable = proj_linear + gcn_modules + fusion_gcn_modules +
part_para + pose_proj = 5.35 M / 243.6 M params. AdamW (eps 1e-9, wd 1e-4), cosine schedule, no
warm-up, grad clip 1.0, label smoothing 0.2, lr 1e-4, batch 8, 2 epochs on 300 train clips fetched via
`data/openasl_pose_fetch.py --split train --limit 300` (fetch is a Mac-side debug slice only; the full
97 K-clip / 30 GB train set for the real runs is not stored on the Mac, per PROJECT-GUIDE 4.1).

| checkpoint | BLEU-4 (200 held-out test clips, beam 4, cap 64) |
|---|---|
| pruned, before training | 14.71 |
| after epoch 0 | 14.08 |
| after epoch 1 (`log.jsonl`) | 14.49 |
| after epoch 1, reloaded via `unisign.eval_openasl` (round-trip check) | 14.55 |

`adapted_full.pth` (571 MB, bf16, same shape/dtype as the source checkpoint) round-trips through
`unisign.eval_openasl` within 0.06 BLEU-4 of the training log's own eval (batching/padding noise, not
a bug). Reading: the ±0.6 wobble across 2 epochs with no input shift is the expected noise floor for
this harness on 300 clips; a real L11 adaptation run needs the fps shift itself to show a signal.
Wall time: ~3 min/epoch train + ~6 min eval (200 clips) on Mac CPU. Files: `runs/c8_smoke/` (not
committed, in `.gitignore`), `results/c8_smoke_check.json`.

### C8.2 16 fps adaptation, Mac dry run (`unisign/train_adapt.py --fps 16`), 300-clip train slice

Same recipe and 200 held-out test clips as C8.1; every clip thinned to 16/24 of its frames
(`fps_ratio` 0.667, `--src-fps 24`) in both training and eval. Purpose: exercise the `--fps` path
end to end before Colab, not to measure adaptation (300 clips cannot).

| checkpoint | BLEU-4 | ROUGE-L | train loss |
|---|---|---|---|
| pruned, full rate (C8.1 baseline) | 14.71 | — | — |
| pruned, 16 fps, before training | 13.79 | — | — |
| 16 fps, after epoch 0 | 13.78 | 33.87 | 3.320 |
| 16 fps, after epoch 1 | 13.55 | 33.55 | 3.294 |

Reading: 16 fps costs 0.92 BLEU-4 on this subset un-adapted; 2 epochs on 300 clips recover nothing
(−0.24, inside the ±0.6 noise floor from C8.1). Expected null; the real L11 answer needs the full
train set on Colab. Open issue: `--src-fps 24` vs the 30 fps assumed in L7.1 must be settled first.
~3 min/epoch train. Files: `runs/c8_adapt16/` (not committed).

### L7.2 True per-clip frame-rate emulation vs BLEU (pruned + W8A32, 976 test clips, Mac CPU, beam 4, cap 64)

Supersedes the "≈ fps" reading of L7.1. OpenASL is **not one frame rate**: measured as frames ÷
duration (duration from the clip-name timestamps) over 400 test clips, **73 % are 30 fps, 21 % are
24 fps**, the rest 25/31/60. So neither a single `--src-fps` nor a length cap emulates a slower camera:
`common/pose_to_unisign.py::fps_ratio_for_clip` derives each clip's own rate and keeps
`round(duration × target_fps)` frames, then the 256 cap applies as before. Verified: a 30.00 fps clip
(T=152 → 81) and a 24.11 fps clip (T=131 → 87) both land on 16.0 fps.

| target fps | ratio vs source | BLEU-4 | ROUGE-L | Δ BLEU-4 vs source rate (paired bootstrap, 1000) |
|---|---|---|---|---|
| source (30/24, unthinned) | 1.0 | 22.79 | 43.00 | — |
| 24 | 0.8 / 1.0 | 22.80 | 43.22 | **+0.02 [−0.38, +0.45]**, P=0.45 |
| 16 | 0.53 / 0.66 | 21.66 | 41.56 | **−1.12 [−1.71, −0.48]**, P=1.00 |
| 12 | 0.4 / 0.5 | 20.26 | 39.15 | −2.52 [−3.22, −1.85], P=1.00 |

Files: `results/eval_test_w8_truefps{24,16,12}.json`; baseline `results/eval_test_pruned_w8_mac.json`.
Reading, and this is the headline for the frame-rate axis:
* **24 fps is free** — CI centred on zero, no retraining. On the 30 fps clips that is a 20 % cut in
  frames processed, so ~20 % off the pose stage's energy for no accuracy cost. Claimable today.
* **16 fps costs 1.12 BLEU-4 un-adapted** (not 1.6 — the L7.1 cap overstated it by only touching long
  clips, and hitting those harder). This 1.12 is the gap the C8/L11 Colab adaptation must close; if it
  recovers most of it, 16 fps (≈47 % fewer frames) becomes the deployment rate.
* **12 fps costs 2.5 BLEU-4**, still large; worth an adaptation run only if 16 fps succeeds.
Board caveat: BLEU is hardware-independent (same model, same inputs, same sentences), but the ms/W
that turn a frame-rate cut into an energy saving must come from the Jetson (J-queue), not the Mac.

### L7.3 Does the frame-rate cost depend on INT8? (interaction check, 976 test clips, Mac CPU)

L7.2's deltas were all measured on the quantized model, so the 1.12 could in principle be an artefact of
INT8 rather than of the frame rate. Repeating 16 fps on the pruned-but-**not**-quantized model and
comparing the two penalties on identical bootstrap resamples (`unisign/did_ci.py`, difference-in-
differences, 1000 resamples, N=976):

| 16 fps penalty measured on | BLEU-4 30 → 16 fps | Δ BLEU-4 [95 % CI] |
|---|---|---|
| pruned, no INT8 | 22.87 → 21.54 | **−1.33 [−2.00, −0.64]** |
| pruned + W8A32 | 22.79 → 21.66 | **−1.12 [−1.71, −0.48]** |
| interaction (difference of the two) | | **+0.21 [−0.14, +0.57]**, P(<0)=0.13 |

Files: `results/eval_test_pruned_truefps16.json`, `results/eval_test_pruned_mac.json`, plus the two L7.2
rows. Reading: the interaction CI straddles zero and lies entirely within the 0.6 sensitivity band, so
**quantization does not change the cost of dropping frame rate** — the two knobs can be quoted
separately for this pair. Caveat: this is one cell of the factorial; pruning × frame rate and decoding ×
frame rate are still unmeasured, and a CI of ±0.6 cannot rule out an interaction the size of the INT8
penalty itself. See ABLATION-REPORT.md §3.1.

### L13 Accuracy surface for the frontier plot: decoder strategy × frame rate (pruned FP32, 976 test clips, Mac CPU, cap 64)

The two levers that move both axes of the deliverable plot, measured as a full 3×3 grid so the accuracy
side is complete before the board's joules arrive. All cells: pruned FP32 (INT8 dropped from the
deployment config, see L9.1 reading and `REPLY-LM-TRACK-2026-09-28.md` §1), `max_new_tokens` 64, true
per-clip frame-rate emulation (`fps_ratio_for_clip`, L7.2). Deltas are a paired bootstrap against
beam 4 at source rate, 1000 resamples, with **one resample draw scoring every cell**, so the cells are
comparable with each other and not only with the reference (`unisign/grid_table.py`).

| decoder | fps | BLEU-4 | ROUGE-L | Δ BLEU-4 vs beam 4 @ source [95 % CI] |
|---|---|---|---|---|
| **beam 4** | source | **22.87** | 42.98 | reference |
| beam 4 | **24** | **22.80** | 43.13 | **−0.07 [−0.53, +0.39]** |
| beam 4 | 16 | 21.54 | 41.35 | −1.33 [−2.00, −0.64] |
| beam 2 | source | 22.06 | 42.22 | −0.81 [−1.29, −0.39] |
| beam 2 | 24 | 21.93 | 42.29 | −0.94 [−1.45, −0.39] |
| beam 2 | 16 | 21.19 | 40.89 | −1.68 [−2.35, −0.98] |
| greedy | source | 20.88 | 41.56 | −2.00 [−2.63, −1.41] |
| greedy | 24 | 20.66 | 41.18 | −2.22 [−2.83, −1.60] |
| greedy | 16 | 19.83 | 39.53 | −3.04 [−3.74, −2.36] |

Files: `results/eval_test_pruned_{mac,beam2_mac,greedy_mac,truefps16}.json` and
`results/eval_test_pruned_b{4,2,1}_fps{24,16}.json`. Protocol note: the three source-rate cells were run
at `max_new_tokens` 100 and the rest at 64, which is safe to mix because L3.2 measured cap 64 as
**bit-identical** to cap 100 on this model (0 of 976 sentences changed, 0 predictions at/over cap).

**Reading.**

* **24 fps is free at every decoder setting.** −0.07 at beam 4, and the increments at beam 2 and greedy
  (−0.13 and −0.22 relative to their own source-rate cells) are all inside the 0.6 noise band. The
  choice of frame rate and the choice of decoder can therefore be made independently.
* **beam 4 @ 24 fps is the best accuracy-per-frame cell**: 22.80, statistically tied with the 22.87
  reference, on ~80 % of the frames. This is the recommended operating point on accuracy grounds alone.
* **Greedy costs ~2 BLEU-4 at every frame rate** (−1.99 / −2.14 / −1.71). It is the expensive lever for
  accuracy and, per L6.1, the cheap one for compute (2.7× faster). Which end of that trade wins is
  exactly what the board's joules decide, and cannot be settled from accuracy alone.
* **The two levers are close to additive, with a hint of sub-additivity.** Naively adding the beam-4
  frame-rate cost (−1.33) to the source-rate greedy cost (−1.99) predicts 19.55 for greedy @ 16 fps; the
  measured value is **19.83, i.e. +0.28 better than additive**. The sign is intuitive — both levers
  discard information, so the second one has less left to discard — but 0.28 is inside the 0.6 band, so
  this is *no detectable interaction*, consistent with the INT8 × frame-rate result in L7.3. Treat the
  grid as additive for planning and quote the measured cell when reporting.

**What this does not settle.** Every cell is accuracy only. The frontier plot needs joules per sentence
for the same nine cells (five board runs cover it, see `REPLY-LM-TRACK-2026-09-28.md` Appendix A), and
all nine are measured on the authors' released poses, so each is an upper bound for a deployment that
uses board-extracted keypoints (pose-track gap: −6.33 BLEU-4 [+0.48, +13.11], their §5).

---

### L5.4 Vocabulary re-pruned from train+dev only (test-token leak fixed) — `unisign/vocab_census.py`

L5.1's keep set was built from a census over **train+dev+test** (98,419 sentences). A model whose
embedding matrix and LM head were *selected using the test set* has, in a real sense, seen the test
set. `unisign/vocab_census.py` rebuilds the keep set from **train+dev only** (97,443 sentences).

| | keep ids | checkpoint (bf16) |
|---|---|---|
| L5.1, train+dev+**test** census | 26,078 | 571 MB |
| **L5.4, train+dev census** | **26,025** | **570 MB** |

**53 tokens dropped, 0 added.** Each of the 53 occurs **zero** times in train+dev — they were purely
test-set tokens. They include `▁Bitcoin`, the key content word of the clip used as this project's
standard single-clip demo, which existed in the deployed vocabulary *only because it appears in the
test split*. Diff in `results/vocab_keep_diff.json`; new keep set in
`results/openasl_vocab_keep_ids_traindev.json`.

**Validation (967 dev clips, beam 4, cap 64):** the corrected checkpoint scores **23.13 BLEU-4 /
42.93 ROUGE-L**, against **23.11 / 42.90** for the leaky one. Identical within noise, exactly as
expected since none of the 53 tokens occur in dev. This confirms the re-prune broke nothing.

**Not yet measured:** the test-split score of the corrected checkpoint. That is where the 53 tokens
actually occur (55 occurrences across 46 of 976 sentences), so a small drop from 22.87 is expected and
would be the true, unleaked number. **Every test-split figure elsewhere in this file still comes from
the leaky keep set** and should be re-stated once that eval is run.

---

### L15 Adaptation harness: recipe diagnosis, and adapted vs un-adapted at 16 fps

All runs: Colab Tesla T4, FP32 (no AMP — T4 is Turing, no bf16, and fp16 would push mT5 activations
toward the overflow documented in L8.2), pose stack + `pose_proj` + `part_para` trainable (5.35 M of
243.5 M), mT5 frozen, 20,000 train clips, 1 epoch, batch 4 × accum 2, `--freeze-bn`, seed 42,
starting from the L5.4 checkpoint. Evaluated on **967 dev clips**, beam 4, cap 64. Logs in
`results/colab_runs/`.

#### L15.1 The first recipe made the model worse, and the control proved it was the recipe

Initial runs used the authors' `--label-smoothing 0.2` with `--warmup-epochs 0` (300 dev clips):

| run | train/eval rate | before | after | Δ BLEU-4 |
|---|---|---|---|---|
| `fps16_seed42` | 16 fps | 18.27 | 16.57 | **−1.70** |
| `nofps_seed42` | **source (control)** | 17.17 | 16.13 | **−1.04** |

The **control also degraded**. With no distribution shift to adapt to, fine-tuning still cost 1.04
BLEU-4 — so the 16 fps number was never evidence about adaptation, only about a broken recipe. Running
the no-shift control is what separated the two, and it should precede any adaptation claim.

Note also that the two *un-adapted* baselines differ by frame rate alone on the same 300 clips:
16 fps scores **+1.10 higher** than source rate, the opposite sign from L7.2's n=976 result. At n=300
BLEU-4 cannot resolve effects of this size — it flipped the sign of a known effect. All later runs use
the full 967-clip dev split.

#### L15.2 Recipe sweep (control condition, no frame-rate shift, 967 dev clips)

Un-adapted baseline: **23.13 BLEU-4 / 42.93 ROUGE-L**.

| recipe | BLEU-4 | ROUGE-L | Δ BLEU-4 | Δ ROUGE-L |
|---|---|---|---|---|
| ls 0.2, lr 1e-4, no warmup | — | — | −1.04 *(at n=300)* | not measured |
| ls 0.0, lr 1e-4, warmup 0.1 | 22.75 | 43.51 | −0.38 | +0.58 |
| **ls 0.0, lr 1e-5, warmup 0.1** | **23.31** | **43.63** | **+0.18** | **+0.70** |

**Label smoothing was the problem.** With `--label-smoothing 0.2` over a 26 K vocabulary the reported
training loss sits at ~3.35 and barely moves; at 0.0 the true cross-entropy is ~0.36. The 3.35 was
mostly the smoothing floor, not the model's fit — a constant being read as a training curve.

`lr 1e-5` is the working recipe: fine-tuning now **improves** the model on both metrics. This is the
first evidence the C8 harness trains rather than merely runs (L4.2/C8.1 only established that the
script executes and checkpoints round-trip).

**Caveat:** the intended isolating control (ls 0.2 at lr 1e-5, which would separate smoothing from
learning rate) was killed by a Colab disconnect after 10 log lines and has **not** been run. So
"label smoothing was the problem" is the best available reading, not an isolated result — the working
recipe changed smoothing, learning rate and warmup together.

#### L15.3 Adapted vs un-adapted at 16 fps (967 dev clips, ls 0.0, lr 1e-5, warmup 0.1, seed 42)

| | un-adapted | adapted | Δ BLEU-4 | Δ ROUGE-L |
|---|---|---|---|---|
| source rate | 23.13 / 42.93 | 23.31 / 43.63 | +0.18 | +0.70 |
| **16 fps** | 22.79 / 41.59 | **23.19 / 42.60** | **+0.40** | **+1.01** |

**Reading.**

* **Adaptation helps more under frame-rate shift than without it.** Difference-in-differences:
  **+0.22 BLEU-4, +0.31 ROUGE-L** — the part attributable to frame-rate adaptation rather than to
  fine-tuning in general.
* **It recovers most of the frame-rate loss.** Un-adapted, 16 fps costs **−1.34 ROUGE-L** (42.93 →
  41.59). Adapted, the 16 fps model sits **−0.33 ROUGE-L** below the un-adapted full-rate baseline —
  roughly **75 % of the loss recovered** — and **+0.06 BLEU-4 above** it.
* **This moves the frontier point.** `unisign/frontier.py` puts 16 fps / beam 4 at 27.0 J against
  43.4 J for the reference, a **38 % system energy saving**. Un-adapted that carries a real accuracy
  cost; adapted, the cost nearly vanishes.
* The un-adapted 16 fps figure (22.79 / 41.59) independently reproduces the Mac result (22.77 / 41.60)
  on different hardware with a different checkpoint.

**What is NOT established.**

1. **No confidence intervals.** `train_adapt.py`'s built-in eval logs summary metrics only, not the
   967 predictions, so none of these deltas can be bootstrapped. Every other comparison in this file
   carries a paired CI; these do not. `eval_openasl.py` must be re-run against the adapted checkpoints
   to produce eval JSONs before any of this is quotable.
2. **One seed.** Guide row 4.3 (L10 seed variance) has not been run. "+0.40 vs +0.18" is
   uninterpretable without the seed-to-seed spread, and that is a different noise source from the
   clip-resampling CI.
3. 20,000 of 96,477 train clips, one epoch.
4. Frame-rate emulation thins already-extracted 30 fps poses; a real 16 fps capture differs in
   exposure and motion blur.

Checkpoint: `weights/adapt_fps16_lr1e5_seed42.pt` (trainable params + optimizer state; the full
570 MB model reconstructs from it plus the L5.4 base). GPU reductions are non-deterministic, so a
re-run produces a similar but not identical model — this file is the one that produced the numbers
above.
