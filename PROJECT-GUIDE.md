# Project guide — sign-language video → text on Jetson Orin Nano 8 GB

Step-by-step plan for CS 6301 (Low-Power CV). Part 1 lists every step already done, each with the
file that holds its evidence. Part 2 lists every pending step in the order we will execute it, with
owner, commands, the file it must produce, and the "done when" check. Update the status column here
and in `WORKSPLIT.md` when a step lands; put every number in `results/RESULTS.md`.

Pipeline: video frames → **RTMPose/RTMW wholebody keypoints (133 × 2)** → Uni-Sign ST-GCN pose
encoder → **mT5-base** decoder → English sentence. Dataset: OpenASL (976 test clips). Metric:
BLEU-4 + ROUGE-L with the authors' scorer. Hardware: Jetson Orin Nano 8 GB, 15 W mode, inside the
course Docker container `lpcv-work`.

Owners: **Atisri = language-model (LM) track** (everything after the keypoints, all training and
export). **Teammate = pose track + every run on the Jetson.** Common items name who does them.

Hard rules (from `WORKSPLIT.md` §4):
- No sudo on the Jetson, ever. 15 W only. Clocks logged, never pinned. Power budgets are measured
  averages, not hardware caps.
- Never pip-install torch on the board. Never `docker run --rm` the `lpcv-work` container.
- Every hardware number in `results/RESULTS.md` comes from a real run on the board. No FLOP estimates.
- The released Uni-Sign checkpoint is the FP32 reference rung; nothing overwrites it.
- Accuracy deltas under ~1 BLEU carry a bootstrap confidence interval (`unisign/bootstrap_ci.py`).

---

## Part 1 — Done (with result files)

### 1.1 Environments

| Step | What | Evidence |
|---|---|---|
| E1 | Jetson container `lpcv-work` (image `lpcv:fall2026`): Python 3.12.3, torch 2.8.0a0 (NVIDIA build), TensorRT 10.11.0.33, CUDA 12.9, L4T R36.4.7. `opencv-python-headless` + `numpy<2` pip-installed in the container. No tegrastats, no nvidia-smi, no onnxruntime. Power mode read from `/var/lib/nvpmodel/status` = 15 W (mode 0) | [SESSION-LOG.md](SESSION-LOG.md) §0, §6 |
| E2 | Mac conda env `mmpose` (py3.10, torch 2.1.2, mmcv 2.2.0, mmpose 1.3.2, mmdet, transformers 4.44.2, rtmlib, yt-dlp, rouge, portalocker). Used for pose export, the standalone LM eval, and all BLEU numbers below. Build fixes: `setuptools<81`, `--no-build-isolation`, `numpy<2`, mmdet's mmcv ceiling relaxed | [SESSION-LOG.md](SESSION-LOG.md) §1 |
| E3 | Mac conda env `unisign` (py3.11, torch 2.14, transformers 4.57.6, onnx 1.22, onnxruntime 1.30). Used only for the mT5 ONNX export | [SESSION-LOG.md](SESSION-LOG.md) Session 2 |
| E4 | Colab (T4) with the authors' Uni-Sign repo, deepspeed, decord; Drive folder `MyDrive/unisign` holds the checkpoint, mT5 and the test poses | [TRACK-B-GUIDE.md](TRACK-B-GUIDE.md) |

### 1.2 Pose stage — FP32 baseline on the Jetson (Phase 1)

| Step | What | Evidence |
|---|---|---|
| A1 | Test clip: OpenASL test clip `Ads-4j06eJY 00:07:37–00:07:47` (Bitcoin), 299 frames at 30 fps, cropped to the signer box 666×720 | [data/test_frames/](data/test_frames/) (299 jpg), [data/test_frames_meta.json](data/test_frames_meta.json) |
| A2 | RTMPose-x wholebody 384×288 weights + config, exported to ONNX ([task1_rtmpose/01_export_onnx.py](task1_rtmpose/01_export_onnx.py)), preprocess constants recorded | [models/rtmpose-x.onnx](models/rtmpose-x.onnx), [models/preproc.json](models/preproc.json) |
| A3 | PyTorch reference outputs for 20 frames ([task1_rtmpose/05_make_reference.py](task1_rtmpose/05_make_reference.py)); standalone preprocess = mmpose pipeline (diff 0.0); ONNX Runtime = PyTorch (simcc diff 8e-6) | [results/rtmpose_reference.npz](results/rtmpose_reference.npz); table in [results/RESULTS.md](results/RESULTS.md) "Correctness ladder" |
| A4 | FP32 TensorRT engine on the board ([task1_rtmpose/02_build_engine.py](task1_rtmpose/02_build_engine.py), 158 s, 231 MB). Correctness gate ([task1_rtmpose/06_compare_trt.py](task1_rtmpose/06_compare_trt.py)): with TF32 (TensorRT default) FAIL at 1.55e-3; with TF32 cleared **PASS** at 7.2e-6, 2416/2416 confident keypoints at 0.0 px | [results/RESULTS.md](results/RESULTS.md) "Correctness ladder"; [common/trt_runner.py](common/trt_runner.py) clears `BuilderFlag.TF32` |
| A5 | FP32 latency, 3 × 299 frames, batch 1 ([task1_rtmpose/03_infer_frames.py](task1_rtmpose/03_infer_frames.py)): pre 10.6 / TRT 44.4 / post 1.5 / **total 56.4 ms** (~18 fps, below the 30 fps source) | [results/RESULTS.md](results/RESULTS.md) "Task 1" row; raw `results/rtmpose_trt_fp32.json` **still on the Jetson** (queue J1) |
| A6 | FP32 power at 15 W via INA3221 sysfs inside the container ([task1_rtmpose/04_infer_power.py](task1_rtmpose/04_infer_power.py), [common/power_logger.py](common/power_logger.py) sysfs backend): 11.62 W avg, 3.73 W idle, 13.33 W peak, **754 mJ/frame** (512 mJ dynamic), GPU 68 %, Tj 56 °C. One 10 s sentence ≈ 225 J of pose extraction | [results/RESULTS.md](results/RESULTS.md) "Task 1" row; raw `results/rtmpose_fp32_15W_power.json/.csv` **still on the Jetson** (queue J1) |
| A7 | Harness scripts verified on the board without root: [common/power_logger.py](common/power_logger.py) (sysfs backend), [common/trt_runner.py](common/trt_runner.py), `task1_rtmpose/` scripts 02–06 | [SESSION-LOG.md](SESSION-LOG.md) §5, §6 |

### 1.3 Language-model stage — off-board (Track B / LM track)

| Step | What | Evidence |
|---|---|---|
| B1 | Metric sanity on the authors' released predictions: their files carry a `sample: …, prediction: ` prefix; stripped, the scorer gives **22.64 BLEU-4 / 42.83 ROUGE-L** (raw gives 58.7, wrong) | [results/RESULTS.md](results/RESULTS.md) "Metric check (B1.3)" |
| B2 | Reproduced the paper on Colab T4 with the authors' repo and released checkpoint: **22.53 / 42.68** (paper 22.67 / 42.77) | [results/RESULTS.md](results/RESULTS.md) "B1.5 Reproduction" |
| B3 | Pose data: [data/openasl_pose_fetch.py](data/openasl_pose_fetch.py) reads single files out of the 32 GB HF archive by HTTP range (no full download). All 976 test-split pose pkls fetched (555 MB) | [data/openasl_test_pose/](data/openasl_test_pose/) (976 pkl), [data/openasl_ref_pose/](data/openasl_ref_pose/) (Bitcoin clip), labels in [data/openasl_labels/](data/openasl_labels/) |
| B4 | **C2 – which extractor made the released poses.** Compared the authors' pkl for the Bitcoin clip against the Jetson RTMPose-x output and two rtmlib runs ([task1_rtmpose/08_compare_extractors.py](task1_rtmpose/08_compare_extractors.py)). Answer: a **256×192 RTMW/RTMPose model** (archive folder `pose-rtmpose-192`). Consequence: Jetson baseline pose model = RTMW-l-m 256×192; RTMPose-x 384×288 is the "oversized" comparison row | [results/extractor_comparison.json](results/extractor_comparison.json), [results/rtmlib_lightweight_Ads-4j06eJY.pkl](results/rtmlib_lightweight_Ads-4j06eJY.pkl), [results/rtmlib_performance_Ads-4j06eJY.pkl](results/rtmlib_performance_Ads-4j06eJY.pkl); [results/RESULTS.md](results/RESULTS.md) "C2" |
| B5 | **C3 – converter** [common/pose_to_unisign.py](common/pose_to_unisign.py): keypoints JSON or pkl → Uni-Sign tensors (body / left / right / face groups, root-normalised, threshold 0.3, `crop_scale`), plus `collate`. Bit-exact against the authors' `load_part_kp` / `crop_scale` on two pose files | [common/pose_to_unisign.py](common/pose_to_unisign.py); [results/RESULTS.md](results/RESULTS.md) "C5" notes |
| B6 | **C5 – standalone model** [unisign/model.py](unisign/model.py) + [unisign/unisign_infer.py](unisign/unisign_infer.py): pose-only Uni-Sign without mmpose / deepspeed / decord, strict checkpoint load, per-stage timing, greedy or beam. First translations of the Bitcoin clip from the authors' pose and from both rtmlib poses | [results/c5_authors_pose_Ads-4j06eJY.json](results/c5_authors_pose_Ads-4j06eJY.json), [results/c5_rtmlib_lightweight_Ads-4j06eJY.json](results/c5_rtmlib_lightweight_Ads-4j06eJY.json), [results/c5_rtmlib_performance_Ads-4j06eJY.json](results/c5_rtmlib_performance_Ads-4j06eJY.json); [results/RESULTS.md](results/RESULTS.md) "C5" |
| B7 | **C7 – standalone eval loop** [unisign/eval_openasl.py](unisign/eval_openasl.py) on all 976 test clips, Mac CPU, beam 4, 16 min: **23.16 / 43.17** (authors' framework 22.53; noise band ≈ 0.6 BLEU-4 from the subsampling difference) | [results/eval_test_released_mac.json](results/eval_test_released_mac.json) (refs + preds + metrics); [results/eval_smoke16.json](results/eval_smoke16.json) (16-clip smoke) |
| B8 | **L5 – vocabulary pruning.** Token census over OpenASL train+dev+test (98,419 sentences): 26,078 of 250,112 mT5 tokens kept. `shared` embedding and `lm_head` sliced, tokenizer remapped. **587.7 M → 243.6 M params, 1187 → 571 MB** checkpoint (bf16). Full-split accuracy 22.87 BLEU-4, **Δ −0.28 [−0.63, +0.05]** paired bootstrap | [results/openasl_token_counts.json](results/openasl_token_counts.json), [results/openasl_vocab_keep_ids.json](results/openasl_vocab_keep_ids.json), [results/eval_test_pruned_mac.json](results/eval_test_pruned_mac.json); weights [weights/openasl_pose_only_slt_pruned.pth](weights/openasl_pose_only_slt_pruned.pth), HF dir [weights/mt5-base-openasl-pruned/](weights/mt5-base-openasl-pruned/); [results/RESULTS.md](results/RESULTS.md) "L5.1–L5.3" |
| B9 | **L12 – bootstrap CIs** [unisign/bootstrap_ci.py](unisign/bootstrap_ci.py) (paired, 1000 resamples, prints CI and P(B<A)); used for every delta above and below | script; numbers in [results/RESULTS.md](results/RESULTS.md) |
| B10 | **L6.1 – decode knobs** on the pruned model, 976 clips: greedy **20.88, Δ −2.00 [−2.63, −1.41]** at 2.7× less time; beam 2 **22.06, Δ −0.81 [−1.29, −0.39]** at 1.7×; beam 4 = reference | [results/eval_test_pruned_greedy_mac.json](results/eval_test_pruned_greedy_mac.json), [results/eval_test_pruned_beam2_mac.json](results/eval_test_pruned_beam2_mac.json); [results/RESULTS.md](results/RESULTS.md) "L6.1" |
| B11 | **L3 – LM per-sentence timing**, Mac CPU (indicative): decoder is 60–80 % of LM time; lever order beam width > pruning > encoder. Board numbers come from J2 | [results/l3_timing_mac_cpu.json](results/l3_timing_mac_cpu.json); [results/RESULTS.md](results/RESULTS.md) "L3" |
| B12 | **L8 export half – mT5 (pruned) → ONNX with explicit KV cache** ([task3_mt5_onnx/02_export_manual.py](task3_mt5_onnx/02_export_manual.py) with `--verify`): encoder / decoder_init / decoder_step graphs. Synthetic verify at 3 encoder lengths, 12 steps: tokens identical, logits ≤ 2.2e-5. Real-pose verify with [unisign/onnx_decode.py](unisign/onnx_decode.py): tokens identical to PyTorch, logprob diff 2.9e-5, ORT CPU 7.4 ms/token | [models/mt5_pruned_onnx/encoder.onnx](models/mt5_pruned_onnx/encoder.onnx), [models/mt5_pruned_onnx/decoder_init.onnx](models/mt5_pruned_onnx/decoder_init.onnx), [models/mt5_pruned_onnx/decoder_step.onnx](models/mt5_pruned_onnx/decoder_step.onnx) (1.5 GB, not in git); [results/RESULTS.md](results/RESULTS.md) "L8.1" |
| B13 | TensorRT decode loop for the board written: [unisign/trt_decode.py](unisign/trt_decode.py) (syntax-checked, **not yet run**; queue J4) | script |
| B14 | Planning docs: [WORKSPLIT.md](WORKSPLIT.md) v2 (ownership, C/P/L task tables, Jetson request queue §5), [TRACK-B-GUIDE.md](TRACK-B-GUIDE.md), [SESSION-LOG.md](SESSION-LOG.md) (Session 1 + 2), [results/RESULTS.md](results/RESULTS.md) | files |

---

## Part 2 — Pending, in execution order

Legend: **who** = Atisri (LM) / Teammate (pose + board) / both. Each step ends with the file it must
produce and the check that marks it done. Steps inside a block are sequential; blocks marked ∥ can run
in parallel with the previous block.

### Block 0 — housekeeping (today)

| # | Who | Step | Produces / done when |
|---|---|---|---|
| 0.1 ✅ | Atisri | **C0 Git.** (commit 375ddfc, `.gitignore` in place; remote not yet added) `git init`, `.gitignore` with `weights/ models/*.onnx models/*.engine models/mt5_pruned_onnx/ data/raw data/test_frames data/calib_frames data/openasl_test_pose results/*.csv results/*.npz results/*.pkl venv/`, first commit, private GitHub remote, branches `lm/…` and `pose/…`. Large files stay on the shared Drive mirroring repo paths | repo pushed; teammate can clone |
| 0.2 | Teammate | **P0 onboarding.** Read `SESSION-LOG.md`, this guide, `WORKSPLIT.md`. Confirm container access (`docker exec -it lpcv-work bash`, cwd `/workspace`). Check `ls /workspace/{common,task1_rtmpose,models,results}` still exists (folders vanished once before; re-`scp` from the Mac if so) | can run `python3 task1_rtmpose/03_infer_frames.py --help` in the container |

### Block 1 — Jetson request queue (teammate, copy-paste; `WORKSPLIT.md` §5)

| # | Who | Step | Produces / done when |
|---|---|---|---|
| 1.1 | Teammate | **J1 = C1.** Copy the Phase-1 raw outputs off the board: on the Jetson host `scp ~/atisri-cv-jetson/results/rtmpose_* <mac>:~/atisri-cv-jetson/results/` (or commit them) | `results/rtmpose_trt_fp32.json`, `results/rtmpose_fp32_15W_power.json/.csv` on the Mac |
| 1.2 | Atisri | **C4 Jetson half.** Run `python -m unisign.unisign_infer --keypoints results/rtmpose_trt_fp32.json --crop-wh 666 720 --ckpt weights/openasl_pose_only_slt.pth --mt5 weights/mt5-base --out results/c5_jetson_rtmposex_Ads-4j06eJY.json`; add the RTMPose-x row to the C2 table in `results/RESULTS.md` | sentence from Jetson keypoints next to the authors'-pkl sentence |
| 1.3 | Teammate | **J2 = L4, first on-board LM run.** Copy `unisign/ common/ data/openasl_ref_pose/ weights/mt5-base weights/openasl_pose_only_slt*.pth` to the board. In the container: `pip3 install "transformers>=4.45,<5" sentencepiece`. Then `python3 -m unisign.unisign_infer --pkl data/openasl_ref_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl --ckpt weights/openasl_pose_only_slt.pth --mt5 weights/mt5-base --device cuda --repeat 3 --out results/l4_jetson_full_fp32.json`, and the same with `_pruned.pth` → `results/l4_jetson_pruned_fp32.json` | two JSONs + printed `[infer]` lines (GCN / encoder / decoder ms, text, peak GPU MB, no OOM). Same sentence as on the Mac |
| 1.4 | Teammate | **J4 = L8 board half.** Copy `models/mt5_pruned_onnx/` (1.5 GB). One engine per process: `python3 task3_mt5_onnx/03_build_engines.py --onnx-dir models/mt5_pruned_onnx --which encoder`, then `--which decoder_init`, then `--which decoder_step --enc-len 1,264,512 --dec-len 1,1,128`. Then `python3 -m unisign.trt_decode --engine-dir models/mt5_pruned_onnx --ckpt weights/openasl_pose_only_slt_pruned.pth --mt5 weights/mt5-base --pkl data/openasl_ref_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl --repeat 3 --out results/l8_jetson_trt_fp32.json`; repeat with `--fp16` → `results/l8_jetson_trt_fp16.json` | `[check] tokens identical: True` for FP32; for FP16 report first divergence + logprob diff. Per-token ms vs the PyTorch line from J2 |
| 1.5 ✅ | Atisri | Add `--power-json/--power-csv` (the `04_infer_power.py` wrapper) to `unisign/unisign_infer.py` and `unisign/trt_decode.py` | scripts accept the flags on the Mac (power backend `none` off-board) |
| 1.6 | Teammate | **J3.** Re-run 1.3 and 1.4 wrapped in the power logger, 3 repeats each | `results/l4_jetson_*_power.json`, `results/l8_jetson_*_power.json`: W avg / idle, J per sentence |
| 1.7 | Atisri | Record L4, L8.2, L6 board rows in `results/RESULTS.md`; mark L4 ✅, L8 ✅ in `WORKSPLIT.md` | rows present |

### Block 2 ∥ — Pose ladder (teammate)

| # | Who | Step | Produces / done when |
|---|---|---|---|
| 2.1 | Teammate | **P1 baseline pose model = RTMW-l-m 256×192** (the C2 answer). Get the ONNX (rtmlib ships it; `rtmlib.Wholebody(mode="lightweight")` downloads it) or export from MMPose. Make a PyTorch/ORT reference on 20 frames (`05_make_reference.py`-style, new `preproc.json` for 256×192), build the FP32 engine (TF32 off), `06_compare_trt.py` PASS, latency (`03_infer_frames.py`) and power (`04_infer_power.py --repeat 3`) rows | `models/rtmw-l-m_256x192.onnx`, `results/rtmw_reference.npz`, `results/rtmw_fp32_15W_power.json`; row in `results/RESULTS.md` |
| 2.2 | Teammate | **P2 FP16 engines**: `02_build_engine.py --fp16` for RTMW-l-m and RTMPose-x; compare with `--simcc-atol 0.05 --kpt-atol-px 2`; power rows | `results/*_fp16_15W_power.json`; rows |
| 2.3 | Teammate | **P3 data**: `data/openasl_fetch.py` — download ≥5 test clips from distinct signers (yt-dlp section download from `SESSION-LOG.md` §3; many videos are private, iterate) and ≥300 train frames from ≥5 signers into `data/calib_frames/`; crop to the signer box | `data/clips/<clip>/frames/`, `data/calib_frames/`, meta JSON |
| 2.4 | Teammate | **P4 keypoint-agreement metric** `task1_rtmpose/07_kpt_agreement.py`: % of confident keypoints within 1 / 2 / 5 px of the FP32 engine, per group, over all frames. This is the pose accuracy axis when BLEU is unavailable | script + `results/kpt_agreement_<config>.json` |
| 2.5 | Teammate | **C7 pose→BLEU path**: run each pose engine over the P3 test clips' frames → `common/pose_to_unisign.py` → pkl → `unisign/eval_openasl.py --poses <dir>` (Mac or Colab). Gives BLEU per pose config on the available clips | `results/eval_<pose_config>.json` |
| 2.6 | Teammate | **P5 INT8 PTQ**: entropy calibrator in `common/trt_runner.py` fed from `data/calib_frames/`; INT8 engine, compare, agreement (2.4), BLEU (2.5), latency + power | `results/*_int8_15W_power.json`; row |
| 2.7 | Teammate | **P6 variant sweep**: RTMW-x-l 384×288, RTMPose-x / -l / -m wholebody. Each: FP32 correctness vs own reference, FP16, rows. **P7** resolution row (384×288 vs 256×192 where checkpoints exist) | rows |
| 2.8 | Teammate | **P11 preprocess on GPU**: move affine + normalize to torch on the GPU; re-measure the 19 % CPU preprocess share | row with pre ms separated |
| 2.9 | Teammate | **P8 temporal subsampling**: engine at 24 / 16 / 12 / 8 fps input (drop frames before extraction); latency + energy per sentence. Accuracy per rate needs the C8 adaptation (Block 4) | rows; feeds L7 |
| 2.10 | Teammate | **P12 segmentation heuristic** for the live demo (hands-return-to-rest on pose velocity) | script + demo clip |

### Block 3 ∥ — LM ladder (Atisri, off-board first, board rows via Block 1 style requests)

| # | Who | Step | Produces / done when |
|---|---|---|---|
| 3.1 ✅ | Atisri | **L3 remainder** (cap 64 = identical; cap 48 −0.14 [−0.30,−0.03]; RESULTS.md L3.2): `max_new_tokens` sweep 100 → 64 → 48 on the 976 clips (pruned, beam 4) with CIs | `results/eval_test_pruned_mnt{64,48}_mac.json`; row |
| 3.2 🔄 | Atisri | **L9 weight-only INT8 (W8A16)** (Mac half done: 293 MB, −0.09 [−0.34,+0.14] vs pruned, RESULTS.md L9.1; board rows pending) on the pruned mT5, in whichever runtime survived L8 (TensorRT engines with INT8 weights, or PyTorch dynamic quant as fallback). BLEU on 976 clips with CI, checkpoint MB, then board memory + power via a J-request | `weights/…_w8a16.*`, `results/eval_test_w8a16_mac.json`; row |
| 3.3 | Atisri | **L6 remainder**: INT8 KV cache, and the board rows for greedy / beam 2 / beam 4 (from J3 numbers) | rows: BLEU vs J per sentence |
| 3.4 ✅ | Atisri | **L7 encoder length** (24 fps free, 16 fps −1.6, 12 fps −3.3, 8 fps −8.0 BLEU-4 un-adapted; RESULTS.md L7.1): encoder ms / energy vs T at 256 / 205 / 137 / 103 / 68 frames (30 / 24 / 16 / 12 / 8 fps-equivalent). Off-board timing first, board via request | `results/l7_encoder_length.json`; row |
| 3.5 | Atisri | **L13 LLM correction stage** (last, cut without regret): confidence-gated on token logprobs from `unisign_infer`, minimal-edit prompt, three-condition ablation (none / single sentence / N prior turns), edit distance + BLEU + tokens-in vs joules. Text-only API | `unisign/llm_correct.py`, `results/l13_*.json` |

### Block 4 — Training harness and adaptation (Atisri builds, both use)

| # | Who | Step | Produces / done when |
|---|---|---|---|
| 4.1 🔄 | Atisri | **Train poses**: `python data/openasl_pose_fetch.py --split train --labels-dir data/openasl_labels --out data/openasl_train_pose` (~97 K files, ~30 GB; resumable). Store on the 4×24 GB rig or Drive, not the Mac. 300-clip Mac-side debug slice (`--limit 300 --out data/openasl_train_pose_smoke`, 239 MB) done for 4.2 only, not counted toward this | `data/openasl_train_pose/` complete count = 96,477 |
| 4.2 ✅ | Atisri | **C8 harness** [unisign/train_adapt.py](unisign/train_adapt.py): mT5 frozen, pose stack + `pose_proj` + `part_para` trainable (5.35 M / 243.6 M params), AdamW, cosine schedule, checkpoint-resume (`--resume`). Smoke: 2 epochs on the 300-clip Mac slice, no input shift → BLEU-4 wobbles 14.71 → 14.08 → 14.49 on 200 held-out test clips (noise, no real signal expected from 300 clips); saved checkpoint round-trips through `unisign.eval_openasl` within 0.06 BLEU-4. Confirms the script and checkpoint format are correct; real signal needs Colab-scale runs (4.3/4.4) | script + [results/RESULTS.md](results/RESULTS.md) "C8.1" |
| 4.3 | Atisri | **L10 seed variance**: 4 seeds in parallel on the 4×24 rig, same config → BLEU spread = the noise band every later delta is judged against | `results/l10_seeds.json`; row with mean ± std |
| 4.4 | Atisri (runs triggered by teammate) | **L11 / P8 / P9 adaptation runs**: one short pass per input shift — frame rates 24 / 16 / 12 / 8 fps (P8, pairs with L7), face group dropped (P9: 3C feature width), pose-model swap if a variant is chosen (P6). Report BLEU adapted vs un-adapted with CI | `results/eval_adapt_<shift>.json`; rows |
| 4.5 | Teammate | **P10 PTQ vs QAT** on the pose front-end using the C8-style loop on the rig | row |
| 4.6 | Atisri | **L14 stretch** (only if everything above is done): context-conditioned encoder with prior-turn text embeddings | probably a null result; row |

### Block 5 — Rigor, frontier, report (both)

| # | Who | Step | Produces / done when |
|---|---|---|---|
| 5.1 | Teammate | **C6 = Milestone M1, first on-device translation**: pose engine → `common/pose_to_unisign.py` → LM on the Jetson for the Bitcoin clip, end to end, with per-stage latency and power. This is the Stage-0 baseline every optimization is compared to | `results/m1_pipeline_fp32.json`; text + total ms + J per sentence |
| 5.2 | Teammate | **C9 results protocol**: every final row = 3 runs mean ± std, DVFS state logged (sysfs clock paths from `SESSION-LOG.md` §8), ≥3 clips / ≥3 signers, one 30-min sustained run for FP32 and for the best compressed config (thermal / throttling check) | `results/sustained_*.csv`; final table in `results/RESULTS.md` with fixed columns |
| 5.3 | both | **Week-4 gate**: M1 + one compressed row on each side with measured energy. If missing, cut L13 and P14 and ship the Jetson-only study | decision noted in `SESSION-LOG.md` |
| 5.4 | Atisri | **C10 = Milestone M4, accuracy–energy frontier**: J per sentence (pose + LM, measured) vs BLEU-4 with CIs across all configs; `results/plot_frontier.py` | `results/frontier.png`, `results/frontier.csv` |
| 5.5 | both | **C11 report + demo video** (record the demo early), written against both rubrics: baseline clarity, deployment validation at 15 W, empirical rigor (3 runs, CIs, sustained), compression justification (accuracy axis for every row) | report + video |
| 5.6 | Teammate | **P14 iOS capture app** streaming frames to the Jetson (optional; cut at the week-4 gate if behind) | app + MetricKit log |

---

## Quick reference

- Scoring anything: `python -m unisign.eval_openasl --ckpt <pth> --mt5 weights/mt5-base --poses <pkl dir> --labels data/openasl_labels/labels.test --num-beams 4 --max-new-tokens 100 --out results/<name>.json` (Mac `mmpose` env, ~16 min CPU for 976 clips).
- Delta with CI: `python -m unisign.bootstrap_ci results/<A>.json results/<B>.json -n 1000`.
- One clip through the LM: `python -m unisign.unisign_infer --pkl <pkl> | --keypoints <json> --crop-wh W H --ckpt <pth> --mt5 weights/mt5-base --out results/<name>.json`.
- Keypoints JSON → Uni-Sign pkl: `common/pose_to_unisign.py` (`load_keypoints_json`, `save_pkl`).
- Pose engine on the board: build `02_build_engine.py` → check `06_compare_trt.py` → latency `03_infer_frames.py` → power `04_infer_power.py --repeat 3 --power-json … --power-csv …`.
- Jetson container: `docker exec -it lpcv-work bash`, work in `/workspace`; power via sysfs INA3221 (no root needed).
