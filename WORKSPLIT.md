# Work split v2 — pose owner vs language-model owner

Ownership by model stage, not by hardware:

- **Pose track (teammate):** RTMPose/RTMW extraction, its TensorRT engines, every pose-stage
  optimization, all Jetson power/latency runs for the pose stage. Needs the Jetson and the Mac
  `mmpose` env.
- **LM track (Atisri):** everything after the keypoints: GCN encoders + mT5, accuracy evaluation,
  every language-model optimization, the ONNX/TensorRT export of mT5, training. Needs Colab / the
  4x24 GB rig. Jetson only to run the finished LM artifacts, which the pose owner can do on request.
- **Common:** anything one track needs from the other, or that both do together. Listed in the
  order it must happen. Each common item names who does it.

Context files: `sign-language-project-context-v2.md`, `jetson-setup-handoff.md`, `SESSION-LOG.md`,
`TRACK-B-GUIDE.md`, `results/RESULTS.md`. Status markers: ✅ done, 🔄 in progress, ⬜ not started.

---

## 0. Common — setup and interfaces (do first, in this order)

| # | Task | Who | Status | Unblocks |
|---|---|---|---|---|
| C0 | Put the repo in git (private GitHub), `.gitignore` for `weights/ models/*.onnx models/*.engine data/raw data/test_frames data/calib_frames results/*.csv results/*.npz venv/`. Large files on a shared Drive folder mirroring repo paths. Branches `pose/...` and `lm/...`, PRs to `main` | Atisri | ⬜ | everything |
| C1 | Copy Jetson results to the Mac and commit `results/rtmpose_trt_fp32.json` (the keypoints JSON for the Bitcoin clip). Schema: per frame `{"frame", "keypoints": 133x2 px in the cropped frame, "scores": 133}` plus `summary` | Atisri (has the board today), then teammate | ⬜ | C2 |
| C2 | **Which extractor made the released poses.** Run `task1_rtmpose/08_compare_extractors.py` with the authors' pkl for the Bitcoin clip (download from Drive `unisign/openasl_test_pose/`), the Jetson RTMPose-x JSON, and the two rtmlib pkls already in `results/`. Write the answer in `RESULTS.md`. Decides the Jetson baseline pose model (P1) | Atisri runs it, both read the result | ✅ answer: 256x192 RTMW-l-m (see RESULTS.md); Jetson RTMPose-x row added when C1 lands | P1, P4, L3 |
| C3 | `common/pose_to_unisign.py`: keypoints JSON → Uni-Sign input tensors (body/left/right/face groups, root-normalised, thr 0.3), lifted from `datasets.py: load_part_kp / crop_scale`. Unit test: our JSON → pkl → their loader gives the same tensors as their pkl. Also writes the authors' pkl format so any keypoints file can be scored | Atisri | ✅ bit-exact vs their loader on 2 pose files | C4, L2, P-everything that needs BLEU |
| C4 | Round-trip on Colab: Jetson keypoints (C3) through the released checkpoint → sentence for the Bitcoin clip, next to the sentence from the authors' pkl for the same clip | Atisri | 🔄 authors'-pkl half done (Bitcoin sentence, `results/c5_*.json`, RESULTS.md C5); Jetson-keypoints half waits on J1 | C5 |
| C5 | `unisign/unisign_infer.py`: plain PyTorch, no mmpose/mmcv/mmdet/deepspeed/decord, `--keypoints --ckpt --mt5 --out`, greedy, `max_new_tokens` cap, per-stage timing, output `{"text","tokens","token_logprobs","timing_ms":{gcn,encoder,decoder}}`. Must import in the Jetson container (Python 3.12, torch 2.8; pin a transformers version that installs there) | Atisri | ✅ runs on Mac CPU, transformers 4.44; Jetson import check pending | C6, P-Phase-2 |
| C6 | **Milestone M1 — first on-device translation.** Pose TRT engine → C3 → C5 on the Jetson for the Bitcoin clip. Save text + per-stage latency + power for the whole pipeline. This is the Stage 0 baseline every optimization is compared to | Teammate runs on the board, Atisri supports | ⬜ | all optimization rows |
| C7 | Keypoint-scoring path: any pose engine output → C3 → BLEU on the 976 test clips on Colab (~5 min/T4). Needs the pose owner to run the engine over the test clips' frames, which needs the test clips' video (P3). Without it, pose ablations are scored by keypoint agreement only | both | 🔄 scoring path built: `unisign/eval_openasl.py`, 23.16 / 43.17 on 976 test clips (Mac); pose-engine test-clip run waits on P3 | pose accuracy column |
| C8 | Adaptation-training harness: `fine_tuning.py` with mT5 frozen, pose stack + projection trainable (~4.5 M params), checkpoint-resume, sharded pose data on local disk. Needs OpenASL **train** poses (97 K files, redo the 32 GB archive extraction for train). Used by both tracks for input-distribution shifts | Atisri builds; runs for pose shifts are triggered by the pose owner | ⬜ | P8, P9, P10, L9 |
| C9 | Results protocol: every final row = 3 runs, mean ± std, DVFS state logged; ≥3 clips / ≥3 signers; one 30-min sustained run for FP32 and for the best compressed config. One shared table in `RESULTS.md` with fixed columns | both, teammate owns board runs | ⬜ | report |
| C10 | **Milestone M4 — accuracy–energy frontier.** J per sentence (pose + LM, measured) vs BLEU-4 across all configs. The plot the report is built around | both | ⬜ | report |
| C11 | Report + demo video (record early), written against both rubrics | both | ⬜ | — |

---

## 1. Pose track (teammate) — in order

| # | Task | Depends on | Status |
|---|---|---|---|
| P0 | Read `SESSION-LOG.md`, rebuild the Mac `mmpose` env on their own machine if needed (steps in SESSION-LOG §1); get container access on the Jetson | — | ⬜ |
| P1 | **Pick the baseline pose model from C2.** If the checkpoint was trained on RTMW 256x192 (likely), export that model to ONNX (rtmlib ships the ONNX; or export from MMPose), build FP32 engine (TF32 off), reference check vs its own PyTorch/ORT run, latency + power row. RTMPose-x 384x288 (already measured: 56 ms, 11.6 W, 754 mJ/frame) becomes the "oversized" comparison row | C2 | ⬜ |
| P2 | FP16 engine for the baseline model: build, compare (`--simcc-atol 0.05 --kpt-atol-px 2`), power row | P1 | ⬜ (RTMPose-x FP16 build was queued, never run) |
| P3 | Data: `data/openasl_fetch.py` — download N test clips (distinct signers) and ≥300 train frames from ≥5 signers for calibration; crop to bbox; frames + meta. Reuse the yt-dlp section download in SESSION-LOG §3. Many videos are private, iterate | — | ⬜ |
| P4 | Keypoint-agreement metric `07_kpt_agreement.py`: % confident keypoints within 1/2/5 px of the FP32 engine, per group, over all frames. The pose accuracy axis when BLEU is not available | P1 | ⬜ |
| P5 | INT8 PTQ: entropy calibrator in `trt_runner.py` fed from `data/calib_frames/`; INT8 row (latency, power, P4 agreement, and BLEU via C7 when P3 test clips exist) | P2, P3, P4 | ⬜ |
| P6 | Variant sweep: RTMW-x-l 384x288 / RTMW-l-m 256x192 / RTMPose-x / RTMPose-l / RTMPose-m wholebody. Each: FP32 correctness vs own reference, then FP16, then rows | P1, P4 | ⬜ |
| P7 | Resolution row: same model at 384x288 vs 256x192 where checkpoints exist | P6 | ⬜ |
| P8 | Temporal subsampling: run the engine at 24/16/12/8 fps input (drop frames before extraction). Cost is linear in T, biggest energy lever. Accuracy needs a C8 adaptation pass per rate (LM-side encoder length shrinks too, see L7) | P1, C8 | ⬜ |
| P9 | Drop the face group: 51 fewer keypoints, enables a body+hands-only model; needs C8 adaptation (feature width 4C → 3C) | C8 | ⬜ |
| P10 | PTQ vs QAT on the pose front-end (QAT on the 4x24 rig, via C8-style loop on the pose model) | P5, C8 | ⬜ |
| P11 | Preprocess cost: CPU JPEG decode + affine is 19 % of frame time; move affine + normalize to GPU (torch) and re-measure; report separately | P1 | ⬜ |
| P12 | Segmentation heuristic for the live demo: hands-return-to-rest on pose velocity | P1 | ⬜ |
| P13 | Board runs for the LM track when asked: C6, L4 (PyTorch mT5 on Jetson), L8 (TRT mT5 engines), plus the C9 protocol runs and the 30-min thermal runs | as requested | ⬜ |
| P14 | iOS capture app streaming frames to the Jetson, MetricKit from day one (plan §2). Cut at the week-4 gate if behind | — | ⬜ |

---

## 2. Language-model track (Atisri) — in order

| # | Task | Depends on | Status |
|---|---|---|---|
| L0 | Colab env, checkpoint + mT5 on Drive, metric sanity, test poses extracted | — | ✅ |
| L1 | Reproduce OpenASL pose-only eval: **22.53 BLEU-4 / 42.68 ROUGE-L** vs paper 22.67 / 42.77 | L0 | ✅ |
| L2 | C3 converter + C4 round-trip + C5 inference script (the common items above; they are LM-track work) | C1 | ✅ (C4 Jetson row pending J1) |
| L3 | Baseline LM cost, off-board: params, checkpoint MB, peak memory, encoder ms / decoder ms per sentence on a T4 for greedy and beam-4, tokens generated per sentence. Also `max_new_tokens` sweep (100 → 64 → 48) vs BLEU: the cap is a free energy lever | L1 | ✅ Mac CPU timing table (RESULTS.md L3); `max_new_tokens` sweep ⬜ |
| L4 | Baseline LM cost, on-board: C5 in plain PyTorch on the Jetson (via P13): latency, power, peak memory, no OOM. Establishes the hybrid runtime | C5 | ⬜ |
| L5 | **Vocabulary pruning.** Token set from OpenASL train+dev+test + specials + prefix prompt; slice `shared` embedding + `lm_head`; remap tokenizer; BLEU / params / MB / peak mem before vs after. Expect ~10–20 K of 250 K tokens kept and ~0 BLEU change. Do before any export so downstream numbers use the pruned model | L1 | ✅ -0.28 BLEU-4 [CI -0.63,+0.05], 587.7M→243.6M, 1187→571 MB |
| L6 | Decode-time knobs, no retraining: greedy vs beam-4 (BLEU vs decode energy ×N), `max_new_tokens` cap, INT8 KV cache. Rows via L4-style board runs | L4, L5 | 🔄 off-board half: greedy −2.00 [−2.63,−1.41], beam-2 −0.81 [−1.29,−0.39] BLEU-4 vs beam-4 (RESULTS.md L6.1); board rows wait on J2/J3 |
| L7 | Encoder input length: with temporal subsampling (P8) the encoder sequence shrinks; measure encoder ms/energy vs T at 30/24/16/12/8 fps-equivalent lengths. One knob, two payoffs | L3 | ⬜ |
| L8 | **mT5 ONNX with KV cache → TensorRT.** `task3_mt5_onnx/02_export_manual.py --verify` on the pruned model; ORT vs PyTorch logits ≥12 steps at 3 encoder lengths; then engines on the Jetson (`03_build_engines.py`, one per process) via P13; single-step, then multi-step drift check. **3-day budget**; fallback is the hybrid runtime (TRT pose + PyTorch mT5), written up as a deployment reality | L5 | 🔄 export half ✅: `models/mt5_pruned_onnx/{encoder,decoder_init,decoder_step}.onnx`, ORT verify PASS at 3 lengths, `unisign/onnx_decode.py` tokens identical to PyTorch; board half = J4 (`unisign/trt_decode.py`) |
| L9 | Weight-only INT8 for mT5 (W8A16) in whichever runtime survives L8; BLEU + memory + power. Activation quant only if W8A16 is clean | L8 | ⬜ |
| L10 | C8 training harness (frozen mT5, trainable pose stack) + **seed variance**: 4 seeds in parallel on the 4x24 rig, BLEU spread → the noise band every later delta is judged against. Needs OpenASL train poses (re-extract from the archive, ~30 GB again) | L1 | ⬜ |
| L11 | Adaptation runs for the pose owner's shifts (P8 frame rates, P9 face drop, P6 variant swap if C2 says the baseline changed). One short pass each; report BLEU vs the un-adapted number so the value of adaptation is itself a result | L10, P8/P9 | ⬜ |
| L12 | Bootstrap CIs on BLEU for every accuracy number (mandatory for anything under ~1 BLEU) | L1 | ✅ `unisign/bootstrap_ci.py`, paired, 1000 resamples |
| L13 | LLM correction stage (Course B): confidence-gated on token logprobs from C5, minimal-edit prompt, three-condition ablation (none / single sentence / N prior turns), edit distance + BLEU + tokens-in vs joules. Text-only API, no VLM. Last; cut without regret | C5, L12 | ⬜ |
| L14 | Optional stretch: context-conditioned encoder `[text_emb(prior turns); proj(F_sign)]`, mT5 frozen, trained with C8. Most likely a null result; week 6 only if everything above is done | L10 | ⬜ |

---

## 3. Milestones

| | Needs | Means |
|---|---|---|
| **M1** first on-device translation | C1–C6 | Stage 0 baseline: text + whole-pipeline latency + power |
| **M2** pose ladder | P1, P2, P4, P5, P6 (+ C7 for BLEU) | FP32 / FP16 / INT8 / variants at agreement + BLEU + latency + power |
| **M3** LM ladder | L3–L9 | full vs pruned, PyTorch vs TRT, W8A16, greedy vs beam, caps: BLEU + memory + power |
| **M4** frontier | M2 + M3 + L12 | J/sentence vs BLEU-4 plot with CIs |
| **Week-4 gate** | M1 + one compressed row on each side with measured energy | If missing: cut LLM stage and phone networking; ship Jetson-only study + local phone app |


## 5. Jetson request queue (LM track → pose owner, P13)

Things the LM track needs run on the board. Each is copy-paste; results come back via git or scp.

| # | Request | Commands | Send back |
|---|---|---|---|
| J1 (= C1) | Copy the existing RTMPose-x keypoints JSON off the board | on the Jetson host: `scp ~/atisri-cv-jetson/results/rtmpose_* <mac>:~/atisri-cv-jetson/results/` or commit it | `results/rtmpose_trt_fp32.json` |
| J2 (= L4) | First on-board LM run, full and pruned checkpoint | copy `unisign/ common/ data/openasl_ref_pose/ weights/mt5-base weights/openasl_pose_only_slt*.pth` to the board; in the container: `pip3 install "transformers>=4.45,<5" sentencepiece`; then `python3 -m unisign.unisign_infer --pkl data/openasl_ref_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl --ckpt weights/openasl_pose_only_slt.pth --mt5 weights/mt5-base --device cuda --repeat 3 --out results/l4_jetson_full_fp32.json` and the same with `_pruned.pth` → `l4_jetson_pruned_fp32.json` | the two JSONs + the printed `[infer]` lines (per-stage ms, text, peak GPU mem) |
| J4 (= L8 board half) | Build TensorRT engines from the pruned mT5 ONNX and run the cached decode loop on the board | copy `models/mt5_pruned_onnx/` (1.5 GB) to the board; in the container, one engine per process: `python3 task3_mt5_onnx/03_build_engines.py --onnx-dir models/mt5_pruned_onnx --which encoder`, then `--which decoder_init`, then `--which decoder_step --enc-len 1,264,512 --dec-len 1,1,128`; then `python3 -m unisign.trt_decode --engine-dir models/mt5_pruned_onnx --ckpt weights/openasl_pose_only_slt_pruned.pth --mt5 weights/mt5-base --pkl data/openasl_ref_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl` (FP32 first; then `--fp16` engines) | printed `[trt]`/`[check]` lines; PASS = tokens identical to PyTorch (FP32) |
| J3 | Same two runs wrapped in the power logger (once J2 works) | `04`-style wrapper to be added to `unisign_infer.py` (`--power-json`) | JSONs |

## 4. Rules

- Every number in `RESULTS.md` comes from a run on the stated hardware. Cells not measured are `—`.
- The released Uni-Sign checkpoint stays the FP32 reference rung; nothing overwrites it.
- Say "blocked on Cx/Px/Lx" the moment it happens. Schedule risk lives in C2, C5, L8.
- No sudo on the Jetson: 15 W mode only, DVFS logged not pinned, power budgets are measured averages.
