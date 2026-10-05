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
| C0 | Put the repo in git (private GitHub), `.gitignore` for `weights/ models/*.onnx models/*.engine data/raw data/test_frames data/calib_frames results/*.csv results/*.npz venv/`. Large files on a shared Drive folder mirroring repo paths. Branches `pose/...` and `lm/...`, PRs to `main` | Atisri | ✅ | everything |
| C1 | Copy Jetson results to the Mac and commit `results/rtmpose_trt_fp32.json` (the keypoints JSON for the Bitcoin clip). Schema: per frame `{"frame", "keypoints": 133x2 px in the cropped frame, "scores": 133}` plus `summary` | Atisri (has the board today), then teammate | ✅ `results/rtmpose_trt_fp32.json` | C2 |
| C2 | **Which extractor made the released poses.** Run `task1_rtmpose/08_compare_extractors.py` with the authors' pkl for the Bitcoin clip (download from Drive `unisign/openasl_test_pose/`), the Jetson RTMPose-x JSON, and the two rtmlib pkls already in `results/`. Write the answer in `RESULTS.md`. Decides the Jetson baseline pose model (P1) | Atisri runs it, both read the result | ✅ answer: 256x192 RTMW-l-m (see RESULTS.md); Jetson RTMPose-x row added when C1 lands | P1, P4, L3 |
| C3 | `common/pose_to_unisign.py`: keypoints JSON → Uni-Sign input tensors (body/left/right/face groups, root-normalised, thr 0.3), lifted from `datasets.py: load_part_kp / crop_scale`. Unit test: our JSON → pkl → their loader gives the same tensors as their pkl. Also writes the authors' pkl format so any keypoints file can be scored | Atisri | ✅ bit-exact vs their loader on 2 pose files | C4, L2, P-everything that needs BLEU |
| C4 | Round-trip on Colab: Jetson keypoints (C3) through the released checkpoint → sentence for the Bitcoin clip, next to the sentence from the authors' pkl for the same clip | Atisri | ✅ RESULTS §2.5 / §2.5c | C5 |
| C5 | `unisign/unisign_infer.py`: plain PyTorch, no mmpose/mmcv/mmdet/deepspeed/decord, `--keypoints --ckpt --mt5 --out`, greedy, `max_new_tokens` cap, per-stage timing, output `{"text","tokens","token_logprobs","timing_ms":{gcn,encoder,decoder}}`. Must import in the Jetson container (Python 3.12, torch 2.8; pin a transformers version that installs there) | Atisri | ✅ runs on Mac CPU, transformers 4.44; Jetson import confirmed by M1 (§5.1) | C6, P-Phase-2 |
| C6 | **Milestone M1 — first on-device translation.** Pose TRT engine → C3 → C5 on the Jetson for the Bitcoin clip. Save text + per-stage latency + power for the whole pipeline. This is the Stage 0 baseline every optimization is compared to | Teammate runs on the board, Atisri supports | ✅ 2026-09-28, RESULTS §5.1: 9052.8 ms, 50.37 J/sentence | all optimization rows |
| C7 | Keypoint-scoring path: any pose engine output → C3 → BLEU on the 976 test clips on Colab (~5 min/T4). Needs the pose owner to run the engine over the test clips' frames, which needs the test clips' video (P3). Without it, pose ablations are scored by keypoint agreement only | both | 🔄 path works (§C7); pose→BLEU done at n=30 (§2.5c), never at n=976 | pose accuracy column |
| C8 | Adaptation-training harness: `fine_tuning.py` with mT5 frozen, pose stack + projection trainable (~4.5 M params), checkpoint-resume, sharded pose data on local disk. Needs OpenASL **train** poses (97 K files, redo the 32 GB archive extraction for train). Used by both tracks for input-distribution shifts | Atisri builds; runs for pose shifts are triggered by the pose owner | ✅ `unisign/train_adapt.py`, §C8.1 / §C8.2 | P8, P9, P10, L9 |
| C9 | Results protocol: every final row = 3 runs, mean ± std, DVFS state logged; ≥3 clips / ≥3 signers; one 30-min sustained run for FP32 and for the best compressed config. One shared table in `RESULTS.md` with fixed columns | both, teammate owns board runs | 🔄 **repeats, DVFS and both sustained runs done 2026-10-03 (§5.6)**; the ≥3-clips-spanning-the-crop-range part is J6, still outstanding | report |
| C10 | **Milestone M4 — accuracy–energy frontier.** J per sentence (pose + LM, measured) vs BLEU-4 across all configs. The plot the report is built around | both | ✅ 2026-09-30 `results/frontier.png` + `.csv`, RESULTS §L16. **Energy is composed, not measured end-to-end** | report |
| C11 | Report + demo video (record early), written against both rubrics | both | 🔄 **`REPORT.md` drafted 2026-10-05**, written against the four rubric dimensions, every headline number machine-verified against `results/`. Two items marked outstanding in it: the J9 adaptation run and the demo video. Needs the LM track's review and the adaptation result folded in | `REPORT.md` |


> **Withdrawn claim, do not resurrect (RESULTS.md §5.2, 2026-10-02).** The "LM load peaks near 4.2 GB
> and memory is binding at load" finding was an occupancy artefact — the probes ran on a shared board
> with no `whoelse` check. Measured peak device allocation is 0.98 GB. The LM track's original
> "memory is not binding at 2.577 GB peak" line is **restored**, and **INT8 is not reopened**.

---

## 1. Pose track (teammate) — in order

> **Statuses below are inferred by the LM track from artifacts committed to this repo and from
> `results/RESULTS.md`, not reported by the pose owner. Please correct them.** Where a row is marked
> done, the evidence file is named so a wrong call is easy to spot.

| # | Task | Depends on | Status |
|---|---|---|---|
| P0 | Read `SESSION-LOG.md`, rebuild the Mac `mmpose` env on their own machine if needed (steps in SESSION-LOG §1); get container access on the Jetson | — | ✅ |
| P1 | **Pick the baseline pose model from C2.** If the checkpoint was trained on RTMW 256x192 (likely), export that model to ONNX (rtmlib ships the ONNX; or export from MMPose), build FP32 engine (TF32 off), reference check vs its own PyTorch/ORT run, latency + power row. RTMPose-x 384x288 (already measured: 56 ms, 11.6 W, 754 mJ/frame) becomes the "oversized" comparison row | C2 | ✅ RTMW-l-m 256x192 baseline, `results/rtmw_trt_fp32.json` |
| P2 | FP16 engine for the baseline model: build, compare (`--simcc-atol 0.05 --kpt-atol-px 2`), power row | P1 | ✅ `results/rtmw_trt_fp16.json`, `rtmw_fp16_15W_power.json`; RTMPose-x FP16 also built |
| P3 | Data: `data/openasl_fetch.py` — download N test clips (distinct signers) and ≥300 train frames from ≥5 signers for calibration; crop to bbox; frames + meta. Reuse the yt-dlp section download in SESSION-LOG §3. Many videos are private, iterate | — | ✅ 30 clips / 5 signers used in §2.5c |
| P4 | Keypoint-agreement metric `07_kpt_agreement.py`: % confident keypoints within 1/2/5 px of the FP32 engine, per group, over all frames. The pose accuracy axis when BLEU is not available | P1 | ✅ §P4 |
| P5 | INT8 PTQ: entropy calibrator in `trt_runner.py` fed from `data/calib_frames/`; INT8 row (latency, power, P4 agreement, and BLEU via C7 when P3 test clips exist) | P2, P3, P4 | ⬜ no INT8 pose artifact in the repo |
| P6 | Variant sweep: RTMW-x-l 384x288 / RTMW-l-m 256x192 / RTMPose-x / RTMPose-l / RTMPose-m wholebody. Each: FP32 correctness vs own reference, then FP16, then rows | P1, P4 | ✅ RTMW / RTMPose-x / rtmlib lightweight+performance, §2.5c |
| P7 | Resolution row: same model at 384x288 vs 256x192 where checkpoints exist | P6 | 🔄 across models (RTMPose-x 384 vs RTMW 256), not same-model |
| P8 | Temporal subsampling: run the engine at 24/16/12/8 fps input (drop frames before extraction). Cost is linear in T, biggest energy lever. Accuracy needs a C8 adaptation pass per rate (LM-side encoder length shrinks too, see L7) | P1, C8 | ✅ §2.9/P8 at 30/24/16/12/8 fps |
| P9 | Drop the face group: 51 fewer keypoints, enables a body+hands-only model; needs C8 adaptation (feature width 4C → 3C) | C8 | ⬜ neither side has run it |
| P10 | PTQ vs QAT on the pose front-end (QAT on the 4x24 rig, via C8-style loop on the pose model) | P5, C8 | ⬜ not started |
| P11 | Preprocess cost: CPU JPEG decode + affine is 19 % of frame time; move affine + normalize to GPU (torch) and re-measure; report separately | P1 | ⬜ not started (§2.2b quantifies the cost: imread 6.02 + preprocess 5.11 ms/frame) |
| P12 | Segmentation heuristic for the live demo: hands-return-to-rest on pose velocity | P1 | ⬜ not started |
| P13 | Board runs for the LM track when asked: C6, L4 (PyTorch mT5 on Jetson), L8 (TRT mT5 engines), plus the C9 protocol runs and the 30-min thermal runs | as requested | 🔄 J1/J2/J4, M1, J5, J6, J7 and J8 delivered; board access restored 2026-10-02. **C9 protocol work closed 2026-10-04.** Open: J10 and J11 (`OPEN-ISSUES-LM-2026-10-05.md`), plus J9 pending a design fix |
| P14 | iOS capture app streaming frames to the Jetson, MetricKit from day one (plan §2). Cut at the week-4 gate if behind | — | ⬜ optional; gate passed so not forced, but nothing built |

---

## 2. Language-model track (Atisri) — in order

| # | Task | Depends on | Status |
|---|---|---|---|
| L0 | Colab env, checkpoint + mT5 on Drive, metric sanity, test poses extracted | — | ✅ |
| L1 | Reproduce OpenASL pose-only eval: **22.53 BLEU-4 / 42.68 ROUGE-L** vs paper 22.67 / 42.77 | L0 | ✅ |
| L2 | C3 converter + C4 round-trip + C5 inference script (the common items above; they are LM-track work) | C1 | ✅ (C4 Jetson row pending J1) |
| L3 | Baseline LM cost, off-board: params, checkpoint MB, peak memory, encoder ms / decoder ms per sentence on a T4 for greedy and beam-4, tokens generated per sentence. Also `max_new_tokens` sweep (100 → 64 → 48) vs BLEU: the cap is a free energy lever | L1 | ✅ §L3, §L3.2 (cap 64 bit-identical to cap 100) |
| L4 | Baseline LM cost, on-board: C5 in plain PyTorch on the Jetson (via P13): latency, power, peak memory, no OOM. Establishes the hybrid runtime | C5 | ✅ §L4 (J2, 2026-09-18) |
| L5 | **Vocabulary pruning.** Token set from OpenASL train+dev+test + specials + prefix prompt; slice `shared` embedding + `lm_head`; remap tokenizer; BLEU / params / MB / peak mem before vs after. Expect ~10–20 K of 250 K tokens kept and ~0 BLEU change. Do before any export so downstream numbers use the pruned model | L1 | ✅ -0.28 BLEU-4 [CI -0.63,+0.05], 587.7M→243.6M, 1187→571 MB |
| L6 | Decode-time knobs, no retraining: greedy vs beam-4 (BLEU vs decode energy ×N), `max_new_tokens` cap, INT8 KV cache. Rows via L4-style board runs | L4, L5 | ✅ §L6.1; board energy grid in §C |
| L7 | Encoder input length: with temporal subsampling (P8) the encoder sequence shrinks; measure encoder ms/energy vs T at 30/24/16/12/8 fps-equivalent lengths. One knob, two payoffs | L3 | ✅ §L7.1 / §L7.2 / §L7.3 |
| L8 | **mT5 ONNX with KV cache → TensorRT.** `task3_mt5_onnx/02_export_manual.py --verify` on the pruned model; ORT vs PyTorch logits ≥12 steps at 3 encoder lengths; then engines on the Jetson (`03_build_engines.py`, one per process) via P13; single-step, then multi-step drift check. **3-day budget**; fallback is the hybrid runtime (TRT pose + PyTorch mT5), written up as a deployment reality | L5 | ✅ §L8.1 ONNX, §L8.2 TRT on board. **FP16 mT5 overflows**, so the hybrid runtime is what deploys |
| L9 | Weight-only INT8 for mT5 (W8A16) in whichever runtime survives L8; BLEU + memory + power. Activation quant only if W8A16 is clean | L8 | ✅ §L9.1 W8A32. **Dropped from the deployment config**; pruned FP32 deploys |
| L10 | C8 training harness (frozen mT5, trainable pose stack) + **seed variance**: 4 seeds in parallel on the 4x24 rig, BLEU spread → the noise band every later delta is judged against. Needs OpenASL train poses (re-extract from the archive, ~30 GB again) | L1 | ✅ `results/l10_seeds.json`: 0.05 BLEU-4 / 0.15 ROUGE-L over 3 seeds |
| L11 | Adaptation runs for the pose owner's shifts (P8 frame rates, P9 face drop, P6 variant swap if C2 says the baseline changed). One short pass each; report BLEU vs the un-adapted number so the value of adaptation is itself a result | L10, P8/P9 | 🔄 16 fps done (§L15: +1.04 ROUGE-L [+0.25, +1.80]). **12 fps, 8 fps and the P9 face drop not run** |
| L12 | Bootstrap CIs on BLEU for every accuracy number (mandatory for anything under ~1 BLEU) | L1 | ✅ `unisign/bootstrap_ci.py`, paired, 1000 resamples |
| L13-llm | LLM correction stage (Course B): confidence-gated on token logprobs from C5, minimal-edit prompt, three-condition ablation (none / single sentence / N prior turns), edit distance + BLEU + tokens-in vs joules. Text-only API, no VLM. Last; cut without regret | C5, L12 | ⬜ cuttable |
| L14 | Optional stretch: context-conditioned encoder `[text_emb(prior turns); proj(F_sign)]`, mT5 frozen, trained with C8. Most likely a null result; week 6 only if everything above is done | L10 | ⬜ not started |

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

Things the LM track needs run on the board. Results come back via git or scp.

### Outstanding (as of 2026-10-05)

**Board access restored 2026-10-02** — the pose track ran the §5.2 memory probes on a verified-empty
board. The demo video for C11 still needs board time.

**J5, J6, J7 and J8 are all delivered** (rows below), so the C9 protocol work is closed. Four items
are open: **J12 first** — the 531 remaining test clips, which §L17 shows J9 needs before it can
resolve anything (`REPLY-PROBE-RESULT-2026-10-05.md`) — then **J9** (adaptation, now gated on J12),
then J10 and J11 from `OPEN-ISSUES-LM-2026-10-05.md`.

> **Updated by the pose track 2026-10-05: J12's extraction is DONE, so three items are open, not
> four.** The board holds **931 / 931** pkl pairs plus the authors' reference poses for the same 931
> clips; write-up in `results/RESULTS.md` §2.5k. The `pruned_ours_fps24` eval at `SLT_TAG=n931` is
> the last step and is running. **J9 is therefore no longer gated** — its step 1 (dev pose
> extraction, 918 clips fetched) starts as soon as the eval frees the board, since only one
> TensorRT process may run at a time. Order is now J9 → J10, with J11 agreed unrun before the report.

| # | Request | Why it matters | Priority |
|---|---|---|---|
| J9 | **Adapt the pose encoder to our own keypoints** (`ASK-ADAPT-TO-OUR-POSES-2026-10-05.md`, **revised** after `REPLY-J9-ADAPT`; response in `REPLY-TO-J9-ISSUES-2026-10-05.md`) | The pose gap is **−2.24 BLEU-4 [−3.50, −1.09]**, established, and is ~2.2 of the ~2.3 accuracy points available. §2.5i/§2.5j show distribution shift, not information loss. **Now carries a control arm** (train on the authors' dev poses, score on our-pose test) because `adapted − un-adapted` cannot separate adaptation from generic fine-tuning — L15.2 measured that at +0.18/+0.70 with no shift. **Step 0 is a 967-clip YouTube fetch, not a copy** (zero dev overlap in `data/clips`), which is the long pole; a 300–400 clip subset is proposed first. Split: steps 1/1b pose track (board), step 2 LM track (Colab — **not** the board). **Now gated on J12**: §L17 shows the n=400 test set cannot resolve the effect size dev-scale adaptation would produce (`REPLY-PROBE-RESULT-2026-10-05.md`) | high — **behind J12** |
| **J12** | **Extract the remaining test clips through the board pose engine**, taking `results/pkl_split_rtmw_fp16{,_raw}/` from 400 clips to every test clip that exists on disk | **Goes ahead of J9 in priority; it does not replace it** — without dev poses there is no adapted arm to score, so J12 makes J9 *interpretable* rather than unnecessary. §L17's scaling probe puts dev-scale adaptation at ~39 % of the 20,000-clip gain; against the 2.24 BLEU-4 pose gap that is **+0.88**, while the n=400 paired interval has a half-width of **±1.20**. So a real J9 effect of the predicted size returns *not established* and the agreed framing licenses nothing. Going to 976 clips narrows the interval to about **±0.77**, which +0.88 clears. More training data cannot help if the measurement cannot resolve the result. **Corrected by the pose track 2026-10-05** (`REPLY-J12-ACCEPTED-2026-10-05.md`): it is **531 clips, not 576**, and the paired ceiling is **931, not 976** — `data/clips/index.json` records `requested 974, n_ok 931, n_failed 43`, so 45 test names have no clip on disk and never will. `sqrt(400/931)` = 0.6555 gives a half-width of **±0.790**, which +0.88 still clears, but by 0.09 rather than 0.11, and 931 is the ceiling rather than a waypoint. Cost measured on the pose-track Mac: 111,388 frames, ~45–47 min of board compute, 3.66 GB of JPEGs staged in batches, ~426 MB of pkls back. **DELIVERED 2026-10-05 (extraction), §2.5k** — and the cost estimate ran low: the real figure is **57.5 min** of wall clock for the 528 clips (20:13:39 → 21:11:10 UTC, 110,357 frames, 32.0 frames/s aggregate), because the estimate counted GPU time and the run also pays for six staging rsyncs and six reclaim passes. On the board now: 340 MB + 340 MB for our two normalisations over all 931 clips, plus 560 MB of the authors' reference set for the same clips; 0 frame dirs and 0 `.part` files left | ✅ extraction done; eval pending |
| J10 | **Re-run the §2.9C beam × T sweep in-process with the RTMW FP16 engine resident.** Pruned checkpoint, nothing else changed, three separate processes per §5.6 | The **2.08–2.27×** understatement of the decoder-width energy term survives the checkpoint fix, and this is the one axis of the frontier `REPORT.md` §6 has to mark unquotable. §5.4's four cells show a consistent sign (greedy overestimated, beam 4 underestimated), which is what a different resident footprint would do — unverified | **high** |
| J11 | **Per-stage energy from one end-to-end run** at `beam 4 @ 24 fps` — pose, convert and LM joules separately, same run | The 3.16 J (~6 %) composition residual is **not** a within-run gap: the 2×2 found the stages exactly additive and the convert step worth 0.143 J. It is an across-run artefact, and the 2×2's per-sentence-only energy total cannot split it between the pose and LM terms. §5.4 already instruments the LM stage alone | low — ~6 % against a ±3 % run-to-run term is ~2σ, consistent with noise |

### Delivered

| # | Request | Result |
|---|---|---|
| J1 (= C1) | RTMPose-x keypoints JSON off the board | ✅ `results/rtmpose_trt_fp32.json` |
| J2 (= L4) | First on-board LM run, full and pruned checkpoint | ✅ §L4 |
| J3 | Power-logged versions of the J2 runs | ✅ folded into §L4 / §C |
| J4 (= L8) | TensorRT engines from the pruned mT5 ONNX + cached decode loop | ✅ §L8.2. **FP16 overflows** (token 0 every step at −ln 26078); FP32 engines are the usable ones |
| — | M1 end-to-end | ✅ §5.1 |
| — | `cpu0_MHz` on LM power runs | ✅ withdrawn as an ask: `power_logger` already samples it into `aux_avg`; M1 carries 866.76 MHz, below the 897 MHz downclock point |
| J5 | 24 fps added to the end-to-end grid | ✅ 2026-10-03, and widened to 3×2 `{source, 24, 16} × {beam 4, greedy}`. **beam 4 @ 24 fps measured end to end: 7728.7 ms, 42.94 J/sentence** (§5.3). The grid also found the composition's decoder-width term understated 3–5× and `mJ_per_frame` rising under subsampling |
| J6 | C9 crop-range rows | ✅ 2026-10-04, §2.9D. Five clips spanning 15.1× of crop area: **15.1× area costs only 1.34× energy**, 85 % fixed. The frontier's "~7 % pessimistic" caveat is really **−2.6 %** for a median clip — an overestimate of ~3×, and no longer the dominant uncertainty on that axis |
| J8 | C9 protocol applied to the final rows | ✅ 2026-10-03, §5.6. Sustained 30 min at all three loads: **no throttling** (drift −0.26 % FP32 pose, −0.56 % end-to-end; Tj max 57.44 / 51.75 °C). Phase 2 = the deployable config, never run before, **38.48 J/sentence sustained vs 37.75 short-window (+1.9 %)** — so every short-window row in RESULTS.md is validated. Process-level repeats: TRT is 0.2–0.3 % CV while imread/preprocess are ~9 % CV **at FP16 only**, so run-to-run variance is a CPU-stage effect and not a GPU one |
| J7 | Multi-clip board accuracy, ~100 clips | ✅ 2026-10-03, §2.5g. 100 clips / **100 distinct videos**, our own board keypoints, pruned checkpoint: **22.99 BLEU-4 / 43.43 ROUGE-L at 24 fps**. The deployable config's accuracy is now **measured directly** rather than composed. Authors' reference poses restricted to the same 100 clips |
| — | `--keep-fps` on `e2e_translate.py` + `jetson/e2e_2x2.sh` | ✅ 2026-10-02, pose track |
| J10 | Re-run the §2.9C beam × T sweep with the pose engine resident | ✅ 2026-10-05, §2.9E. **Answer is NO** — residency does not explain the 2.08–2.27× understatement. The decoder-width term goes 2.16/1.94/1.34/2.49/1.15 → 2.15/1.80/1.21/2.40/1.13, i.e. **0.90–1.00×**: the spread *narrows* ~5 % where the hypothesis needed it to widen ~110 %. Per-cell energy within ±0.15 J, latency within ±1 %. The leading candidate is eliminated and the beam-width axis **stays unquotable**; `REPORT.md` §6 keeps it open-and-narrowed. Incidental: a resident TRT pose context costs ~292 MB and ~0 W idle |
| J12 | Extract the remaining test clips through the board pose engine | ✅ 2026-10-05, §2.5k. **931 / 931** pkl pairs in `results/pkl_split_rtmw_fp16{,_raw}` plus the authors' reference poses for the same 931 clips. 528 clips / 110,357 frames in 57.5 min at 32.0 frames/s aggregate. Per-batch timings are **reconstructed from pkl mtimes** (`jetson/j12_timeline.py --gap-min 5`), because the driver's log was lost to EPIPE; the method reproduces §2.5g's independently recorded 24,365 frames / 36.0 frames/s / 11.3 min, and the rungs sum to the same 203,562 frames as the clip metas. Accuracy row pending the `n931` eval |

### Note on re-exporting the TRT engines

The engines were built at the old 26,078 keep set; the leak-free set is 26,025. Re-export is
bookkeeping — the leak is worth −0.03 BLEU-4 [−0.11, +0.01] — and the latency rows stand either way.
A pre-pruned directory at 26,025 with `keep_ids.json` is ready (Block 5).

---

## 6. Week-4 gate (row 5.3) — PASSED, 2026-09-30

The gate asks for M1 plus one compressed row with measured energy on each side; L13 and P14 get cut
only if that is missing.

* Pose side: M1 (§5.1) plus the P8 rate sweep with measured pose energy at 30/24/16 fps.
* LM side: vocabulary-pruned FP32 with the measured beam × T energy grid (§C) and 976-clip accuracy.

**Nothing is cut.** INT8 was dropped for its own reason (mT5 FP16 overflow), not by this gate.

---

## 7. Naming collision to be aware of when merging

**`L13` means two different things in this project.** In this file it is the LLM correction stage
(Course B), now renamed **`L13-llm`**. In `results/RESULTS.md` §L13 it is the *accuracy surface for
the frontier plot* (decoder strategy × frame rate). They are unrelated. The RESULTS.md numbering
(L15 adaptation, L16 frontier) is the one the report should follow.

---

## 4. Rules


**Say which Mac.** There are two checkouts — the pose track's machine and the LM track's — and "the
Mac" appears 52 times across the docs without once being qualified. That ambiguity produced two wrong
cross-track corrections on 2026-10-05 (`REPLY-TO-J9-ISSUES-2026-10-05.md`). Write **"pose-track Mac"**
or **"LM-track Mac"**. And since `weights/`, `data/clips*/`, `data/*pose*/` and `results/pkl_*/` are
all gitignored, **treat anything gitignored as invisible to the other track** unless they have said
otherwise: a `ls` on your own checkout is not evidence about theirs.

- Every number in `RESULTS.md` comes from a run on the stated hardware. Cells not measured are `—`.
- The released Uni-Sign checkpoint stays the FP32 reference rung; nothing overwrites it.
- Say "blocked on Cx/Px/Lx" the moment it happens. Schedule risk lives in C2, C5, L8.
- No sudo on the Jetson: 15 W mode only, DVFS logged not pinned, power budgets are measured averages.
