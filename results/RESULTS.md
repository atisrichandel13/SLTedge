# Results log

All numbers measured on the board. Nothing here is estimated.

## Setup

- Jetson Orin Nano 8GB, L4T R36.4.7, TensorRT 10.11.0.33 (container `lpcv:fall2026`), CUDA 12.9.
- Power mode: **15W (mode 0)**, read from `/var/lib/nvpmodel/status`. The board is course-owned with no
  sudo, permanently. So: no 7W mode, no `jetson_clocks`. Every run is at 15W with the default DVFS
  governor. Power budgets in this project are therefore *measured average watts*, not a hardware cap.
  Clock frequency and fan state are logged from sysfs alongside each run to document the DVFS state.
- **Power modes on these boards** (`/etc/nvpmodel/nvpmodel_p3767_0003_super.conf`, JetPack 6.2):
  mode 0 = 15W caps the GPU at **612 MHz** (CPU 1.5 GHz, EMC 2133 MHz); mode 1 = 25W caps it at
  **918 MHz** (EMC 3199 MHz) and is the conf's DEFAULT; mode 2 = MAXN_SUPER uncapped (1020 MHz);
  mode 3 = 7W at 408 MHz. TensorRT latency scales almost exactly with the GPU cap, so every row must
  carry its measured `gpu_MHz` (now a column in every power CSV, `aux_avg.gpu_MHz` in the JSON, and
  `gpu_max_MHz` in `env`). Rows without it are labelled by inference from the latency ratio.
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
| TensorRT FP32 vs PyTorch, engine rebuilt on `jetson-lpcv-03` (own image `slt-jetson:25.06`, 1 GB workspace) | simcc max diff 6.9e-6, 2416/2416 confident kpts at 0.0 px, PASS |

Methodology note: TensorRT enables TF32 by default on Ampere. An "FP32" engine built with defaults is
not FP32. `common/trt_runner.py` now clears the flag for the FP32 baseline.

## Task 1: RTMPose-x pose extraction, per-frame

| Config | Board | GPU MHz | pre ms | TRT ms | post ms | total ms | avg W | idle W | mJ/frame | dyn mJ/frame | GPU % | Tj C |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FP32 (TF32 off), first pass, 3x299 frames (Atisri, Phase 1) | Atisri's board | not logged; **~918 inferred** (see note) | 10.6 | 44.4 | 1.5 | 56.4 | 11.62 | 3.73 | 754 | 512 | 68 | 56 |
| FP32 (TF32 off), 3x299 frames, 2026-09-18 | `jetson-lpcv-03`, mode 0 (15W) | **612 measured** (every busy sample) | 10.7 | 66.7 | 1.5 | 79.2 | 8.20 | 3.62 | 733 | 410 | 74 | 53 |
| **RTMW-l-m 256x192 FP32 (TF32 off), 3x299 frames, 2026-09-18 (P1 baseline)** | `jetson-lpcv-03`, mode 0 (15W) | 598 mean over busy samples (governor 408–612; 306/353 at 612) | 6.1 | **25.6** | 1.2 | **33.0** | 6.53 | 3.62 | **275** | 123 | 60 | 51 |
| **RTMW-l-m 256x192 FP16, 3x299 frames, 2026-09-18 (P2)** | `jetson-lpcv-03`, mode 0 (15W) | **310 mean** over busy samples (0.9 % at 612) | 4.7 | **13.8** | 1.2 | **19.7** | 4.93 | 3.59 | **127** | 34 | 53 | 50 |
| RTMPose-x FP16 plain — **fails the gate** (head overflow), 3x299 frames, 2026-09-18 (P2) | `jetson-lpcv-03`, mode 0 (15W) | 414 mean over busy samples (1.8 % at 612) | 9.4 | 22.7 | 1.5 | 33.5 | 6.20 | 3.62 | 254 | 106 | 58 | 51 |
| **RTMPose-x FP16 mixed** (head kept in FP32), 3x299 frames, 2026-09-18 (P2) | `jetson-lpcv-03`, mode 0 (15W) | 413 mean over busy samples (2.0 % at 612) | 10.2 | **23.3** | 1.5 | **35.0** | 6.15 | 3.60 | **264** | 110 | 58 | 51 |

Raw: `results/rtmpose_fp32_15W_power.json`, `.csv` (lpcv-03 run, in the repo); logs in `results/logs/`.
RTMW-l-m: `results/rtmw_fp32_15W_power.json`, `.csv`, `results/rtmw_trt_fp32.json`, reference
`results/rtmw_reference.npz` (ORT CPU, 20 frames, gitignored), logs `results/logs/*rtmw*`.

**P1: RTMW-l-m 256×192 is the Jetson pose baseline (2026-09-18).** ONNX from the OpenMMLab SDK zip
(opset 11, input `input` 1×3×256×192, outputs `simcc_x` 133×384 / `simcc_y` 133×512), preprocessing
from `models/rtmw/preproc.json`. Gate against an ONNX Runtime CPU reference on 20 frames (there is no
PyTorch checkpoint for this export; on RTMPose-x ORT matched PyTorch to 8e-6): simcc max diff 2.0e-5,
mean 6.1e-7, keypoint max diff **0.0 px** over 2458 confident keypoints, score diff 2.6e-6 → **PASS**.
Engine 158.5 MB, built in 123 s with a 1 GB workspace (builder peak 384 MiB). Versus RTMPose-x on the
same board and power mode: TRT time 66.7 → 25.6 ms (**2.6×**), total 79 → 33 ms (30 fps, 25 fps
end-to-end including JPEG decode), energy 733 → 275 mJ/frame (**2.7×**), average power 8.2 → 6.5 W, GPU
busy 74 → 60 %. Note the GPU governor no longer sits at 612 MHz for this lighter model: busy samples
range 408–612 MHz, so the row's "GPU MHz" is the busy-sample mean; energy per frame is the robust
number. Together with L4/L8: one 10 s sentence (300 frames) now costs ~83 J of pose extraction vs
~220 J with RTMPose-x, before any FP16 or frame-rate reduction.

**P2: FP16 pose engines (2026-09-18).** `02_build_engine.py --fp16 --workspace-gb 1`, gated with
`06_compare_trt.py --simcc-atol 0.05 --kpt-atol-px 2` against the same references as the FP32 rung.
The two models behave completely differently.

| Engine | build s | size MB | simcc max abs Δ | kpt max Δpx | % kpts exact (≤0.5 px) | % ≤2 px | % ≤5 px | gate |
|---|---|---|---|---|---|---|---|---|
| RTMW-l-m FP16 | 534 | 68.7 | 0.039 | 6.50 | 88.5 | 88.5 | 99.8 | **FAIL on px only** |
| RTMPose-x FP16 plain | 631 | 117.1 | 0.649 | 383.0 | 40.8 | 67.7 | 71.7 | **FAIL, unusable** |
| RTMPose-x FP16 mixed (7/465 layers FP32) | 629 | 132.3 | 0.023 | 20.2 | 65.0 | 96.1 | 99.5 | **FAIL on px only** |

- **RTMW-l-m FP16 is usable; the 2 px gate is the wrong instrument.** Every keypoint error is an exact
  multiple of one simcc bin (0.5 crop px = 2.17 image px at this bbox scale): p50 0.0, p90 = p99 =
  2.17 (one bin), worst 6.50 (three bins) on a keypoint whose reference score is 0.58. 88.5 % of the
  2458 confident keypoints are bit-exact and 99.8 % are within 5 px. Nothing overflows (see the scan
  below), so this is argmax quantization noise on flat heatmaps, not numerical failure. The honest
  accuracy statement for a heatmap model is the agreement distribution, not a max over one argmax
  flip — that is what P4's `07_kpt_agreement.py` is for, and the pose→BLEU path (P5) is the decider.
- **RTMPose-x FP16 plain is genuinely broken**, and `08_fp16_range_scan.py` (new; runs the ONNX on
  ORT CPU with every intermediate exposed and reports activations against the FP16 max of 65504)
  names the single cause: the head's ScaleNorm computes a sum of squares, `/mlp/mlp.0/ReduceSum`,
  that peaks at **142 293** — 2.2× over the FP16 limit, with `/mlp/mlp.0/Pow` at 23 565 already
  within 4× of it. One tensor in 376 saturates and takes the whole head with it (p90 error 130 px).
  The same scan over RTMW-l-m's 398 tensors finds **zero** overflowing or near-overflowing tensors,
  which is exactly why its FP16 engine is fine. Raw: `results/{rtmposex,rtmw}_fp16_range.json`.
- **Pinning seven layers of 465 repairs RTMPose-x FP16.** `02_build_engine.py --fp16 --fp32-layers
  'mlp\.0'` keeps the ScaleNorm block in FP32 under `OBEY_PRECISION_CONSTRAINTS` and leaves the rest
  of the network in half precision. Simcc error falls 0.649 → **0.023** (inside the 0.05 tolerance),
  keypoints within 2 px go 67.7 → **96.1 %**, within 5 px 71.7 → **99.5 %**, and the worst remaining
  error sits on a keypoint whose reference score is 0.357, i.e. right at the 0.3 confidence floor.
  Engine 132.3 MB vs 117.1 MB plain FP16 and 231 MB FP32; build time is unchanged at ~630 s. Cost of
  the repair at runtime: **23.3 ms vs 22.7 ms** TRT time and 264 vs 254 mJ/frame, under 3 %. So the
  whole FP16 failure was two tensors, and the fix is nearly free. **This is the RTMPose-x FP16 row to
  use**; the plain FP16 row is kept only as the negative result.
- **Raising the confidence floor removes RTMW's tail entirely.** Re-gating RTMW-l-m FP16 at
  `--min-score 0.7` (1962 keypoints): worst error **2.17 px = exactly one simcc bin**, p90 0.0,
  everything within 5 px, 91.2 % bit-exact. The 6.50 px worst case at the 0.3 floor was three bins on
  a 0.58-score keypoint. A keypoint the model is unsure about has a flat heatmap whose argmax moves
  on rounding noise, which is a property of argmax decoding, not of FP16.
- **Latency comparisons across precisions need the clock column.** The governor drops the GPU as the
  work gets lighter: FP32 RTMW ran at 598 MHz mean over busy samples (86 % of them at the 612 MHz
  cap), FP16 RTMW at **310 MHz** (0.9 % at the cap) and FP16 RTMPose-x at 414 MHz. So RTMW's
  25.6 → 13.8 ms is a 1.9× wall-clock gain achieved *at roughly half the clock*; the per-clock
  speedup is larger, and the honest headline is the energy: **275 → 127 mJ/frame, 2.2×**, average
  power 6.5 → 4.9 W, Tj 50 °C. RTMW FP16 is 19.7 ms/frame of pose-stage compute = **51 fps**, or
  **40 fps** measured on the harness's wall clock, which also pays JPEG decode from disk (§2.2b).
  Either way it is the first pose configuration in this project that clears the 30 fps source rate at
  15 W with headroom, but the two figures are different measurements and must not be mixed.
- **Where P2 leaves the pose stage.** RTMW-l-m FP16 is the configuration to carry forward: 19.7 ms
  of pose-stage compute (51 fps; 40 fps wall clock), 127 mJ/frame at 15 W, against 79.2 ms (12.6 fps;
  11.5 fps wall clock) and 733 mJ for the original RTMPose-x FP32 baseline. That is **4.0× the frame
  rate on the compute basis (3.5× end to end) for 5.8× less energy per frame**, and one 10 s
  sentence costs 38 J of pose extraction instead of 219 J. What is *not* yet established is accuracy
  in the only units that matter for this project: these gates compare engines against their own FP32
  reference on 20 frames of one clip, and say nothing about BLEU. P3 (clips from several signers) and
  the pose→BLEU path decide whether FP16's argmax flips cost translation quality.

**On the gate itself.** Both FP16 engines are reported as FAIL above, and both verdicts come from a
single keypoint. The criterion is a max over per-keypoint pixel error, which one argmax flip on a
flat heatmap saturates, so the number that fails the gate is the number least related to pose
quality. The distribution columns were added to `06_compare_trt.py` for exactly this reason. The
pass criterion has deliberately not been loosened to match: P4's `07_kpt_agreement.py` over full
clips is the intended replacement, and loosening a gate to make a result pass, before the better
measurement exists, is how a project talks itself into a regression.

### 2.2b Where the frame time actually goes, and why there are two fps numbers (2026-09-26)

Prompted by a discrepancy Tushar spotted: the FP16 row was quoted as 19.7 ms *and* 40 fps, but
1000/19.7 = 51, and every other row in the briefing used 1000/total. Both numbers were real; they
were two different metrics printed as one. `03_infer_frames.py` now times the two stages that were
never instrumented, so the wall clock adds up instead of being inferred.

Board `jetson-lpcv-03`, mode 0 (15 W), 3×299 frames of the baseline clip, occupancy checked with the
new `jetson/run.sh whoelse` (no other users, no other GPU client, load 0.0 before start).

| stage | in `total_ms`? | RTMW FP16 | RTMW FP32 | what it is |
|---|---|---:|---:|---|
| `imread_ms` | **no** | 5.33 | 6.83 | `cv2.imread`: JPEG decode of the frame from disk |
| `preprocess_ms` | yes | 4.76 | 6.10 | affine warp to 192×256, BGR→RGB, normalise, transpose (CPU, numpy) |
| `trt_ms` | yes | 13.66 | 25.83 | H2D upload of the 590 KB input + `execute_async_v3` + stream sync |
| `postprocess_ms` | yes | 1.28 | 1.22 | D2H copy of simcc (133×384 + 133×512 = 477 KB), argmax, affine back |
| **`total_ms`** | — | **19.71** | **33.15** | preprocess + TRT + postprocess = the pose stage |
| `collect_ms` | **no** | 0.16 | 0.16 | `kpts.round(2).tolist()` into the results list |
| residual | **no** | 0.01 | 0.01 | loop overhead — the accounting closes |
| **wall ms/frame** | — | **25.20** | **40.14** | what `fps_end_to_end` divides into |
| fps (compute) | — | 50.7 | 30.2 | 1000 / `total_ms` |
| fps (end-to-end) | — | 39.7 | 24.9 | frames / wall_s |

`total_ms` reproduces the published P2 rows exactly (19.708 vs 19.679; 33.146 vs 32.996), so this is
the same measurement with more of it visible, not a new one.

**Answer to "what is the overhead": JPEG decode, 5.3 ms of it, plus 0.16 ms of result
serialisation.** Nothing is missing — the residual is 0.01 ms. Which fps to quote depends on the
claim:
- **fps (compute) is the right number for the design**, because the 5.3 ms JPEG decode is an artifact
  of this harness reading pre-extracted JPEGs off disk. A deployed pipeline decodes H.264 on NVDEC,
  not JPEG on the CPU, so that cost is not a property of our pose stage.
- **fps (end-to-end) is the right number for this harness**, and is what any reproduction of these
  commands will observe.

Both are now emitted as `fps_compute` and `fps_end_to_end` in every run JSON so the choice is
explicit rather than accidental.

**The finding that was not the question: CPU-side stages are a function of the GPU's power draw.**
Between the two runs, `imread` went 5.33 → 6.83 ms and `preprocess` 4.76 → 6.10 ms — both **×1.281,
the same factor to three digits**, on identical input and identical code. Measured `cpu0_MHz` over
the same window: 1045 (FP16) vs 897 (FP32), ×1.165. The GPU at 604 MHz instead of 312 MHz takes
enough of the 15 W cap that the CPU downclocks, and DRAM contention accounts for the rest of the
factor the clock does not explain. `postprocess` is flat (1.28 vs 1.22) because it is half a
device→host copy, which gets *faster* at the higher GPU clock, cancelling out.

Consequences:
- **`pre ms` is not a property of the preprocessing code.** It is a property of the whole
  configuration's power budget. Comparing `pre ms` across engines measures the governor as much as
  the code. (This retires the old §Task-1 note claiming preprocess "does not shrink with GPU
  quantization" — on RTMW it shrinks 22 %.)
- Part of FP16's wall-clock win is a *CPU* win it gets for free: of the 14.9 ms/frame RTMW FP16 saves
  over FP32 end to end, 12.2 ms is the engine (TRT 25.83 → 13.66) and **2.8 ms is the CPU stages
  speeding up** (imread 1.50 + preprocess 1.34, less 0.06 given back by postprocess) because the
  engine left power on the table.
- `cpu0_MHz` was already in every power CSV; it had simply never been read alongside the CPU stages.

Raw: `results/p2b_split_rtmw_fp16.json`, `results/p2b_split_rtmw_fp32.json` (+ `.csv` on the board).

### P4 Keypoint agreement vs the FP32 engine (`task1_rtmpose/07_kpt_agreement.py`, 2026-09-26)

Two samples: 299 frames of the baseline clip for all three FP16 variants, then 1510 frames over
five signers for the two that survive. 133 keypoints each, judged only where the FP32 engine scores the
joint at least 0.3 (the threshold `common/pose_to_unisign.py` gates joints on, so a joint below it
is zeroed before the ST-GCN and its position cannot reach the translation). Distance is Euclidean,
unlike `06_compare_trt.py`'s per-axis max. `unisign_used` is the 69 of 133 keypoints Uni-Sign's
pose-only path actually consumes.

| config | group | n conf | ≤1 px | ≤2 px | ≤5 px | p50 | p90 | p99.9 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RTMW-l-m FP16 | body | 3504 | 85.9 % | 85.9 % | 99.7 % | 0.00 | 2.17 | 7.6 | 36.9 |
| | hands | 11603 | 78.0 % | 78.0 % | 99.2 % | 0.00 | 2.17 | 31.3 | 134.4 |
| | face | 20332 | 91.9 % | 91.9 % | 100.0 % | 0.00 | 0.00 | 3.1 | 3.1 |
| | unisign_used | 19609 | 82.7 % | 82.7 % | 99.5 % | 0.00 | 2.17 | 17.3 | 134.4 |
| | all | 35999 | 86.6 % | 86.6 % | 99.7 % | 0.00 | 2.17 | 8.7 | 134.4 |
| RTMPose-x FP16 plain | body | 3271 | 36.4 % | 70.4 % | 87.6 % | 1.44 | 122.9 | 351.8 | 352.4 |
| | hands | 10740 | 32.5 % | 74.2 % | 98.9 % | 1.44 | 2.05 | 27.2 | 111.3 |
| | face | 20332 | 41.0 % | 49.3 % | **49.6 %** | **108.40** | 323.8 | 500.8 | 507.2 |
| | unisign_used | 18722 | 37.5 % | 69.4 % | 86.5 % | 1.44 | 124.3 | 464.0 | 506.4 |
| | all | 34347 | 37.9 % | 59.1 % | 68.6 % | 1.45 | 144.5 | 465.0 | 507.2 |
| RTMPose-x FP16 mixed | body | 3271 | 48.1 % | 81.2 % | 99.5 % | 1.44 | 2.05 | 8.8 | 39.1 |
| | hands | 10740 | 31.6 % | 73.7 % | 98.9 % | 1.44 | 2.05 | 28.6 | 108.4 |
| | face | 20332 | 86.2 % | 99.5 % | 100.0 % | 0.00 | 1.44 | 2.0 | 2.1 |
| | unisign_used | 18722 | 48.3 % | 81.7 % | 99.3 % | 1.44 | 2.05 | 14.9 | 108.4 |
| | all | 34347 | 65.5 % | 89.7 % | 99.6 % | 0.00 | 2.04 | 11.7 | 108.4 |

- **The 1 px and 2 px columns are not two measurements, they are one bin apart — and the bin is not a
  constant.** A simcc head decodes by argmax over bins, so an error is a whole number of bins, and one
  bin in image pixels is `max(crop_w, crop_h × 0.75) × 1.25 / (input_w × simcc_split_ratio)`: it scales
  with the signer's bbox. Measuring the smallest non-zero error per clip recovers exactly that bin,
  and it varies by 43 % across the five clips:

| clip | crop | RTMW bin | RTMW ≤2 px | RTMPose-x bin | RTMPose-x ≤2 px |
|---|---|---:|---:|---:|---:|
| ImwA3Ctckfk | 548×720 | 1.78 px | 99.2 % | 1.18 px | 98.2 % |
| UoU3ZSuTef4 | 754×720 | **2.45 px** | **86.5 %** | 1.63 px | 88.9 % |
| ZdEwfVNtSmw | 530×720 | 1.75 px | 99.4 % | 1.17 px | 98.4 % |
| ixq65EiuJ_c | 644×720 | **2.09 px** | **87.9 %** | 1.39 px | 98.1 % |
| y7KIrON1uco | 502×702 | 1.71 px | 99.4 % | 1.14 px | 98.2 % |

  RTMW's ≤2 px column reads 99.2–99.4 % on the three clips whose bin is under 2 px and 86.5–87.9 % on
  the two whose bin is over it. That is a 13-point swing produced by bbox geometry alone, with no
  difference in numerical accuracy — pooling the clips (the 94.00 % in the table above) averages the
  two regimes and means nothing. **The ≤5 px column is the only comparable one**, and for a principled
  reason: the largest bin over all five clips and both models is 2.45 px, so 5 px is "within two bins"
  everywhere, uniformly. Any future threshold must be justified in bins, not pixels.
- **Mixed precision repairs the face branch, which plain FP16 destroys.** Plain FP16 RTMPose-x agrees
  with its own FP32 engine on under half of confident face keypoints, with a *median* error of 108 px
  — the head's ScaleNorm overflow (P2) is not a tail effect there, it is the typical case. Pinning
  seven `mlp.0` layers takes face to 100 % within 5 px and body from 87.6 % to 99.5 %.
- **The overflow damage is group-selective, and hands are not where it lands.** Hand agreement is
  the same in the broken and the repaired engine (98.87 % vs 98.93 % within 5 px, p99.9 of 27 vs
  29 px). So hand disagreement is not caused by the FP16 overflow at all; it is argmax instability on
  low-confidence hand joints, and it survives every fix applied so far. Why the same corrupted head
  wrecks face and body while leaving hands intact is not established here.
- **Hands are the accuracy floor for every configuration**, at roughly 99 % within 5 px with a
  p99.9 near 30 px, against 100 % and 3 px for the face in the two good engines. That is the opposite
  of the ordering this project needs: hands carry most of the sign lexicon. The worst confident
  keypoint in each good config is a hand joint scoring barely above threshold (RTMW: 134 px at score
  0.403; mixed: 108 px at 0.438), i.e. a joint the FP32 engine is itself unsure of.
- **Score-threshold crossings are rare but not zero**: 32 keypoints for RTMW FP16 (0.080 %), 9 for
  mixed (0.023 %), 41 for plain FP16 (0.103 %). Each one changes the tensor the LM sees even at 0 px
  of motion, because the joint is present in one engine's input and zeroed in the other's. The overall
  confident fraction is stable to 0.1 pp, so no engine is systematically more or less certain.
- **This replaces the max-px gate, and it changes the verdict.** Both engines that the gate failed
  agree with FP32 on 99.3–99.5 % of the keypoints Uni-Sign consumes, within 5 px. The engine the
  gate also failed for a real reason, plain FP16 RTMPose-x, sits at 86.5 %, with the median face
  keypoint a hundred pixels out. One number separates a bin flip from an overflow; the gate's worst-case
  criterion did not.
- **Five signers, 1510 frames, confirms it.** `jetson/p4_clips.sh` rebuilt all four deployable engines
  on the board and dumped one file per config per clip into `results/kpts/` (20 dumps). The verdict does
  not move, and the ≤5 px figure for the keypoints Uni-Sign consumes improves slightly on the wider
  sample: **99.81 % for RTMW FP16** (was 99.5 % on one clip) and **99.75 % for RTMPose-x FP16 mixed**
  (was 99.3 %). Score-threshold crossings stay rare — 33 of 201 k keypoints (0.016 %) for RTMW and 27
  (0.013 %) for mixed — and the confident fraction is unchanged to 0.1 pp in both. Plain FP16
  RTMPose-x was excluded from this run as already-disqualified, so its numbers remain single-clip.

| config | group | n conf | ≤1 px | ≤2 px | ≤5 px | p50 | p90 | p99.9 | max |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RTMW-l-m FP16 | body | 19589 | 86.3 % | 94.2 % | 99.87 % | 0.00 | 1.76 | 5.3 | 7.8 |
| | hands | 63343 | 83.4 % | 92.5 % | 99.69 % | 0.00 | 1.78 | 7.4 | 122.7 |
| | face | 102680 | 91.8 % | 97.1 % | 100.00 % | 0.00 | 0.00 | 2.5 | 3.5 |
| | feet | 1527 | 80.0 % | 88.5 % | 98.89 % | 0.00 | 2.45 | 12.3 | 58.0 |
| | unisign_used | 104113 | 86.1 % | 94.0 % | **99.81 %** | 0.00 | 1.76 | 6.9 | 122.7 |
| | all | 187139 | 88.3 % | 95.1 % | 99.87 % | 0.00 | 1.72 | 5.5 | 122.7 |
| RTMPose-x FP16 mixed | body | 19118 | 51.3 % | 93.5 % | 99.76 % | 0.00 | 1.68 | 5.9 | 12.8 |
| | hands | 62604 | 44.3 % | 92.2 % | 99.61 % | 1.15 | 1.68 | 9.9 | 229.1 |
| | face | 102680 | 86.8 % | 99.9 % | 100.00 % | 0.00 | 1.17 | 2.0 | 2.3 |
| | feet | 30 | 40.0 % | 83.3 % | 90.00 % | 1.17 | 2.79 | 41.3 | 42.3 |
| | unisign_used | 103301 | 56.1 % | 94.6 % | **99.75 %** | 0.00 | 1.65 | 7.3 | 229.1 |
| | all | 184432 | 68.6 % | 96.6 % | 99.84 % | 0.00 | 1.63 | 5.9 | 229.1 |

- **Hands remain the floor across all five signers, and the tail got worse, not better.** Hand
  agreement is the lowest ≤5 px figure in both engines (99.69 % RTMW, 99.61 % mixed) and the worst
  confident keypoint is a hand joint in both (122.7 px at ref score 0.337 for RTMW; 229.1 px at 0.470
  for mixed, both on `UoU3ZSuTef4`). With five signers the max roughly doubles versus one clip, which
  is what a tail driven by rare low-confidence argmax flips should do when the sample grows — further
  evidence it is instability, not precision. This is the group P5's INT8 work has to be judged on.
- **RTMPose-x barely sees feet: 30 confident foot keypoints against RTMW's 1527**, out of 9060
  possible. It does not affect anything here, since feet are not among the 69 keypoints Uni-Sign
  consumes, but it is a real behavioural difference between the two extractors and worth one line in
  the report.

### 2.5 Pose→BLEU: does the pose front-end change the translation? (`unisign/eval_openasl.py`, 2026-09-26)

The five P3 clips, each config's `results/kpts/` dump → `common/dumps_to_pkl.py` → the released
pose-only checkpoint on the board (FP32, beam 4, max_new_tokens 100, max_length 256, batch 8, ~18 s per
config). The **ceiling row** is the OpenASL authors' own released `pose-rtmpose-192` pkls for these same
five clips, scored through the identical path: it separates the cost of *our pose extractor* from the
cost of the checkpoint, because everything downstream is held fixed.

| poses | BLEU-1 | BLEU-4 | ROUGE-L | preds identical to ceiling | to own FP32 |
|---|---:|---:|---:|---:|---:|
| **Authors' released poses (ceiling)** | 57.30 | **16.23** | **46.17** | 5/5 | — |
| RTMW-l-m FP32 | 47.68 | 9.89 | 43.39 | 1/5 | — |
| **RTMW-l-m FP16** | 47.68 | 9.89 | 43.39 | 1/5 | **5/5** |
| RTMPose-x FP32 | 39.32 | 8.96 | 31.94 | 0/5 | — |
| RTMPose-x FP16 mixed | 42.62 | 9.39 | 33.76 | 0/5 | 3/5 |

**Read the identical-prediction counts, not the BLEU.** Five sentences cannot support a BLEU
comparison: the authors' own poses score 16.23 BLEU-4 here against 22.53–22.67 on the full 976-clip
split (B1.4/B1.5) with the same checkpoint and the same metric code, so the subset alone moves BLEU-4
by 6 points. Any difference below ~1 BLEU-4 in this table is noise. The prediction counts, by contrast,
are exact and discrete.

- **RTMW FP16 is translation-neutral: 5/5 predictions token-identical to its own FP32 engine**, so
  every metric agrees to 12 decimal places. This is the result the pose track needed. P4 said FP16
  moves 0.19 % of consumed keypoints by more than 5 px; 2.5 says that costs *exactly nothing* in
  output text on this sample. **FP16 is adopted for the pose stage**, and the max-px gate that failed
  this engine is now refuted twice, once on keypoints and once on the translation itself.
- **RTMPose-x mixed precision is *not* neutral: 3/5 identical to its own FP32.** Two of five sentences
  change. That tracks P4, where mixed RTMPose-x agreed with FP32 on 99.75 % of consumed keypoints
  against RTMW's 99.81 %, and where its worst confident error was 229 px against RTMW's 123 px. So the
  keypoint metric and the translation agree on the ordering of the two engines, which is the first
  evidence that P4's ≤5 px figure actually predicts downstream behaviour.
- **RTMW-l-m beats RTMPose-x here by 11.5 ROUGE-L** (43.39 vs 31.94) and is the only one of our engines
  to reproduce a ceiling prediction exactly (1/5 vs 0/5). The likely cause is extractor mismatch against
  the frozen ST-GCN rather than pose quality, but see the contradiction below — **this is a provisional
  reading on 5 sentences, not a settled result.**

  **The Uni-Sign paper specifies RTMPose-x, which contradicts the archive.** The paper extracts 133
  COCO-WholeBody keypoints with RTMPose-x from MMPose, and the four index groups in
  `common/pose_to_unisign.py` come from it verbatim. So the paper-faithful extractor is the 384×288
  RTMPose-x we are proposing to drop. Against that, the OpenASL poses the released checkpoint actually
  consumes measure as 192-wide. Measuring the simcc grid in the authors' own released pkls (smallest
  intra-frame coordinate gap = one bin; bbox span / bin is scale-free and equals
  `input_w × split_ratio / padding`):

| poses | span / bin | vs ours-192 | vs ours-288 |
|---|---:|---:|---:|
| ours, RTMW-l-m 256×192 | 291 | 1.000 | — |
| ours, RTMPose-x 384×288 | 447 | 1.539 (theory 1.500 ✓) | 1.000 |
| **authors' released OpenASL poses** | **293** | **1.008** | 0.655 |

  The estimator is validated by our own two engines recovering 288/192 = 1.500 to within 2.6 %. The
  authors' poses match a 192-wide extractor to 0.8 % and miss a 288-wide one by 35 %, which agrees with
  their archive folder being named `pose-rtmpose-192`. **Caveat: the ratio conflates input width with
  `simcc_split_ratio` and bbox padding**, so a 288-wide model padded ~1.9× would fit the same number;
  192 is the best-supported reading, not a proof. Most likely the paper describes their pipeline
  (notably for CSL-News, their own dataset) while the distributed OpenASL poses were produced at 192.
  Since we use the released checkpoint **frozen**, what governs is the archive's distribution, not the
  paper's prose — but anyone *training* a model should follow the paper and use RTMPose-x.
- **Our RTMPose-x pipeline is not the problem, which was the obvious competing explanation.** Its FP32
  engine matched its PyTorch checkpoint to 8e-6 at P1, and its per-group confidences track RTMW's
  closely (body 0.627 vs 0.715 mean, face 100 % vs 100 % above threshold, hands 98.4/99.1 % vs
  99.9/99.9 %). The one striking difference, 0.3 % vs 16.9 % of foot keypoints above threshold, is both
  models failing on feet with RTMPose-x landing just under the 0.3 gate and RTMW just over it (means
  0.136 vs 0.208); feet are outside the 69 keypoints Uni-Sign consumes. So RTMPose-x is extracting
  correctly and is simply off-distribution for this checkpoint.
- **Front-end: RTMW-l-m FP16, provisionally, and mostly on cost.** It is 1.8× faster and 2× more
  energy-efficient than RTMPose-x FP16 mixed on the board (19.7 ms / 127 mJ vs 35.0 ms / 264 mJ, P2),
  so it wins even at equal accuracy; the accuracy claim above only has to survive *not being much
  worse*. The FP16 half of the decision is independent of all this and is solid, because 5/5 identical
  predictions is a within-extractor comparison. **To settle the extractor question**, re-run this table
  over ~25 more clips: the authors' poses for all 976 test clips are already in
  `data/openasl_test_pose/`, so only the videos need fetching (`data/openasl_fetch.py`) plus one board
  pass. 1/5 vs 0/5 ceiling matches cannot carry a front-end decision in the report.
- **There is still a real gap to the ceiling: 6.3 BLEU-4 / 2.8 ROUGE-L, with only 1/5 predictions
  shared.** Two candidate causes, not yet separated. (a) Crop recipe: P3 crops the signer bbox at
  native resolution, the authors square/pad/resize to 224, and our confident keypoints consequently
  span ~0.01–0.99 of the frame where theirs span ~0.06–0.89 — a normalisation shift the frozen encoder
  never saw. (b) Extractor identity: RTMW-l-m is 256×192 but is not the same network as their
  `pose-rtmpose-192`. (a) is testable without any new model by re-cropping the five clips the authors'
  way and re-running this table; that is the highest-value next experiment on the pose track, because
  a normalisation fix would lift every config at once.
- Qualitatively all five configs are fluent and largely wrong, matching the pattern noted at B1.4. On
  clip 0 RTMW reproduces the ceiling sentence verbatim ("Meteorologists say some areas could see heavy
  rain until Friday afternoon."), while RTMPose-x invents "up to a foot of flooding until Wednesday
  evening". On clip 2 every config fails: the reference is a cave rescue and RTMW emits "a shark attack
  took place on Haiti". 22 BLEU is the state of the art here, so this is expected, not a bug.

**Board note.** The first attempt failed five times with `NvMapMemAllocInternalTagged: error 12` (ENOMEM)
inside `MT5ForConditionalGeneration.from_pretrained`, because nvmap allocates only from `MemFree` and
never reclaims page cache: 3.1 GB was sitting in `Cached`. `python3 jetson/drop_file_cache.py` before
each eval (MemFree 2.8 → 5.0 GB) fixes it. This is the same failure mode as the engine builds in J4,
and the rule is now general: **drop the page cache before loading any large model on the board**, not
just before TensorRT builds. Separately, `eval_openasl.py` needs `portalocker` and `rouge`, which the
image lacked; both are now in `jetson/Dockerfile`.

### 2.5c Pose→BLEU at n=30, with CIs: the extractor question answered (2026-09-26)

30 clips from 30 distinct YouTube videos, 7299 frames, released Uni-Sign checkpoint, beam 4, board
`jetson-lpcv-03` mode 0. All rows scored on **the same 30 clips** (the ceiling's raw file says 16.79 at
n=40 because `data/openasl_pose` holds 40 reference poses; restricted to the shared 30 it is 18.17 —
**do not quote the n=40 figure against these**). `sqnorm` = our own keypoints re-expressed in the
authors' square normalisation frame by `common/renorm_to_openasl.py`; extractor, precision, frames and
checkpoint all held fixed, so a delta there is the coordinate frame alone.

| config | n | BLEU-4 | ROUGE-L |
|---|---:|---:|---:|
| authors' poses (ceiling) | 30 | 18.17 | **45.03** |
| **RTMW FP16 + frame fix** | 30 | **18.52** | 43.91 |
| RTMW FP32 + frame fix | 30 | 17.52 | 42.73 |
| RTMW FP32 | 30 | 16.55 | 40.72 |
| RTMW FP16 | 30 | 16.42 | 40.76 |
| RTMPose-x FP32 | 30 | 9.42 | 32.99 |
| RTMPose-x FP16 mixed | 30 | 8.13 | 32.49 |

Paired clip-level bootstrap, 2000 resamples, aligned by clip name (`unisign/bootstrap_ci.py`):

| comparison | BLEU-4 delta | 95 % CI | ROUGE-L delta | 95 % CI |
|---|---:|---|---:|---|
| RTMW FP32 → **RTMPose-x FP32** | **−7.13** | **[−10.94, −3.07]** | **−7.73** | **[−12.10, −3.57]** |
| RTMW FP32 → RTMW FP16 | −0.13 | [−1.48, +1.21] | +0.04 | [−1.03, +1.22] |
| RTMPose-x FP32 → FP16 mixed | −1.29 | [−3.73, +0.37] | −0.49 | [−3.02, +2.15] |
| RTMW FP32 → ceiling | +1.61 | [−2.57, +5.94] | **+4.32** | **[+0.47, +8.30]** |
| RTMW FP32 → **+ frame fix** | +0.97 | [−2.62, +4.64] | +2.01 | [−1.81, +5.74] |
| RTMW FP32 + frame fix → ceiling | +0.65 | [−3.29, +4.45] | +2.31 | [−1.51, +6.09] |
| RTMW FP16 + frame fix → ceiling | −0.35 | [−4.23, +3.17] | +1.13 | [−2.25, +4.57] |

**1. The extractor question is answered, and it reverses the worry.** The paper specifies RTMPose-x and
we chose RTMW on cost, which was flagged as a risk to accuracy. On 30 clips RTMW beats RTMPose-x by
**7.13 BLEU-4 and 7.73 ROUGE-L, sign established on both metrics**. The front-end choice is no longer
provisional or cost-led: it is the accuracy-correct choice on our pipeline too. Note this does *not*
contradict the paper — their RTMPose-x ran at 192 width on their own square crops, ours at 384×288 on
native crops; what it establishes is that within our pipeline RTMW is much the better of the two.

**2. FP16 is free, and this replaces the retracted claim properly.** §2.5b retracted "RTMW FP16 is
translation-neutral (5/5 identical)" as a five-sentence coin flip. The properly powered version: at
n=30, FP16 vs FP32 is −0.13 BLEU-4 (CI [−1.48, +1.21]) and +0.04 ROUGE-L (CI [−1.03, +1.22]) — no
measured difference on either metric. Under the frame fix the point estimate flips sign (+1.00 BLEU-4,
CI [−0.05, +2.57]), which is what "no real effect" looks like when measured twice.

**3. The gap to the ceiling is small, and the n=5 figure was a subset artifact.** At n=5 our poses
scored 9.89 against a 16.23 ceiling — a 6.33 BLEU-4 gap that I described as the project's largest
accuracy risk. At n=30 the same comparison is **+1.61 BLEU-4, not established** (CI spans zero), with
the ROUGE-L gap of +4.32 the only established one. Those five clips were unusually hard for our poses.

**4. The coordinate-frame fix closes most of what remains.** Re-expressing our keypoints in the
authors' square frame (§2.5d for the mechanism) moves both metrics toward the ceiling and drops the gap
from +1.61 / +4.32 to **+0.65 / +2.31** — and the ROUGE-L gap, the *only* established gap in the whole
table, becomes unestablished. The best configuration, RTMW FP16 + frame fix, is statistically
indistinguishable from the authors' own poses on both metrics (BLEU-4 −0.35, ROUGE-L +1.13).

**What is not established:** the frame fix's own effect (+0.97 BLEU-4, CI [−2.62, +4.64]). n=30 cannot
resolve a ~1-point effect. The direction is consistent across all four of its comparisons and the
mechanism is measured independently (§2.5d, 24× reduction in body-keypoint disagreement), but the
effect size needs the 100-clip run. **Do not report the frame fix as a measured BLEU gain yet.**

Raw: `results/eval_30clip_*.json` (7 evals), `results/ci_30clip_*.json`, board logs
`results/logs/p5_all.log`, `results/logs/final_30clip_summary.txt`.

### 2.5f The deployable config had never had its accuracy measured (2026-10-03)

`jetson/p9_deployable_eval.sh`, raw `results/eval_30clip_pruned_{fps24,src}_bs1.json`.

Every board eval above uses the **full released checkpoint**. The model we intend to ship is the
**pruned** one — §5.4 showed it is −12% system energy, half the peak memory, half the load time, and
the only one that runs `source × beam 4` at all. So until today "our accuracy on the deployable model"
was not a measurement: it was the n=976 decode-config number on the *authors'* keypoints composed with
the n=30 pose-substitution term, and nobody had run the pruned checkpoint on our own keypoints.

Same 30 clips, same pose pkls as §2.5c, pruned checkpoint with the pre-pruned mT5 directory, beam 4,
`max_new_tokens` 64, **batch size 1** (held fixed because §5.1 found 2 of 30 clips change output
between batch 8 and batch 1).

| config | poses | ckpt | n | BLEU-4 | ROUGE-L |
|---|---|---|---:|---:|---:|
| **deployable: 24 fps** | ours, RTMW FP16 + frame fix | **pruned** | 30 | **19.70** | **46.48** |
| 24 fps | ours, RTMW FP16 + frame fix | full | 30 | 20.15 | 46.31 |
| source rate | ours, RTMW FP16 + frame fix | **pruned** | 30 | 17.73 | 42.85 |
| source rate | ours, RTMW FP16 + frame fix | full | 30 | 18.64 | 44.19 |

**Pruning on our own keypoints, paired bootstrap at 24 fps (2000 resamples):**
**−0.46 BLEU-4 [−3.82, +2.03]**, ROUGE-L **+0.17 [−1.86, +2.14]** — not established, and consistent
with the −0.28 [−0.63, +0.05] measured on the authors' keypoints at n=976 (L5.3). Pruning does not
appear to cost anything extra when the poses come from our own extractor, which is the question this
run existed to ask.

**Do not read the 24 fps vs source rows as a frame-rate result.** +1.97 BLEU-4 is the same artefact
§2.9B already warns about: at n=30, 24 fps scoring above source rate is noise, and the n=976
measurement puts the true effect at −0.07 [−0.53, +0.39]. The rows are here to hold the rate fixed
while the checkpoint changes, nothing more.

**The honest summary of deployable accuracy as of this run.** The point estimate is ~22.8 BLEU-4 /
~43.1 ROUGE-L, composed from the decode config at n=976 on the authors' keypoints (22.80 / 43.13,
L13) and a pose-substitution term of −0.35 [−4.23, +3.17] at n=30 (§2.5c). **The composition is
dominated by that n=30 interval: ±4 BLEU-4.** The absolute 30-clip figures above (17.7–20.2) are a
small, harder subset whose own ceiling is 18.17 and must never be compared across clip sets. Narrowing
this is what the n=100 pass (§2.5g) is for.

### 2.5g n=100 board pass: the deployable config, measured rather than composed (2026-10-03)

`jetson/p10_n100.sh`, raw `results/eval_n100_*.json`, CIs `results/ci_n100_*.json`.

**100 clips from 100 distinct videos** — one clip per video, so no signer is counted twice. Our own
keypoints extracted on the board (RTMW-l-m FP16 + frame fix, 24,365 frames at **36.0 frames/s**,
11.3 min), and the authors' released reference poses **restricted to the same 100 clips**. That
restriction is the point: §2.5c records us quoting a ceiling at n=40 against our rows at n=30, and
this run must not repeat it. Decode held at the deployable settings throughout — beam 4,
`max_new_tokens` 64, **batch size 1** (§5.1: 2 of 30 clips change output between batch 8 and 1).

| row | poses | ckpt | rate | BLEU-4 | ROUGE-L |
|---|---|---|---|---:|---:|
| **the deployable config** | **ours** | **pruned** | **24 fps** | **22.99** | **43.43** |
| | ours | pruned | source | 22.94 | 43.26 |
| | ours | full | 24 fps | 23.44 | 43.84 |
| ceiling | authors' | pruned | 24 fps | 24.89 | 43.94 |
| ceiling | authors' | pruned | source | 25.56 | 45.24 |

**The headline is that the first row is a measurement.** Board pose extraction, pruned checkpoint,
24 fps, beam 4 — the configuration we intend to ship, scored end to end with nothing borrowed. Until
today its accuracy was a *composition* of the n=976 decode figure on the authors' keypoints with an
n=30 pose-substitution term (§2.5f). The composed estimate was ~22.8 BLEU-4 and the measurement is
**22.99**, which is the reassuring part.

Paired clip-level bootstrap, 2000 resamples, aligned by clip name:

| comparison | BLEU-4 | 95 % CI | ROUGE-L | 95 % CI |
|---|---:|---|---:|---|
| **24 fps → source, our poses** | **+0.05** | **[−1.11, +1.03]** | +0.17 | [−1.23, +1.64] |
| pruning (full → pruned), our poses @ 24 fps | −0.44 | [−1.87, +0.94] | −0.42 | [−1.49, +0.59] |
| pose substitution (authors' → ours) @ 24 fps | **−1.89** | [−4.26, +0.27] | −0.51 | [−3.49, +2.80] |
| pose substitution (authors' → ours) @ source | **−2.62** | [−5.38, +0.35] | −1.98 | [−4.86, +1.09] |

**1. 24 fps is free on our own keypoints, and this is the cleanest evidence for the operating point
yet.** +0.05 BLEU-4 [−1.11, +1.03] is a tight null, and it independently reproduces the n=976
authors'-poses result of −0.07 [−0.53, +0.39] on a different pose source. It also retires the n=30
artefact: §2.9B measured +1.97 BLEU-4 for 24 fps over source and warned it was noise; at n=100 the
effect is +0.05. **The warning was correct and is now quantified.**

**2. Pruning costs about −0.4 and is not established**, now with a usable interval: −0.44
[−1.87, +0.94] against §2.5f's −0.46 [−3.82, +2.03] at n=30 — same point estimate, interval less than
half as wide — and consistent with −0.28 [−0.63, +0.05] on the authors' keypoints at n=976 (L5.3).
Three independent measurements agree that pruning is cheap. Combined with §5.4's −12 % system energy,
half the peak memory and half the load time, **the pruned checkpoint is the right deployment choice on
every axis we measure.**

**3. The pose-substitution term moved against us, and §2.5f's reading of it must be revised.** At n=30
it was −0.35 [−4.23, +3.17] and the pose track wrote that our extractor costs nothing detectable. At
n=100 the point estimate is **−1.89 at 24 fps and −2.62 at source**, with upper bounds of +0.27 and
+0.35 — still not established, but only just, and both intervals now sit almost entirely below zero.

**The correct statement is that our extractor plausibly costs ~2 BLEU-4 against the authors'
keypoints and n=100 still cannot resolve it.** The n=30 figure was not wrong, it was uninformative,
and a point estimate near zero inside a ±4 interval was read as evidence of no effect. **That is the
same error this document already records twice** — the 16 fps "free" claim in §2.9 and the ceiling-gap
retraction in §2.5b. Note too that ROUGE-L is *not* carrying this one (−0.51 and −1.98, both wide),
so the metric-power argument does not rescue it either.

**Consequence for the frontier.** L16 addendum 3 states the deployed system's absolute accuracy using
the n=30 term. That addendum's reasoning is right and its number is now stale: the deployed absolute
column is better read as **~2 BLEU-4 below** the plotted cells, not ~0.35, with the caveat that the
term is unestablished. Relative ordering is unaffected, for the reason given there — every cell shares
one pose source.

**4. One marginally "established" result that must NOT be promoted.** The fifth comparison — 24 fps
vs source on the **authors'** poses at n=100 — came out at BLEU-4 −0.67 [−2.24, +1.29] and ROUGE-L
**−1.30 [−2.71, −0.00]**, which the CI script flags as sign established.

It should not be read as one, for two reasons.

* **A better-powered measurement of the identical comparison disagrees.** The same pose source, same
  pruned checkpoint, same beam 4 at n=976 gives ROUGE-L **+0.15** (42.98 → 43.13, L13/frontier) and
  BLEU-4 −0.07. n=976 governs over n=100 on the same quantity, so the −1.30 here is a subset
  fluctuation in these particular 100 clips, not a frame-rate effect.
* **The bound is −0.00.** "Established" at an upper bound that rounds to zero is a threshold artefact,
  and this document's own rule is that a marginal P-value straddling the line must not be quoted as
  either established or null (§2.9B, L15.3).

Recorded rather than dropped, because the asymmetry is real and might matter later: on *our* poses the
same comparison is a null (+0.17 [−1.23, +1.64]) while on *theirs* it is −1.30. A tempting story is
that our keypoints are already noisy enough that thinning them adds relatively little — but with these
intervals that is speculation, and it is written here as speculation.

**What would settle it:** the remaining 831 test clips. The interval narrowed from ±3.7 to ±2.3
BLEU-4 going from n=30 to n=100, i.e. roughly as 1/√n, so n≈400 would bring it to about ±1.2 and
n=931 to ±0.75 — enough to establish or kill a 2-point effect. The frames are on the Mac and the
pipeline is the one used here; it is ~9 rounds of push, extract, pull, delete at 11 min of extraction
per 100 clips.

**Board housekeeping.** The 70 newly pushed clips' frames were deleted after verifying both pkl
normalisations existed for each, returning `data/clips` to 251 MB. The original 30 clips' frames are
kept deliberately: they include `ixq65EiuJ_c-00:03:47.633-00:03:56.133`, the end-to-end clip of
§5.1/§5.3/§5.4.

### 2.5d Why our poses differed: a 1.68x coordinate-frame mismatch (2026-09-26)

Diffed our 30-clip poses against the authors' released poses in the units the frozen encoder consumes
(`common/pose_to_unisign.py`: 9 body joints absolute, 21+21 hands wrist-relative, 18 face
nose-tip-relative, anything under score 0.3 zeroed).

| group | frame | median difference, ours vs theirs |
|---|---|---:|
| body | **absolute** | **0.1175** |
| left hand | wrist-relative | 0.0509 |
| right hand | wrist-relative | 0.0572 |
| face | nose-tip-relative | 0.0298 |

Body is the only absolute group and the worst by 2–4×. That is the signature of a **global scale
error**: subtracting a root cancels it for the other three.

**Cause.** The authors normalise over a square — bbox expanded on its *short* side, black-padded where
it leaves the frame, resized to 224. We normalise over the bbox crop. Signer boxes are tall and narrow,
so their square side is much wider than our crop width and the signer fills less of their frame.
Predicted `square_side / our_crop_width` against measured shoulder-width ratio, 30 clips:
**correlation 0.968**, mean abs error 0.028 (predicted 1.657 ± 0.070, measured 1.679 ± 0.083).

**Ruled out along the way:**
- **Confidence gating.** We zero 2.8 % of consumed joints, they zero 2.8 %; every group within 0.1 pp;
  frames with a whole hand gated off 8.3 % vs 8.3 %. The THR=0.3 interaction is not involved.
- **A translation component.** Removing a per-clip mean shift accounts for 2 % of the difference.
- **Detector quality**, mostly: the residual in root-relative units is 0.03–0.05.

**Fix, and it needs no re-extraction.** `common/renorm_to_openasl.py` maps our normalised coordinates
into their square frame by exact arithmetic from geometry already in each clip's `meta.json`:

| | shoulder ratio ours/theirs | body median difference |
|---|---:|---:|
| our crop frame | 1.679 ± 0.083 | 0.1328 |
| **their square frame** | **1.014 ± 0.014** | **0.0055** |

A 24× reduction. What remains is genuine detector disagreement — our detector ran on different pixels
(native 502–754 px crops vs their 224 square with 15–50 % black padding), and renormalising coordinates
cannot change what the detector saw. See §2.5c for the BLEU effect.

**This corrects §2.5b.** That section reported the crop hypothesis as "tested, not supported" and
retracted it, judged on five clips where BLEU-4 rose 9.89 → 12.38 against a 16.23 ceiling while ROUGE-L
fell. The direction was right; the sample could not support the call, and a real signal was read as
noise.

**Deployment consequence:** the transform belongs in `common/pose_to_unisign.py`, in the live path, not
only in an offline script — otherwise the demo reproduces the bug the offline results just fixed.

### 2.5b The crop experiment: hypothesis not supported, and a result of mine retracted (2026-09-26)

> **SUPERSEDED by §2.5c and §2.5d (same day, n=30).** Both conclusions in this section were wrong
> in the same way -- drawn from five clips. The crop/normalisation direction *was* right (§2.5d
> measures the mechanism: a 1.68x coordinate-frame mismatch), and FP16 neutrality is real but was
> established properly only at n=30 (§2.5c). Kept in place because the reasoning error is the
> lesson: five sentences cannot resolve a one-to-two point BLEU effect, in either direction.

§2.5 blamed the 6.3 BLEU-4 gap to the ceiling on our crop convention. Tested directly:
`data/openasl_fetch.py --crop-style openasl` re-fetched the *same five clips* using OpenASL's own
`prep/crop_video.py` recipe (square the bbox, black-pad outside the frame, resize to 224), and all four
engines re-ran over them. `data/verify_openasl_crop.py` first confirmed the geometry reproduces theirs:
their confident keypoints fall inside the real-pixel region of our square on all five clips, with the
y-maxima tracking the predicted content edge to 0.01–0.03.

**Their frames are 15–50 % black padding.** The OpenASL bboxes extend well outside the frame
(`UoU3ZSuTef4`: y from −141 to 1082 in a 720-tall frame), so squaring pads heavily. In their normalised
space the signer never reaches the frame edge; our native crop fills it. Confirmed by the re-crop: our
confident spans moved from ~0.01–0.99 to ~0.14–0.88, against the authors' ~0.10–0.91.

| poses | BLEU-4 | ROUGE-L | preds = ceiling | preds = own FP32 |
|---|---:|---:|---:|---:|
| ceiling (authors' poses) | 16.23 | 46.17 | 5/5 | — |
| RTMW FP32, **native** crop | 9.89 | **43.39** | 1/5 | — |
| RTMW FP32, **openasl** crop | **12.38** | 38.87 | 1/5 | — |
| RTMW FP16, native crop | 9.89 | 43.39 | 1/5 | **5/5** |
| RTMW FP16, openasl crop | 8.64 | 36.77 | 0/5 | **3/5** |
| RTMPose-x FP32, native / openasl | 8.96 / 5.95 | 31.94 / 30.37 | 0/5 / 0/5 | — |
| RTMPose-x FP16 mixed, native / openasl | 9.39 / 6.21 | 33.76 / 30.22 | 0/5 / 0/5 | 3/5 / 3/5 |

- **The crop hypothesis is not supported.** Matching their crop did not move us toward the ceiling.
  BLEU-4 rose for RTMW FP32 (9.89 → 12.38) while ROUGE-L *fell* (43.39 → 38.87), the two metrics
  disagree in direction, and the only discrete measure — predictions identical to the ceiling — stayed
  at **1/5**. RTMPose-x got worse on both metrics. Changing crop changes almost every sentence
  (native vs openasl: 0–1 of 5 identical per engine), so the effect is large but not toward the target.
- **This makes extractor identity the leading explanation for the ceiling gap.** Note the direction: the
  authors' 224 crop *discards* resolution, since our native crops are 502–754 px wide. Our native-crop
  pose input is strictly better-resolved, and their poses still translate better. So the ceiling
  advantage is not crop convention and not pose sharpness; the remaining candidate is that their
  specific network places keypoints differently (joint centre conventions, hand-model priors) and the
  frozen ST-GCN learned *those* placements. That is not fixable by re-cropping and would require either
  their extractor or fine-tuning.
- **Retracted: "RTMW FP16 is translation-neutral (5/5 identical)" does not survive.** On the openasl
  crop the same engine pair gives **3/5**, i.e. two of five sentences change. Crucially this is *not* a
  resolution-dependent FP16 effect — measured in simcc bins (the only cross-crop-valid unit, see the P4
  bin note), agreement is if anything *better* at 224:

| crop | 1 bin | RTMW FP16 ≤1 bin | ≤2 bins | ≤4 bins | worst |
|---|---:|---:|---:|---:|---:|
| native (502–754 px) | 1.71 px | 86.95 % | 99.39 % | 99.95 % | 53 bins |
| openasl (224 px) | 0.72 px | 88.17 % | **99.76 %** | 99.98 % | 44 bins |

  Keypoint agreement is essentially unchanged, so the jump from 5/5 to 3/5 sentences is **the fragility
  of a five-sentence sample, not a real effect**. The 5/5 was a favourable coin flip. What is robust is
  the keypoint agreement: **99.4–99.8 % of consumed keypoints within 2 bins of FP32 on both crops.**
  Sentence-level neutrality is unproven at n=5 and must be re-tested on ~25 clips before any report
  repeats it.
- **One ordering does reproduce across crops, which is worth having:** RTMW agrees with its own FP32
  better than RTMPose-x mixed does, in bins, on both crops (≤2 bins: 99.39 % vs 96.58 % native;
  99.76 % vs 98.72 % openasl; worst 53 vs 147 bins native, 44 vs 77 openasl). Two independent crops
  giving the same ordering is much stronger than the single-crop comparison in §2.5.
- **Method note for the report.** This experiment is the clearest demonstration of why pixel thresholds
  are unusable here. The same engine pair reads 99.81 % (≤5 px) at native crop and 99.99 % at 224,
  which would suggest the 224 crop is numerically safer — but 5 px is ~2 bins at native and ~7 bins at
  224. In bins the two are within 0.4 pp. **Any agreement threshold must be stated in bins.**

**Second-board reproduction and the 15W/25W discrepancy.** The same ONNX, the same script and the same
TensorRT 10.11 on `jetson-lpcv-03` give 66.7 ms TRT time vs Atisri's 44.4 ms, at 8.2 W vs 11.6 W. The
CSV shows the lpcv-03 GPU pinned at 612 MHz whenever it was busy, which is the nvpmodel 15W cap.
Scaling 66.7 ms by 612/918 gives 44.5 ms, i.e. Atisri's number to within 0.1 ms, and the 25W mode is
the conf's default. The Phase-1 row was therefore almost certainly taken with the GPU at the 25W cap
(918 MHz), whatever `/var/lib/nvpmodel/status` said at the time (the file can be stale, and nothing
logged the clock). Energy per frame barely moves (754 vs 733 mJ): the slower clock draws
proportionally less power, so **latency rows are not comparable across power modes but energy rows
nearly are.** From here on the lpcv-03 mode-0 row is the FP32 baseline for every ratio in this
document, and no row without a logged `gpu_MHz` is used for a latency comparison.

Reading it:
- At the true 15W cap: 79 ms/frame = ~12.6 fps of pose-stage compute (11.5 fps on the harness wall
  clock), 66.7 ms of it TensorRT; source video is 30 fps, so FP32 RTMPose-x cannot keep up without
  dropping frames. That is the motivation for every knob that follows. (At 918 MHz it was 56 ms =
  ~18 fps, still short.)
- Preprocess (affine warp + normalise, CPU) is 13% of frame time at 15W. **Two corrections to what
  this line used to say**: JPEG decode is *not* inside it (`cv2.imread` sits outside the timed window
  — see §2.2b), and it *does* shrink with GPU quantization, by 22 % on RTMW, because a cheaper engine
  leaves the CPU more of the 15 W budget to clock up with.
- Peak `VDD_IN` 9.1 W at 15W mode (13.3 W in the 918 MHz run). TRT latency jitter <1 ms, Tj 53 C, so no throttling.
- One 10 s sentence costs 299 x 0.733 J = 219 J of pose extraction at FP32.

First pass only: single run per cell. Final table needs 3 runs, mean+std, several clips/signers, 30 min sustained.

## 2.9 / P8 + L7: energy and accuracy versus capture rate (2026-09-28)

`jetson/p8_l7_sweep.sh`, board `jetson-lpcv-03` mode 0 (15 W), one other user present with an idle shell
(no process above 1 % CPU, GPU load 0). Three parts: real pose runs at each rate for energy, accuracy at
the same rates from subsampled keypoints, and the LM's own latency/energy matrix.

### A. Pose stage vs capture rate (real runs, 3 repeats)

| fps | frames | ms/frame | mJ/frame | dyn mJ | avg W | gpu MHz | cpu MHz | **J per second of video** | vs 30 fps |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 255 | 19.82 | 141.6 | 40.1 | 4.88 | 309 | 897 | **4.25** | 1.00× |
| 24 | 204 | 21.29 | 139.9 | 33.6 | 4.88 | 308 | 868 | **3.36** | 0.79× |
| 16 | 136 | 20.68 | 141.8 | 39.1 | 4.85 | 312 | 882 | **2.27** | 0.53× |
| 12 | 102 | 20.54 | 139.5 | 39.1 | 4.90 | 313 | 901 | **1.67** | 0.39× |
| 8 | 68 | 19.99 | 145.9 | 40.1 | 4.87 | 318 | 1039 | **1.17** | 0.27× |

`mJ/frame` is flat by construction — every frame costs the same to process. What scales is how many
frames exist per second of video, so **J per second of video is the only comparable column**. The CPU
clock rising at low rates (897 → 1039 MHz) is the 15 W coupling of §2.2b again: less sustained GPU work
leaves the CPU more budget.

**Do not compare the absolute 141.6 mJ/frame with P2's 127.** Each of the 3 repeats pays 20 warm-up
inferences and this clip has 255 frames against P2's 299, so the warm-up amortises differently. The
ratios across rates are what P8 needs and they are unaffected.

### B. Accuracy vs capture rate (30 clips, beam 4, batch 1, released checkpoint)

> **SUPERSEDED 2026-09-30 by the LM track at n=976/967** (`lm-track-report-2026-09-30.md` §3.6, §7.1).
> The conclusion below — "16 fps at no measured accuracy cost" — was **n=30 with a CI ~6 BLEU-4 wide**
> and does not survive. At n=976 the 30→16 fps cost is **BLEU-4 −1.33 [−2.00, −0.64], ROUGE-L −1.63**;
> at n=967 on dev, ROUGE-L **−1.34 [−2.42, −0.30]** against BLEU-4 **−0.35 [−1.00, +0.45]** on the
> *same* clips and the *same* resample draws (`unisign/adapt_ci.py`, leak-free checkpoint, paired
> bootstrap — the metric-power argument demonstrated rather than asserted). **Independently reproduced
> on the Mac at four seeds** (`results/block4/adapt_ci_mac_repro_4seeds.json`): point estimates identical,
> CI bounds stable to ±0.06, and ROUGE-L P(delta<0) = 0.991–0.998. The bounds our run and theirs report
> differ by ~0.03, which is inside that seed-to-seed spread, so **n=1000 resamples is already adequate**
> — the CI width is set by the 967 clips, not by the number of draws. ROUGE-L replicates on both
> splits and excludes zero; BLEU-4 does not, because a
> handful of sentences flipping one 4-gram match is the whole effect at n≈1000. **Effects below ~1
> BLEU-4 at n≈1000 must be carried by ROUGE-L.** At n=300 the same model scored +1.10 BLEU-4 *higher*
> at 16 fps than at source — the opposite sign — so n=30 was far inside the unreliable regime.
>
> **Two separate errors of mine here.** First, "not established" meant *unknown* and I wrote it up as
> "no cost". Second, a real bug: OpenASL is **not one frame rate** — of our 931 clips, 76.5 % are
> ~30 fps, 22.2 % ~24, 7 are 59.94 — and `subsample_pkl.py` / `--keep-fps` hardcoded `src_fps=29.97`,
> so a 24 fps clip labelled "16 fps" was really subsampled to ~12.8. **6 of these 30 clips were
> off-rate.** Fixed by deriving each clip's rate from frames ÷ duration.
>
> **One caution on the adaptation half of that 2x2.** The same four-seed run puts adaptation's gain at
> *source* rate at ROUGE-L **+0.60**, with P(delta<0) spanning **0.044–0.056** across seeds — it sits
> exactly on the 0.05 line, so "established" flips with the resample draw. It must not be quoted as
> either established or null. Whether adaptation specifically repairs frame-rate shift, as opposed to
> helping everywhere, is the difference-in-differences that `adapt_ci.py` computes and that is still
> waiting on one eval JSON (`adapt_fps16_s42_cap100`, quoted in L15.3 but not yet pushed).
>
> **Revised operating point: 24 fps, not 16.** Free on both metrics (test −0.07 BLEU-4, dev +0.26;
> ROUGE-L +0.15) for 21 % less pose energy. 16 fps buys 47 % but costs ~1.3–1.6 ROUGE-L. **Table A
> (energy) is unaffected** — it is one clip at 29.97 fps, so the ratios hold.


Poses subsampled from the full-rate keypoints. **This is exact, not an approximation**: pose extraction
is per-frame independent, so a frame kept at a reduced rate receives exactly the keypoints it would have
received at 30 fps (`common/subsample_pkl.py`). Confirmed by the 30 fps row reproducing the unsubsampled
batch-1 eval to two decimals. Paired bootstrap against 30 fps, 2000 resamples:

| fps | BLEU-4 | ROUGE-L | Δ BLEU-4 | 95 % CI | verdict |
|---:|---:|---:|---:|---|---|
| 30 | 18.64 | 44.19 | — | — | baseline |
| 24 | 20.15 | 46.31 | +1.51 | [−2.32, +5.78] | no measured cost |
| **16** | 18.49 | 43.56 | −0.15 | [−3.20, +2.89] | **no measured cost** |
| 12 | 15.39 | 43.58 | −3.25 | [−7.22, +0.80] | ambiguous |
| 8 | 11.46 | 36.28 | −7.18 | **[−12.55, −2.13]** | **established loss** |

~~**16 fps is the operating point: 47 % less pose energy at no measured accuracy cost.**~~ **RETRACTED
— see the note above; 24 fps is the free operating point.** 8 fps is an established loss even at n=30,
so there is a floor. 12 fps should not be used — the point estimate is a meaningful
−3.25 and n=30 cannot resolve it, so "ambiguous" here means *unknown*, not *free*. **24 fps scoring
above 30 fps is noise and must not be reported as an improvement.**

These rows are **un-adapted**: the frozen encoder never saw a reduced rate. They are the baseline any
C8 adaptation gain is measured against, and they say an adaptation run at 24 fps has nothing to recover.

### C. LM latency and energy: beam width × encoder length (pruned checkpoint)

One process for all 15 configurations (model load is ~64 s), each with its own power window.

| beams | T requested | frames used | total ms | **encoder ms** | **decoder ms** | J/sentence | dyn J | avg W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 215 | 1372 | **49** | **1228** | 8.88 | 3.62 | 6.48 |
| 2 | 256 | 215 | 1582 | 50 | 1447 | 10.26 | 4.30 | 6.49 |
| 4 | 256 | 215 | 1676 | 49 | 1538 | **11.04** | 4.78 | 6.62 |
| 1 | 137 | 137 | 1351 | 39 | 1236 | 8.55 | 3.49 | 6.35 |
| 4 | 137 | 137 | 1574 | 41 | 1459 | 9.89 | 3.99 | 6.30 |
| 1 | 68 | 68 | 1264 | 41 | 1168 | **7.79** | 3.08 | 6.19 |
| 4 | 68 | 68 | 1465 | 41 | 1370 | 8.94 | 3.47 | 6.14 |

Full matrix in `results/lm_sweep_pruned.json`.

**Encoder length is nearly useless as a lever.** The encoder costs **39–50 ms** across the whole range
while the decoder costs **1062–1538 ms**. Cutting T from 256 to 68 — 73 % fewer pose frames — saves the
LM only 8.88 → 7.79 J, **12 %**, because the decoder dominates and its cost tracks tokens generated
rather than input length. **Reduced frame rate is a pose-stage saving, not an LM saving**, and a report
that presents it as the latter is claiming ~12 % where the pose side delivers 47–73 %.

**Beam width is the LM's real lever — but the worst SYSTEM lever.** Greedy 8.88 J vs beam 4 11.04 J
— 2.16 J, 24 % more *LM* energy — against 2.00 BLEU-4 (CI [−2.63, −1.41], n=976, §L6.1). **Corrected
2026-09-30:** because the LM is only ~25 % of system energy, that is just 5 % of system joules for
2 BLEU-4, i.e. ~1.1 J per BLEU-4 point, against **103.5** for 30→24 fps. Frame rate is 10–100× the more
efficient lever at the system level, and calling beam width "the frontier" was true only of the LM in
isolation. That single trade is the *LM's* accuracy–energy tradeoff and it can be plotted without INT8, which is
why INT8 was dropped (guide row 3.2).

### What this means for the system

For the M1 sentence (8.51 s of video, measured 50.37 J total): pose 4.25 × 8.51 = 36.2 J plus LM
11.04 J at 30 fps / beam 4. Moving to **24 fps** gives ~36 J, a **17 % system saving that is free on
both metrics**; 16 fps gives ~29 J, a ~38 % saving, but **costs ~1.3–1.6 ROUGE-L** (corrected
2026-09-30 — the earlier "at no measured accuracy cost" was an n=30 artifact). **Caveat on the absolute
joules:** this clip's 644×720 crop is at the **83rd percentile** of crop area across the 931 clips
(9k–770k px, median 395k). ~11.1 of the 25.1 ms/frame scales with crop area, so a median clip is ~7 %
cheaper and a p10 clip ~27 % cheaper. Ratios across rates hold; the absolute J is clip-specific. Going further to greedy adds
another ~1.3 J of saving for a 2.00 BLEU-4 loss, which is the trade to present rather than to take
silently.

Raw: `results/p8_pose_fps*.json`/`.csv`, `results/eval_30clip_fps*.json`, `results/ci_fps30_vs_fps*.json`,
`results/lm_sweep_pruned.json`, board summary `results/logs/p8_summary.txt` (its CI section is empty —
the board had a stale `bootstrap_ci.py` without `--out`; the CIs above were computed on the Mac).

## 5.1 / M1: first on-device end-to-end translation (2026-09-28)

`unisign/e2e_translate.py` — pose TensorRT engine and the language model in **one process** on
`jetson-lpcv-03`, mode 0 (15 W). Clip `ixq65EiuJ_c-00:03:47.633-00:03:56.133`, 255 frames (8.51 s of
video), RTMW-l-m FP16, released checkpoint, beam 4, frame-fix (§2.5d) applied, 3 timed sentences after
a warm-up. Board occupancy checked first (`jetson/run.sh whoelse`): one other user with an idle shell,
no foreign process above 1 % CPU, GPU load 0.

| | value |
|---|---|
| output text | "When I first moved to Texas two years ago, it was my first time doing a plane invasion." |
| reference | "When I moved to Texas one year and a half ago, this was my first artwork." |
| **sentence total** | **9052.8 ms ± 335.6** (runs 9527 / 8811 / 8820) |
| ├ pose, 255 frames | 6640.7 ms |
| ├ keypoints → model inputs | 25.7 ms |
| └ language model | 2386.5 ms |
| per frame | imread 6.02 + preprocess 5.11 + TRT 12.71 + post 1.26 = **25.10 ms** |
| **energy** | **50.37 J/sentence** total, **14.68 J/sentence** dynamic |
| peak GPU | 2.577 GB |
| startup (outside the window) | LM load **64.4 s**, pose engine 1.1 s |

**Validated against the offline pipeline.** M1's text initially disagreed with the §2.5c eval for the
same clip, engine and checkpoint. The cause was **batch size**, not the new code path: `eval_openasl.py`
batches 8 clips and pads them, M1 runs one. Re-running the eval at `--batch-size 1` reproduces M1's
sentence exactly, and 2 of the 30 clips differ between batch 8 and batch 1 (BLEU-4 18.52 → 18.64,
ROUGE-L 43.91 → 44.19). So the end-to-end path is correct, **and every offline BLEU number in this
document carries a small batching dependency** — smaller than any effect we reason about, but it should
be stated rather than discovered later.

**It is 1.06× slower than real time, and the LM is why.** 9.05 s to process 8.51 s of video. The pose
stage alone is 6.64 s = **0.78× real time**, comfortably real-time; the LM's 2.39 s per sentence pushes
the total over. Because LM cost is per *sentence* and pose is per *frame*, longer sentences get better
and short ones worse. This is a direct link to the decode-knob frontier (§L6.1): beam 2 costs 0.81
BLEU-4 and greedy 2.00, and either would bring the pipeline under real time.

**The contention I predicted did not appear — because the stages are sequential.** §2.2b found CPU
stages slow by ×1.281 when the GPU takes more of the 15 W budget, so I expected end-to-end to exceed
pose-alone plus LM-alone. It does not: per-frame pose is 25.10 ms here against 25.03 ms measured
standalone (§2.2b), with TRT slightly *faster* (12.71 vs 13.66) and the CPU stages slightly slower.
The reason is that this implementation runs all frames and *then* the LM, so the two never overlap —
having the LM merely resident costs nothing. **A streaming pipeline that decodes sentence N while
extracting frames for N+1 would contend, and that remains unmeasured.** Do not cite this row as
evidence that a streaming design is free.

**FP16 is now a fitting requirement, not only an energy choice.** The same run with the FP32 engine
(159 MB vs 68 MB) fails partway through LM decode, out of memory in the KV-cache concat. End-to-end
fits at FP16 and does not at FP32 on this 8 GB board.

Raw: `results/m1_e2e_rtmw_fp16.json`, `.csv`, log `results/logs/m1_e2e.log`.

### Infrastructure findings from this run

**The NVML assert masks an out-of-memory.** Every failure in this project that printed
`RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1017` was preceded
by `NvMapMemAllocInternalTagged: ... error 12` (ENOMEM). nvmap fails, PyTorch tries to build an
informative OOM message, NVML is only partly supported on Jetson, and the assert replaces the real
error. **Read that assert as "out of memory", not as a PyTorch bug.** This retroactively explains the
five evals lost on 2026-09-26 and the two that died under concurrency.

**`posix_fadvise` is not enough on a shared board.** `jetson/drop_file_cache.py` can only drop pages
backed by files it names, and only when nothing else references them. With another user's desktop
session active, MemFree sat at 100 MB with 6.2 GB cached and fadvise freed 14 MB. Dropping caches
properly needs root, which we do not have. Added `--target-free-mb=N`, which briefly allocates
anonymous memory to make the kernel reclaim page cache, then releases it: MemFree 1555 → 3744 (fadvise)
→ **4618 MB** (reclaim). The first version of this sized the allocation to the *deficit*, which fits in
already-free memory and evicts nothing; it must be sized to the target.

## 5.2 / The LM-load memory ceiling was an occupancy artefact (2026-10-02)

> **This section WITHDRAWS a claim this track made and sent to the LM track.** On 2026-09-29 four
> probes concluded "the LM load alone exhausts the board — memory IS binding at load". The LM track
> adopted it, struck their own "memory is not binding" line, and **partly reopened INT8** on the
> strength of it. It does not replicate.

**The process failure first.** Those probes were run with **no occupancy check**. `jetson/run.sh
whoelse` exists (added 09-26) and was used correctly before M1 — §5.1 records "one other user with an
idle shell, no foreign process above 1 % CPU, GPU load 0" — but there is no occupancy check anywhere
between 09-28T15:55Z and 09-30T19:54Z, which brackets every one of these probes. The board is shared
scratch; an unrecorded co-tenant is exactly the variable that invalidates an allocation measurement.

**Re-run 2026-10-02 on a verified-empty board** (0 users, 0 foreign processes >1 % CPU, no foreign
containers, no `/dev/nvidia*` clients, checked before *and* after every run), `probe2.py` unchanged,
and with **no** cache reclaim at all. Full data in `results/mem_probe_2026-10-02.json`.

| mode | MemFree at start | 2026-09-29 (occupancy unknown) | 2026-10-02 (empty) |
|---|---:|---|---|
| `lm_only` | 3863 → 4034 MB | ❌ fail | ✅ **success**, peak torch 0.98 GB |
| `lm_only`, max reclaim | 4641 MB | ❌ fail | — |
| `lm_only` | 1508–2014 MB | — | ✅ **success ×4** (Cached 3791–4255 MB) |
| `lm_only` | 1084 MB | — | ❌ fail |
| `trt_then_lm` | 3926 → 1258 MB | ❌ fail | ✅ **success** |
| `lm_then_trt` | 4058 → 1470 MB | ❌ fail | ✅ **success** |
| `trt_only` | 3988 → 1283 MB | ✅ success | ✅ success |

**Seven successes at 1508–4034 MB free against one failure at 1084 MB.** The 09-29 failures at
3863–4641 MB are not reproducible on an empty board, and the ordering is impossible to explain with a
MemFree threshold: 09-29 failed at 4641 MB while 10-02 succeeded at 4034 MB and again at 1508 MB.

**Three further claims fall with it:**

1. ~~"The board physically cannot give much more than ~5 GB free."~~ That came from `MemAvailable`
   capping at 5304 MB on 09-29. Empty, it is **6692–6732 MB** — about 1.4 GB higher, which is the
   scale of a co-tenant's resident set.
2. ~~"The pruned load path peaks near 4.2 GB."~~ **Never measured.** It was arithmetic over an assumed
   250k materialisation (2.3 full + 0.95 sliced + 0.95 device). Measured peak torch device allocation
   is **0.98 GB on every run**, and host consumption is ~1.1–1.6 GB of MemFree.
3. ~~"nvmap allocates only from MemFree and never reclaims page cache."~~ `lm_only` succeeds with
   **Cached at 3791–4255 MB and MemFree at 1508 MB**, so the load does not require the cache to be
   evicted first. The `--target-free-mb` reclaim above is still a useful tool; it was not the
   precondition it was described as.

**What is still true, and the limit of this result.** `probe2.py` loads the **pruned** checkpoint
(545 MB on disk, 0.98 GB device peak). The end-to-end and sustained runs load the **full** released
checkpoint (1.2 GB on disk, **2.577 GB** device peak per §5.1), so this probe does not speak for them,
and an earlier draft of this section wrongly said it did. Measured the same day: the 2x2 driver
(§5.3) failed all four configs with the same NVML assert at **1654–1830 MB free on a verifiably empty
board**, and M1 succeeded at 4625 MB after a reclaim. **The full-checkpoint path needs roughly 4 GB of
MemFree; the pruned path needs ~1.5 GB.** Both numbers are now measured rather than assumed.
Re-run with reclaim-to-target, three of those four configs then passed at 5292–5478 MB -- and the
fourth, 255 frames at beam 4, still failed at **5691 MB on an empty board**, although M1 ran that
same cell at 4625 MB. So ~4 GB is the figure for *loading* the full checkpoint, and **MemFree is
evidently not a sufficient readiness check**: §5.3 has the detail, and the protocol here may need a
stronger test than a free-pages threshold.

So the corrected statement is narrower than the withdrawal above might suggest. For the **pruned**
path the 09-29 ceiling does not exist. For the **full** checkpoint memory is genuinely tight, the
reclaim step is mandatory rather than precautionary, and the sustained run's failures are not yet
explained — but they must be re-diagnosed against a reclaim-and-verify protocol, not against the
09-29 numbers.

**Why M1 stood up anyway.** M1's occupancy *was* checked and recorded, and its one co-tenant held an
idle shell. The result is unaffected. What was wrong was calling one success robust, and then
explaining the sustained run's failures with an uncontrolled measurement instead of re-testing.

## 5.3 / The composition 2x2: frame rate x decoder width, on-device (2026-10-02)

`jetson/e2e_2x2.sh`, `results/e2e_2x2/` (raw summaries + `NOTES.json`). One test clip,
`ixq65EiuJ_c-00:03:47.633-00:03:56.133`, 255 frames at 30 fps, full released checkpoint, unpruned
mT5-base, nvpmodel 0 (15 W), 3 repeats per cell. The driver logged `users` / `foreign` / `MemFree`
before every cell (0 and 0 throughout) and reclaimed to a 5000 MB target first -- the protocol §5.2
says is mandatory for the full checkpoint.

M1 gave one point. One extra point cannot attribute a composition residual, so this varies frame
count and decoder width independently on a shared clip instead of taking the single corner.

| cell | frames | beams | total ms (±std) | pose ms | convert ms | LM ms | J/sentence | dyn J | peak GPU GB |
|---|---|---|---|---|---|---|---|---|---|
| source, beam 4 | 255 | 4 | **OOM** (ran once as M1, §5.1) | — | — | — | — | — | — |
| source, greedy | 255 | 1 | 7998.0 ± 201.7 | 6260.7 | 24.66 | 1712.6 | 43.11 | 11.32 | 2.406 |
| **24 fps, beam 4** | **204** | **4** | **7728.7 ± 320.7** | **5349.5** | **21.00** | **2358.2** | **42.94** | **12.95** | **2.555** |
| 24 fps, greedy | 204 | 1 | 6672.1 ± 347.0 | 5102.2 | 18.99 | 1550.9 | 35.92 | 9.43 | 2.399 |
| 16 fps, beam 4 | 136 | 4 | 5616.1 ± 393.8 | 3414.3 | 17.57 | 2184.2 | 32.75 | 10.70 | 2.530 |
| 16 fps, greedy | 136 | 1 | 4705.7 ± 121.4 | 3336.7 | 13.33 | 1355.6 | 25.80 | 7.27 | 2.390 |

The **24 fps row is the frontier's recommended operating point (L16), measured end to end on the board
for the first time** — it had only ever been composed. The 16 fps row was the pre-frontier priority and
is kept because three frame counts constrain the per-frame term better than two.

**The stages are exactly additive.** pose + convert + LM = total to within 0.04 ms in every cell, so
the end-to-end residual noted at M1 is not a within-run accounting gap. It has to come from the
comparison across runs, which is where it will have to be chased.

**Pose is invariant to decoder width** (3336.7 → 3414.3 ms, +2.3%, inside the run-to-run std) and
**linear in frame count** at ~23.7 ms/frame: imread 5.3–5.7, preprocess 4.6–4.9, TRT 11.9–12.8,
postprocess 1.19, plus ~3% fixed overhead. Both are the independence the pipeline is supposed to
have, so this doubles as a consistency check that passed.

**Frame rate is the big knob, decoder width the small one.** Source → 16 fps at greedy cuts latency
**41.2%** and energy **40.2%**. Greedy → beam 4 at 16 fps costs **19.3%** latency and **26.9%**
energy, landing almost entirely in the LM (+61.1%). Subsampling also shifts the bottleneck: the pose
share of latency falls 78.3% → 70.9% → 60.8% across the three cells.

### What the grid says about the frontier's energy model

This is the reason the grid was run, and it splits cleanly in two.

**The frame-rate term is well calibrated.** Source → 24 fps at greedy measures **−16.7%** system
energy (43.11 → 35.92 J); `frontier.py` predicts **−17%**. The pose side of the composition can be
trusted.

**The decoder-width term is understated by 3–5×, and now replicated at two frame rates.**

| | measured Δ J/sentence | composed Δ (`frontier.py` `LM_J`) | ratio |
|---|---:|---:|---:|
| greedy → beam 4 at 24 fps (T=204) | **+7.02** | +2.04 | 3.4× |
| greedy → beam 4 at 16 fps (T=136) | **+6.95** | +1.34 | 5.2× |

Note that **the measured penalty is essentially constant in absolute joules** (7.02 vs 6.95 J) across a
1.5× change in frame count — exactly what theory predicts, since decoder cost tracks tokens generated
rather than input length. The composition instead makes it grow with T, because it interpolates `LM_J`
in T.

**The cause is a model mismatch, not a measurement error.** `frontier.py` composes `LM_J` from §2.9C,
which is headed *"(pruned checkpoint)"* — vocabulary 26,078. Every end-to-end run here loads the
**full released** checkpoint with unpruned `mt5-base` — vocabulary **250,112**. The output projection
cost scales with `beams × vocab`, so the full model's beam-4 penalty is far larger than the pruned
model's. §L4 had already measured the per-token consequence (full 69 ms/token vs pruned 55) but it was
never carried into the energy model.

Applying a flat 1.255 factor (69/55) to `LM_J` brings **all three beam-4 cells** inside ±3% — M1
−1.1%, 24 fps +1.3%, 16 fps +2.8% — and makes **both greedy cells worse** (−9.4%, −14.2%). A flat
factor is therefore the wrong correction, which is itself consistent with the penalty scaling with
beam count rather than being constant.

**What follows.** The deployable model is the pruned checkpoint (4× smaller, −0.28 BLEU-4
[−0.63, +0.05], and it loads in ~1.5 GB against the >5.3 GB below). If that is the intended
deployment then the frontier's energy model is right and **these end-to-end runs used the wrong
checkpoint** — M1 chose the released one because it is the FP32 reference rung, correct for a first
run and wrong as a basis for validating the frontier. **Re-running this grid on the pruned checkpoint
is the decisive experiment**, and it is queued. Until it lands, do not quote the composition's
beam-width energy cost: it is a pruned-model number, and on the full model we measure 3–5× more.

**`mJ_per_frame` is the wrong denominator and this run proves it.** It *rises* under subsampling,
169.07 → 189.70 mJ/frame, because the per-sentence LM cost is amortised over fewer frames. Any
end-to-end energy claim must be per sentence. The per-frame basis stays valid for the pose stage
alone, which is where it was defined.

### Two separate memory problems, and the thresholds we had were both wrong

**(a) The load threshold was wrong by 1.3 GB.** §5.2 put the full checkpoint's requirement at ~4 GB of
MemFree, inferred from M1's single success at 4625 MB. That does not reproduce. Measured 2026-10-03,
loading the full checkpoint:

| MemFree after reclaim | outcome |
|---:|---|
| 5214 MB | **fails** in `_load_state_dict_into_meta_model` |
| 5268 MB | succeeds |
| 5292–5478 MB | succeeds (the three cells of 10-02) |
| 5726–5768 MB | succeeds |

There is a **cliff just above 5.2 GB**. `drop_file_cache.py --target-free-mb=5000` reclaims to only
~5200, so it straddles the cliff — which is why three cells failed *at load* on 10-03 while the
identical 16 fps/greedy cell reproduced exactly at 5268 MB. The driver now targets **6200** with the
guard at **5400**, and every cell has passed since. **A single success is not a threshold**, which is
the same error shape as the occupancy episode in §5.2.

**Scope of that cliff, corrected 2026-10-03.** It is the requirement for the full checkpoint **plus a
resident TensorRT pose engine**, not for the checkpoint alone. §2.5g's `full_ours_fps24` loaded the
same full checkpoint and ran to completion at **4136 MB** of MemFree, because `eval_openasl.py` reads
pose pkls and holds no engine. So the >5.3 GB figure applies to the end-to-end path (`e2e_translate`),
and the eval path needs roughly a gigabyte less. Worth stating because it is also circumstantial
support for the open question in §5.4: if a resident engine moves the memory requirement this much, a
standalone LM sweep is plausibly a different operating condition than an in-process one.

**(b) `source × beam 4` is genuinely marginal, and this time the evidence supports it.** It has now
failed three times — at 5691 MB (in `_beam_search`), and twice more at 5768 MB with
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which only moved the failure site to
`cache_utils.py:120`, the **KV-cache concat**. That is the same site where §5.1's FP32-engine attempt
died. Meanwhile `24 fps × beam 4` passes reliably at effectively the same MemFree (5757 MB), so this
is a working-set limit specific to 255 frames at beam 4, not a reclaim problem.

Against that stands **M1, which ran this exact cell at 4625 MB on 09-28**. The configurations match on
every field we can still compare (`max_new_tokens` 64 and `max_length` 256 are the script defaults
that both runs took; M1's own summary JSON was never saved, only its power JSON, so the comparison is
not airtight). The honest reading is that the cell sits **right at the edge**: it fits sometimes.

**An earlier draft of this section got this wrong twice** — first by calling it a hard ceiling and
concluding that frame subsampling is a *feasibility* requirement for beam search, then by swinging to
"reproducibility failure, not a ceiling" on the strength of M1 alone. With `expandable_segments` ruling
out simple fragmentation and 24 fps passing at the same MemFree, a real working-set limit is the better
explanation, and M1 is the outlier to be explained rather than the refutation.

**It does not block the deliverable.** `source × beam 4` is the frontier's *accuracy* reference, and
that was measured on the Mac at n=976; nothing requires it to run on the board. The recommended cell,
24 fps × beam 4, runs reliably.

**Confirmed the same day (§5.4): the pruned checkpoint runs this cell at 4238 MB of MemFree** — 1.5 GB
*less* headroom than the 5768 MB at which the full checkpoint failed it, with a peak of 1.09 GB against
2.56 GB. So the limit here is specifically a **full-checkpoint** limit, and the deployable model does
not have it.

### Scope

**n = 1 clip, so no accuracy claim is made here.** The reference is *"When I moved to Texas one year
and a half ago, this was my first artwork."* The decodes were:

- source, greedy -- "When I moved to Texas two years ago, it was my first Airbnb subscription."
- 16 fps, beam 4 -- "When I moved to Texas two years ago, it was my first trip."
- 16 fps, greedy -- "I moved to Texas two years ago and it was my first time."

All three recover the clause structure and all three miss the content word. The source-rate decode
tracks the reference most closely, which is *consistent with* L7.2 and nothing more -- one clip
cannot separate that from chance. The 2×2 interaction term is **not computable** because the
`source × beam 4` cell does not exist.

## 5.4 / The pruned checkpoint end to end, and per-stage energy (2026-10-03)

`jetson/e2e_pruned.sh`, raw in `results/e2e_pruned/`. Same clip and protocol as §5.3, but loading
`openasl_pose_only_slt_pruned.pth` with the pre-pruned `mt5-base-openasl-pruned` directory, and with
the new per-stage power windows (`--power-interval-ms 50`). This is the experiment §5.3 said was
decisive.

### Per-stage energy now exists (the LM track's ASK-E2E-KNEE ask)

`unisign/e2e_translate.py` marks pose / convert / LM boundaries inside the power window, and each
stage's joules are integrated from **its own samples** rather than apportioned from the sentence total
by latency share — which would be invalid, since the stages do not draw equal power. The numbers bear
that out: the pose stage runs at **~5.0 W** and the LM stage at **~6.2 W**, so a latency-share split
would have mis-assigned roughly a quarter of the LM's energy.

| cell | frames | total ms | pose J (W) | convert J | LM J (W) | stage sum | window total | residual | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| source, beam 4 | 255 | 8742.9 | 33.240 (5.026) | 0.000 | 12.726 (6.348) | 45.966 | 46.51 | +1.17% | 1.090 |
| source, greedy | 255 | 7894.4 | 32.519 (5.055) | 0.000 | 8.227 (6.227) | 40.746 | 41.37 | +1.51% | 1.030 |
| **24 fps, beam 4** | **204** | **6999.1** | **25.261 (5.092)** | 0.000 | **11.855 (6.226)** | **37.116** | **37.75** | +1.67% | **1.073** |
| 24 fps, greedy | 204 | 6733.7 | 26.891 (4.985) | 0.000 | 7.210 (5.991) | 34.101 | 34.72 | +1.78% | 1.021 |

All four cells ran, including `source × beam 4`.

**The convert stage is below the measurement floor and must not be quoted from this table.** At ~20 ms
it is shorter than the 50 ms sampling interval, so its window catches one sample and integrates to
exactly 0.000 J. That is a limit of the method, not a finding — the LM track's own estimate of 0.143 J
at M1's average power is the better figure, and it is negligible either way. Everything else carries
25–133 samples per window and the three-stage sum closes on the sentence total to **1.2–1.8%**, which
is the expected edge quantisation plus the inter-stage gaps.

### Pruning is worth ~12% of system energy, and that had never been measured

| cell | full ckpt J/sentence | pruned J/sentence | saving |
|---|---:|---:|---:|
| **24 fps, beam 4** | 42.94 | **37.75** | **−5.19 J (−12.1%)** |
| source, beam 4 | 50.37 (M1, §5.1) | 46.51 | −3.86 J (−7.7%) |
| source, greedy | 43.11 | 41.37 | −1.74 J (−4.0%) |
| 24 fps, greedy | 35.92 | 34.72 | −1.20 J (−3.3%) |

Vocabulary pruning had only ever been argued on **memory and accuracy** grounds (4× smaller,
−0.28 BLEU-4 [−0.63, +0.05]). It is also a double-digit system-energy win at the recommended operating
point.

**And the saving is 3–4× larger at beam 4 than at greedy** (−12.1% / −7.7% versus −3.3% / −4.0%),
which is the mechanism confirming itself: the layer pruning shrinks is the output projection, and that
is evaluated once per beam per token. A vocabulary win should therefore scale with beam width, and it
does.

Two further deployment numbers, both roughly halved against the full checkpoint: **peak GPU 1.02–1.09 GB**
against 2.39–2.56 GB (§5.3), and **mT5 load 28.6–29.0 s** against 56.7–58.2 s. The load figure sits
outside the power window by construction, but it is the dominant term in cold-start latency.

### It also clears the working-set limit that blocked `source × beam 4`

That cell failed three times on the full checkpoint, twice under `expandable_segments`, at up to
**5768 MB** of MemFree (§5.3). On the pruned checkpoint it **succeeded at 4238 MB** — 1.5 GB *less*
headroom. So the §5.3 limit is specifically a full-checkpoint limit, and the deployable model does not
have it.

### The composition diagnosis is half right, and the other half is still open

§5.3 attributed the frontier's understated decoder-width term to the pruned/full checkpoint mismatch.
Testing it directly, with the LM stage now measured rather than inferred:

| greedy → beam 4 | at 24 fps (T=204) | at source (T=263) |
|---|---:|---:|
| full checkpoint, system total (§5.3) | +7.02 | — *(cell OOMs)* |
| **pruned checkpoint, LM stage measured** | **+4.65** | **+4.50** |
| `frontier.py` composition (`LM_J`) | +2.04 | **+2.16** ~~+2.67~~ |

> **Correction, 2026-10-05.** The T=263 cell read **+2.67**, which is a linear *extrapolation* of
> `LM_J` past the measured grid. `frontier.py` does not extrapolate: `lm_energy()` clamps
> `frames >= grid[-1]` to the T=215 row, so the composed value it actually ships at T=263 is
> `11.04 − 8.88 = `**`+2.16 J`**. The ratio band is therefore **2.08–2.27×**, which is what the
> "2.1–2.3×" sentence below already says; the old cell made the table read as 1.7–2.3× and that
> wrong band reached `REPORT.md` §6. Nothing measured changes — both corrected numbers are composed.

So the checkpoint explains roughly **half** the gap (7.02 → 4.65) and a **2.1–2.3× discrepancy survives
on the pruned model itself**, which is the configuration `frontier.py` claims to describe. The
composition is therefore still wrong about beam width, and the checkpoint is not the whole story.

Note also that **the measured penalty is again flat in absolute joules** — 4.65 J at 204 frames against
4.50 J at 263, a 1.3× change in frame count — exactly as in §5.3 on the full checkpoint (6.95 / 7.02 J).
The composition instead makes it *grow* with T (2.04 → 2.67) because it interpolates `LM_J` in T. Two
checkpoints and four frame counts now say the same thing: **decoder-width energy is a per-sentence
constant and should not be modelled as a function of encoder length.**

The leading remaining candidate is that §2.9C measured the LM **standalone** — one process, no pose
engine resident, no TensorRT context in the shared 8 GB pool, its own thermal state — whereas these are
end-to-end runs with both models loaded. The direction is consistent at all four cells: the composition
**overestimates greedy and underestimates beam 4**, compressing the spread from both ends.

| cell | composed `LM_J` | measured LM stage | error |
|---|---:|---:|---:|
| source, beam 4 | **11.04** ~~11.75~~ | 12.726 | **+15.3%** ~~+8.3%~~ |
| source, greedy | **8.88** ~~9.08~~ | 8.227 | **−7.4%** ~~−9.4%~~ |
| 24 fps, beam 4 | 10.88 | 11.855 | +9.0% |
| 24 fps, greedy | 8.833 | 7.210 | −18.4% |

> **Correction, 2026-10-05, same cause as above.** The two `source` rows are at T=263, past the
> measured grid top of 215, and were composed by extrapolation. `frontier.py` clamps instead, so its
> shipped values are 11.04 and 8.88. The 24 fps rows are at T=204, inside the grid, and are unchanged.
> The pattern the table is here to show is unaffected and in fact sharper: the composition still
> **overestimates greedy and underestimates beam 4** in all four cells, and the beam-4 underestimate
> at source is 15.3% rather than 8.3%.

A standalone sweep compressing the spread in both directions is what a different resident footprint
would do, but **this is unverified** — four cells showing a consistent sign is a pattern, not a cause. The clean test is to re-run
the §2.9C sweep in-process with the pose engine loaded, and it is not yet done.

**What to do with the frontier meanwhile.** Its *relative* frame-rate ordering is sound — the frame-rate
term checks out at −16.7% measured against −17% composed (§5.3) — and the accuracy axis was always
measured. The beam-width energy step is the one quantity not to quote from it.

## 5.5 / Integrity of the pose fetch path, and an audit of every file it produced (2026-10-03)

`data/openasl_pose_fetch.py`. Every accuracy number in this document is scored on pose pkls this
script pulled out of the 32 GB HuggingFace archive by HTTP range request, so its failure modes are
worth more than the one line they previously got.

**Four defects, found by reading it rather than by being bitten.**

1. **Resume trusted existence, not size.** The check was `os.path.exists(dst)`. A file truncated by an
   interrupted run exists, so it was accepted on that run and on **every later run**, permanently and
   silently. The archive index carries each member's uncompressed size, so this was detectable for
   free and simply was not checked.
2. **The write was not atomic.** `open(dst, "wb")` then `write(...)` means a kill mid-write leaves a
   partial file under the *final* name — which defect 1 then blesses forever. The rest of this repo
   already uses write-to-`.part`-then-rename (`task1_rtmpose/09_batch_clips.py`); this path did not.
3. **A short range response corrupted the stream.** `_fetch` advanced its cursor by the number of
   bytes *requested* (`start += hi - lo + 1`) rather than received. A proxy trimming a body or a
   connection cut mid-body would shift every subsequent byte. It now advances by `len(r.content)`,
   which makes the loop re-request the remainder and is self-correcting.
4. **A server ignoring `Range` was undetected.** Such a server answers `200` with the whole object;
   treating that as the requested slice would misalign everything. It now raises instead of guessing.

**Fixes.** Write to `<name>.pkl.part`, verify the written size against `ZipInfo.file_size`, then
`os.replace`. A full-member `read()` makes `zipfile` verify the member CRC, so size plus CRC now both
have to pass before a file gets its final name. Resuming re-checks size and re-fetches on mismatch,
reporting each one. Failures are collected and the script exits non-zero instead of reporting success.
A new `--verify-only` audits an existing directory for the cost of the index alone (13.6 MB).

**Verified on a deliberately truncated file.** Halving the Bitcoin-clip pkl (700431 → 350215 B) is
caught as `REFETCH ...: 350215 B on disk, archive says 700431 B`; `--verify-only` reports it and
downloads nothing; the repair restores a file that is **byte-identical** (SHA-256) to the Mac's
original, unpickles, and leaves no `.part` behind.

**Audit of everything the old script produced — all clean.**

| directory | files | result |
|---|---:|---|
| `data/openasl_pose` (test split, the ceiling set) | 976 | 0 mismatches |
| `data/openasl_train_pose_smoke` | 300 | 0 mismatches |
| `data/openasl_5clip_pose` | 5 | 0 mismatches |

**So no result in this document is affected.** That is the outcome worth stating plainly: the bug was
real, the exposure was total — every BLEU and ROUGE number rests on these files — and in *this* path
it happens not to have fired. The audit is cheap enough to re-run whenever the fetch path is used
again.

**It is not a latent defect, though: the same defect fired on the LM track and cost a training run.**
Defect 1 — resume trusting existence rather than size — also existed in `colab_setup.py`, the parallel
path that extracts poses from the archive on Colab. It fired on 2026-09-29: a pose pkl truncated by a
killed download passed the bare existence check, and the adaptation run died at **step 2250** with
`EOFError: Ran out of input`. Fixed there by the same means, validating against `ZipInfo.file_size`
(commit `60a6000`).

Two consequences worth carrying. **The defect class is demonstrated, not hypothetical**, which raises
rather than lowers the value of the audit above. And **the failure mode was loud rather than silent**
in that instance — an unpickling error that stopped the run — whereas a truncation that still
unpickles would degrade an accuracy number without any error at all. The audit covers the silent case;
the EOFError only ever covered the loud one.

## 5.6 / J8: the C9 protocol, finally applied (2026-10-03)

Two halves: sustained 30-minute runs (`jetson/c9_sustained.sh`, `unisign/sustained_run.py`) and
process-level repeats (`jetson/c9_process_repeats.sh`, `results/c9_reps/`).

### A. The board does not throttle, at any of the three loads

| phase | load | duration | drift (1st→5th fifth) | Tj max | avg W | mJ/frame | verdict |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | pose FP16 only | 1804.7 s, 68,850 frames | — | 53.33 °C | 4.907 | 128.62 | no throttling |
| 1b | pose FP32 only | 1802.4 s, 43,095 frames | **−0.26 %** | 57.44 °C | 6.559 | 274.33 | no throttling |
| **2** | **end-to-end, pruned @ 24 fps** | **1804.7 s, 51,816 frames** | **−0.56 %** | **51.75 °C** | **5.415** | **188.62** | **no throttling** |

Phase 2 is the deployable configuration and had never been run. Over **254 consecutive sentences** it
drifted **−0.56 %** — *negative*, i.e. marginally faster at the end than the start — with Tj peaking at
51.75 °C, well short of anything that would clock down.

**This is the result every short-window row in this document needed.** Phase 2 works out to
**38.48 J/sentence** sustained (9,773,456 mJ / 254) against §5.4's three-sentence measurement of
**37.75 J** — **+1.9 %**. So the 8–25 s windows that produced every latency and energy row are not
optimistic, and the frontier's energy axis does not move under realistic duty cycle. That was a real
risk: the whole table was measured with Tj never above 53 °C and nothing had tested what happens after
half an hour.

Note phase 1b runs hottest (57.44 °C) — FP32 pose is the heaviest sustained load we impose, hotter
than the full pipeline, because the pipeline spends ~30 % of each sentence in the host-bound decoder
with the GPU mostly idle.

### B. Process-to-process variance, and the variance is not where it was assumed

`04_infer_power.py` run as **three separate processes** per config — fresh engine load, CUDA context
and power window each time, with a settle between so the reps are independent rather than a thermal
ramp. Mean ± std across processes, 255-frame clip × 3 passes each:

| | RTMW FP16 | CV | RTMW FP32 | CV |
|---|---:|---:|---:|---:|
| wall ms/frame | 26.667 ± 1.130 | **4.2 %** | 42.051 ± 0.126 | 0.3 % |
| ├ imread | 6.104 ± 0.544 | **8.9 %** | 7.887 ± 0.023 | 0.3 % |
| ├ preprocess | 5.235 ± 0.478 | **9.1 %** | 6.857 ± 0.045 | 0.7 % |
| ├ **TRT** | **13.857 ± 0.047** | **0.3 %** | **25.892 ± 0.062** | **0.2 %** |
| └ postprocess | 1.311 ± 0.055 | 4.2 % | 1.243 ± 0.004 | 0.3 % |
| avg W | 4.881 ± 0.029 | 0.6 % | 6.440 ± 0.025 | 0.4 % |
| mJ/frame | 137.307 ± 2.021 | 1.5 % | 289.353 ± 2.344 | 0.8 % |
| gpu MHz | 308.8 ± 1.5 | 0.5 % | 575.6 ± 0.9 | 0.2 % |

**The GPU is the stable part and the CPU is not.** TRT reproduces at **0.3 % CV** in both precisions —
and reproduces the recorded rows: 13.857 vs §2.2's 13.8 ms, 25.892 vs 25.6 ms. All the variance lives
in `imread` and `preprocess`, at ~9 % CV, and **only at FP16**. At FP32 everything is ≤0.7 %.

The asymmetry has a mechanism: FP32 occupies the GPU for 25.9 ms per frame against FP16's 13.9, so the
CPU has proportionally more slack per frame and the governor settles (gpu 575 MHz vs 309 MHz average).
FP16 is the faster configuration *and* the jittery one, because it is the one whose frame time the CPU
actually bounds.

**Basis note, to prevent a false alarm.** `wall_ms_per_frame` (26.667) includes JPEG decode; §2.2's
"19.7 ms total" does not. Pre + TRT + post here is **20.40 ms**, against 19.7 — consistent. This is
exactly the two-fps-numbers distinction §2.2b exists to document, and it is the first thing to check
before reading these as a regression.

### C. This refines L16 addendum 5 rather than confirming it

Addendum 5 priced the composition residual against §2.2b's **×1.281** CPU-stage drift and concluded a
~10 % system-J swing, which brackets the 6.3 % residual. Measured across three processes the spread is
smaller: FP16 frame time is 4.2 % CV, and at a ~74 % pose share of system energy that is a **~3 %
system-J swing at 1σ**, not 10 %.

**So the hypothesis is weakened, not refuted, and three processes cannot settle it.** The 6.3 % residual
needs roughly 2σ of what we just measured, which a sample of three neither excludes nor establishes —
and §2.2b's ×1.281 was a real single observation, so the distribution plausibly has excursions far
larger than this σ. What is now certain is the *location*: whatever run-to-run term exists is in the
CPU stages, not in TRT, which rules out any explanation that depends on GPU variability.

**Consequence for the protocol.** C9's "3 runs, mean ± std" is now met for the pose rows and the
numbers are stable enough to quote, with the caveat that **FP16 wall-clock rows deserve a ±4 % band
and FP32 ±0.5 %**. The energy rows are tighter than the latency rows (1.5 % and 0.8 % CV), because
power averaging absorbs the CPU jitter that frame time exposes.

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
RTMPose-family extractor, not the 384x288 RTMPose-x used for the Jetson FP32 row; measured in 2.5).
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

### L4 Uni-Sign on the Jetson, PyTorch FP32 (J2, 2026-09-18, `jetson-lpcv-03`, nvpmodel 15W, GPU 612 MHz)

Same Bitcoin clip and authors' poses as L3 (256 frames → encoder length 264, greedy, 24 tokens),
`unisign.unisign_infer --device cuda --repeat 3`, container `slt-jetson:25.06` (torch 2.8 nv25.06).
Both checkpoints reproduce the Mac output **exactly**: identical text, identical token ids, mean
logprob −0.749 (full; Mac −0.75) and −0.492 (pruned). Files: `results/l4_jetson_full_fp32.json`,
`results/l4_jetson_pruned_fp32.json`, logs in `results/logs/j2_*.log`.

| Model | load s | GCN ms | encoder ms | decoder ms | ms/token | total ms | peak GPU GB |
|---|---|---|---|---|---|---|---|
| full (582M) | 32 | 79–90 | 57–58 | 1653–1664 | 69 | 1789–1812 | 2.41 |
| pruned (238M) | 32 | 86–94 | 57–63 | 1330–1331 | 55 | 1475–1483 | 1.03 |

Reading it:
- **The board decoder is 5–7× slower than the Mac CPU** (1.65 s vs 0.27 s full; 1.33 s vs 0.19 s
  pruned) while the **encoder is faster** (57 ms vs 86–105 ms). One 264-token encoder pass is a
  GPU-friendly graph; greedy decoding through `generate()` is 24 sequential steps of Python plus
  ~200 small kernel launches each, and the Orin's ARM cores at 15W (0.9–1.5 GHz) pay that overhead.
  This is launch-bound, not compute-bound, which is exactly what the TensorRT KV-cache engines (L8/J4)
  and, later, CUDA graphs address. The Jetson PyTorch line is the baseline those must beat.
- Pruning saves 20% per token (69 → 55 ms), in line with L3's 20–30% on CPU.
- Pruned decoder time drifted 1240 → 1303 → 1330 ms across three back-to-back runs of the same
  config (another student's container was idle; likely thermal/DVFS). Treat ±5% as run-to-run noise
  on this shared board until the power logger wraps these runs.
- **Memory**: the full FP32 model is 2.3 GB. Building it on the CPU and then `.to("cuda")` OOMed
  twice (NvMap error 12, then `CUDA error: out of memory`) because the board had only ~1.8 GB free
  outside page cache and the two copies coexist. Fix in `unisign/model.py`: the full checkpoint now
  materialises mT5 directly on the target device (`from_pretrained(device_map=...)`), the checkpoint
  is `mmap`-loaded and freed before the device copy, and the pruned tokenizer's `decode` accepts CUDA
  ids (it crashed on the GPU before). The pruned path still builds on CPU and slices first: 1.03 GB
  peak instead of 2.57 GB if the full vocabulary were sliced on the GPU.

### L8.1 mT5 (pruned) → ONNX with explicit KV cache (`task3_mt5_onnx/02_export_manual.py`)

Three graphs, fp32, opset 17, from `weights/mt5-base-openasl-pruned` (vocab 26,078):
`encoder.onnx` 340 MB, `decoder_init.onnx` 614 MB, `decoder_step.onnx` 557 MB (50 inputs: ids, mask,
48 past K/V; 25 outputs). Verified on real poses only (`unisign/onnx_decode.py`, Bitcoin clip, greedy):
**tokens identical to PyTorch, max |logprob diff| 2.9e-5**; ORT CPU encoder 62 ms, decoder 178 ms for
24 tokens (7.4 ms/token) vs PyTorch 191 ms. Export + verification took one attempt (budget was 3 days).
Remaining for L8: TensorRT engines on the Jetson (queue J4) and the drift check there.

### L8.2 mT5 (pruned) TensorRT engines on the Jetson (J4, 2026-09-18, `jetson-lpcv-03`, 15W, GPU 612 MHz)

`task3_mt5_onnx/03_build_engines.py --enc-len 1,264,512 --dec-len 1,1,128 --workspace-gb 1`, one
engine per process, TF32 off for the FP32 engines. Decode loop `unisign/trt_decode.py` (greedy,
explicit KV cache, PyTorch reference computed on the same board then freed), Bitcoin clip, 24 tokens,
`--repeat 3`. Files `results/l8_jetson_trt_{fp32,fp16,bf16}.json`, logs `results/logs/build_mt5_*`, `results/logs/trt_decode_*`.

| Engine set | encoder / init / step size | build s | builder peak GPU MiB | tokens vs PyTorch | max \|Δlogprob\| | encoder ms | decoder ms (24 tok) | ms/token |
|---|---|---|---|---|---|---|---|---|
| FP32 | 341 / 616 / 558 MB | 15 / 23 / 13 | 381 / 698 / 585 | **identical** | 5.6e-4 | 82–86 | 459–460 | **19.2** |
| FP16 | 172 / 309 / 280 MB | 58 / 54 / 43 | 394 / 375 / 319 | **broken**: 64× token 0 | 10.2 (uniform) | 19–25 | (1132–1372 for 64 tok) | 17.7–21.4 |
| BF16 | 171 / 311 / 282 MB | 37 / 43 / 31 | 394 / 358 / 292 | diverges at step 3 (22 vs 24 tok) | 1.1 | 19–28 | 475–485 (22 tok) | 21.8 |
| PyTorch FP32 (L4, same board) | — | — | — | reference | — | 57 | 1330 | 55 |

Reading it:
- **FP32 engines pass the gate**: identical token ids to PyTorch; the logprob drift (5.6e-4) is
  larger than ORT-CPU's 2.9e-5 but far below any token-flip margin (min per-token logprob on this
  clip is −1.39).
- **Decoder 2.9× faster than PyTorch on the same board** (19 vs 55 ms/token): the explicit-KV-cache
  engine turns ~200 launches per step into one; the remaining 19 ms is still mostly per-step host
  work (50 input bindings re-fed, output clones) rather than the 238M-param matmuls, so CUDA graphs /
  keeping the KV cache device-resident are the next lever.
- The TRT encoder (82 ms) is **slower** than PyTorch (57 ms) at this length; with TF32 off TensorRT
  falls back to plain FP32 FFMA kernels while cuBLAS in PyTorch has better FP32 GEMM tiles. The
  encoder is one pass per sentence, so this costs ~25 ms per sentence vs the ~870 ms saved on the decoder.
- **FP16 engines are numerically broken for mT5**: every step returns token 0 with logprob
  −10.169 = −ln(26078), i.e. a uniform distribution from saturated logits. This is the known T5/mT5
  FP16 overflow (feed-forward activations exceed 65504 in later blocks; HF's own fp16 path clamps
  them). Latency is still informative: the FP16 **encoder** is 3–4× faster (19–25 vs 82 ms), but the
  FP16 **decoder step is no faster** (18 vs 19 ms/token), confirming the step is host-bound, not
  GEMM-bound.
- **BF16 engines run but drift**: coherent output (*"The tweet appears to target donations like those
  involved in the Biden campaign."*) that diverges from the FP32 greedy path at step 3 where the
  reference margin is small (logprob −1.05 for the chosen token). BF16's 8-bit mantissa is too coarse
  for token-exact greedy decoding; its BLEU cost would need the 976-clip eval on the board (not done).
  Decoder step 21.8 ms/token — again no faster than FP32, so **reduced precision buys nothing on the
  decoder here; the deployable LM engine set is FP32** (2.07 GB resident) until the step loop is
  made GPU-resident. If memory forces it, the right FP16 recipe is mixed precision with the residual
  stream / FFN output projections kept in FP32, not a blanket FP16 or BF16 flag.
- Memory: engines resident take 2.07 GB of MemFree (1.5 GB weights + activations + TRT context); the
  PyTorch reference (1.03 GB) had to be freed first, and the page cache evicted (see the L4 note and
  `jetson/README.md`), or engine deserialisation failed with `Cuda Runtime (out of memory)`.

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

**Measured on the test split (2026-09-30, Colab A100, 976 clips, beam 4):**

| keep set | cap | BLEU-4 | ROUGE-L |
|---|---|---|---|
| leaky, 26,078 | 100 | 22.87 | 42.98 |
| **leak-free, 26,025** | 100 | **22.84** | **42.99** |
| leak-free, 26,025 | 64 | 22.84 | 42.99 |

**The leak was worth 0.03 BLEU-4** (ROUGE-L rose 0.01). Report it as a methodology finding — the

Paired bootstrap on the 976 test clips (1000 resamples, `unisign/bootstrap_ci.py`):

| | leaky → leak-free | 95 % CI | verdict |
|---|---|---|---|
| BLEU-4 | −0.03 | [−0.11, +0.01] | not established |
| ROUGE-L | +0.01 | [−0.02, +0.04] | not established |

The intervals are unusually tight because the two models differ by only 53 of 26,078 rows, so most
sentences are word-identical. That makes this a **precise** null rather than an underpowered one: the
leak's effect on the test score is bounded at roughly 0.1 BLEU-4.
selection channel is real, generalisable and unchecked by most work — rather than as a results
correction, because its magnitude here is inside the noise. Every other test-split figure in this file
uses the leaky keep set; the difference does not change any relative conclusion.
Files: `results/block5/eval_test_pruned_traindev_cap{100,64}.json`.

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
learning rate) was killed by a Colab disconnect and **never completed**, so it has no eval. "Label
smoothing was the problem" is therefore the best available reading, not an isolated result — the
working recipe changed smoothing, learning rate and warmup together.

> **Correction, 2026-10-05.** This caveat said the run died "after 10 log lines". Its log,
> `results/colab_runs/ctrl_ls02_lr1e5__train.log`, reaches **step 400 — 3,200 of 20,000 clips**,
> ~16 % of the epoch. Partial, not absent. Over those 400 steps the loss is flat at
> **3.3865–3.3993**, i.e. the smoothing floor is present at `lr 1e-5` as well as at `lr 1e-4`. That is
> consistent with the diagnosis — the floor tracks label smoothing rather than the learning rate — but
> it is a partial training curve with no evaluation, so it does not promote the reading to an isolated
> result.

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

~~Checkpoint: `weights/adapt_fps16_lr1e5_seed42.pt` … this file is the one that produced the numbers
above.~~

> **RETRACTED 2026-10-05 — the file was mislabelled three ways and did not produce these numbers.**
> Read back from the checkpoint's own saved `args`, it records `out_dir =
> /content/runs/ctrl_ls0_lr1e4`, `fps = None` and `lr = 1e-4`, with `epoch_done = False` at step
> 2000. So it is the **`ctrl_ls0_lr1e4` run** — the *control* condition at source rate on the
> *rejected* learning rate — and it is a **mid-epoch partial**, not a finished model. Its filename
> claimed the opposite on every count. Renamed on the LM-track Mac to
> `weights/ctrl_ls0_lr1e4_seed42_step2000_PARTIAL.pt`; `weights/` is gitignored, so nothing in the
> repo moved.
>
> **No checkpoint for L15.3's 16 fps adaptation survives on the LM-track Mac.** The two Colab
> runtimes that trained Block 4 were wiped, and the one file rescued was not the one believed.
>
> **The L15.3 results themselves are unaffected.** They rest on the eight eval JSONs in
> `results/block4/`, which hold all 967 per-clip predictions and are what every CI in §L15.3 was
> bootstrapped from. The retraction is about *provenance and reproducibility*, not about the numbers:
> anyone reconstructing a model from that file would have got a different one.
>
> **§L17 is now the reproduction path.** Its 20,000-clip rung retrains this adaptation from scratch
> and lands within −0.054 BLEU-4 / −0.024 ROUGE-L of L15.3, with the un-adapted baseline bit-exact —
> so the result is independently reproducible even though the original artifact is not recoverable.

### L16 Milestone M4 / C10: the accuracy–energy frontier plot (`results/frontier.png`, `results/frontier.csv`)

The project deliverable. `results/plot_frontier.py` builds it from two files rather than from any
number typed into the script:

* **Accuracy** from `results/frontier_accuracy.json`, which `unisign/grid_table.py --out` writes by
  scoring the nine eval JSONs over all 976 test clips. Regenerate with
  `python -m unisign.grid_table -n 1000 --out results/frontier_accuracy.json` (~12 min on the Mac).
  It reproduces the L13 table exactly.
* **Energy** by importing `unisign.frontier`, which composes the pose track's two measured board
  stages. `frontier.py` was refactored into `build_rows()` / `mark_pareto()` behind a `__main__`
  guard so it can be imported without printing its report; its printed output is unchanged.

**Error bars.** The bootstrap estimates the *delta* from the beam-4/source reference under draws
shared by every cell, so a cell's bar is drawn at `[ref + lo, ref + hi]`, i.e. half-widths
`(d − lo)` and `(hi − d)` around that cell's own point. The reference cell has no bar **by
construction**, which is not missing data. (The first version of the script re-centred the raw
`[lo, hi]` on the cell's own score, which put every bar entirely below its point and raised a
negative-`yerr` error from matplotlib — the endpoints are now checked against `ref + CI` directly.)

**What the plot says.** 6 of 9 cells are Pareto-optimal.

| | cell | system J | BLEU-4 | vs reference |
|---|---|---:|---:|---|
| reference | beam 4 @ source | 43.4 | 22.87 | — |
| **knee** | **beam 4 @ 24 fps** | **36.1** | **22.80** | **−17 % J for −0.07 BLEU-4** |
| cheapest | greedy @ 16 fps | 25.7 | 19.83 | −41 % J for −3.04 BLEU-4 |

The knee is the recommendation: the −0.07 sits inside [−0.53, +0.39], so it is not a measurable
accuracy cost, and it buys 17 % of system energy. Below that the frontier gets expensive — the
remaining 24 % costs ~3 BLEU-4.

**Three things the figure is not**, all annotated on the figure itself so a reader cannot take it
for more than it is: system joules are **composed, not measured end-to-end** (~6 % low against the
one M1 end-to-end run, ~7 % high for a median-crop clip — opposing biases, neither of which reorders
the cells); the reduced frame rates are **emulated** by thinning 30 fps keypoints; and every cell
carries the −0.03 BLEU-4 test-set vocabulary leak, uniformly, so it cannot reorder them either.

#### L16 addendum (2026-10-03): the grid validated the frame-rate term and falsified the beam-width term

The pose track ran the composition grid end to end on the board (§5.3), including **24 fps × beam 4 —
the cell L16 recommends, measured for the first time: 7728.7 ms, 42.94 J/sentence.**

* **Frame-rate term: validated.** Source → 24 fps at greedy measures **−16.7 %** system energy against
  the **−17 %** `frontier.py` predicts. The pose side of the composition can be trusted.
* **Decoder-width term: not validated, and not quotable.** Measured greedy → beam 4 costs **+7.02 J**
  at 24 fps and **+6.95 J** at 16 fps, against composed +2.04 and +1.34 — understated 3–5×.

**The cause is a checkpoint mismatch.** `LM_J` comes from §2.9C, measured on the **pruned** checkpoint
(vocab 26,078); every end-to-end run so far loads the **full released** checkpoint (vocab 250,112).
The output projection costs `beams × vocab`, so the full model's beam penalty is genuinely larger.
Since the pruned checkpoint is what deploys, the composition is right for the deployed system and the
end-to-end runs used the wrong model. **The pruned grid is queued and is the decisive experiment.**

**Our prediction for it, recorded before it runs.** §5.3 reads the constant absolute penalty as
theory-confirming, on the grounds that decoder cost tracks tokens generated rather than input length.
That holds for the output projection (`beams × vocab`) but not for cross-attention, where each beam
attends over all `T` encoder states every step (`beams × T`). On the full model the 250 K projection
swamps the T term; on the pruned model it is 9.6× smaller and the T term should become visible. Our
own §2.9C rows already show it — the penalty is 1.15 / 1.34 / 2.16 J at T = 68 / 137 / 215, growing
**1.88×** across a 3.2× change in T, and those are measured rows, not interpolation.

**So: the pruned grid should show a penalty that RISES with frame count, landing near 1.3–2.0 J, not a
flat ~7 J.** If it does, the only error was the checkpoint. If the pruned penalty is also flat near
7 J, then §2.9C and the end-to-end runs disagree about the *same* model, which would be a more serious
problem. Written down in advance so the prediction cannot be fitted afterwards.

**What does not change.** The lever ordering survives even at the full-checkpoint penalty: frame rate
is 102.9 J per BLEU-4 point against 3.5 for beam width, so frame rate remains the better lever by
~29× instead of ~94×. The recommendation of beam 4 @ 24 fps rests on the frame-rate term, which is
the validated one.

#### L16 addendum 2 (2026-10-03): our prediction was falsified

The pruned grid ran (§5.4). The prediction recorded in the previous addendum — *"the pruned grid
should show a penalty that RISES with frame count, landing near 1.3–2.0 J"* — is **wrong on both
counts.**

| | predicted | measured (§5.4, LM stage integrated from its own power samples) |
|---|---|---|
| direction vs T | rises | **flat**: 4.65 J at T=204, 4.50 J at T=263 |
| magnitude | 1.3–2.0 J | **4.5–4.65 J** |

**The cross-attention argument is not supported.** We reasoned that beam cost has a T-linear
cross-attention term (`beams × T`) which the 250 K output projection swamps on the full model and
which should re-emerge once pruning shrinks that projection 9.6×. If that were the mechanism, the
pruned penalty would grow with T. It does not. **Two checkpoints and four frame counts now agree
that decoder-width energy is a per-sentence constant**, and `frontier.py` is wrong to interpolate it
in T.

**The checkpoint mismatch explains about half the gap, not all of it.** 7.02 J (full) → 4.65 J
(pruned) against our composed 2.04 J, so a **2.1–2.3× discrepancy survives on the pruned checkpoint
— the configuration `frontier.py` claims to describe.** Our §2.9C rows showing 1.15 / 1.34 / 2.16 J
at T = 68 / 137 / 215 are therefore not simply reproduced end to end, and the growth in them is the
thing now in doubt.

**The leading candidate is ours to test, not theirs.** §2.9C measured the LM **standalone** — one
process, no pose engine resident, no TensorRT context in the shared 8 GB pool, its own thermal state.
The end-to-end error has a consistent sign at all four cells: the composition **overestimates greedy
(−9.4%, −18.4%) and underestimates beam 4 (+8.3%, +9.0%)**, compressing the spread from both ends,
which is what a different resident footprint would do. Four cells with a consistent sign is a pattern,
not a cause. **The clean test is to re-run the §2.9C beam × T sweep in-process with the pose engine
loaded.** Not yet done.

**What this does and does not change.**

* The **recommendation is unaffected**: beam 4 @ 24 fps rests on the frame-rate term, which board
  measurement confirms (−16.7% measured vs −17% composed).
* The **beam-width energy numbers in `frontier.py` remain not quotable**, now for a reason that
  survives the checkpoint fix.
* **Pruning is a bigger win than we claimed.** §5.4 measures it at **−5.19 J, −12.1% of system
  energy** at the recommended cell — we had only ever argued pruning on memory and accuracy. The
  saving is 3–4× larger at beam 4 than greedy, because the layer pruning shrinks is the output
  projection, evaluated once per beam per token. The mechanism confirms itself.
* Pruned peak GPU is **1.02–1.09 GB** against 2.39–2.56 GB, mT5 load **28.6–29.0 s** against
  56.7–58.2 s, and it **clears the working-set failure** that blocked `source × beam 4` on the full
  checkpoint — succeeding at 4238 MB where the full model failed at 5768 MB.

#### L16 addendum 3 (2026-10-03): the accuracy axis is on the authors' keypoints, now labelled

§2.5f measured the deployable config — pruned checkpoint on **our own** RTMW FP16 keypoints — for the
first time, and in doing so exposed an assumption the frontier had never stated.

**The frontier's nine accuracy cells are scored on the AUTHORS' released keypoints.** The deployed
system feeds poses from our own extractor. Substituting them is worth **−0.35 BLEU-4 [−4.23, +3.17]**
at n=30 (§2.5c), so the deployed system's absolute accuracy is a *composed* estimate of ~22.8 BLEU-4 /
~43.1 ROUGE-L whose interval is dominated by that n=30 pose term at **±4 BLEU-4** — far wider than
most differences the plot resolves.

**Pruning does not add to it.** On our own keypoints pruning costs **−0.46 BLEU-4 [−3.82, +2.03]**,
consistent with the **−0.28 [−0.63, +0.05]** we measured on the authors' at n=976. The two pose
sources agree about pruning, which is the question §2.5f existed to ask.

**Relative ordering is unaffected**, because every cell uses the same poses — the same argument that
protects the ordering from the energy biases. The absolute column was already the weak one and this
widens it further. `unisign/frontier.py`, `results/plot_frontier.py` and the figure's caption now say
so explicitly; the plot carries it as a fourth caveat paragraph.

**What narrows it:** the n=100 pass (§2.5g, `jetson/p10_n100.sh`), which takes the pose-substitution
comparison to 100 distinct videos with the authors' reference poses restricted to the same 100 clips.

#### L16 addendum 4 (2026-10-03): addendum 3's pose number is superseded, and the deployed config is now measured

§2.5g took the pose-substitution comparison to **n=100 from 100 distinct videos**, with the authors'
reference poses restricted to the same clips. Two things follow for the frontier.

**1. Addendum 3's number is stale and moved against us.** It quoted the pose-substitution term as
−0.35 BLEU-4 [−4.23, +3.17] from n=30 and read it as "our extractor costs nothing detectable". At
n=100 it is **−1.89 [−4.26, +0.27]** at 24 fps and **−2.62 [−5.38, +0.35]** at source. Still not
established — but only just, and the honest statement is that **our extractor plausibly costs ~2
BLEU-4**, not ~0. ROUGE-L does not rescue it.

This is the same error pattern this file already records twice — the 16 fps "free at n=30" claim and
the ceiling-gap retraction. **A point estimate near zero inside a ±4 interval is not evidence of no
effect.** We restated that principle in the metric-power work and then failed to apply it to a number
handed to us. `unisign/frontier.py` and the figure caption now carry the n=100 figures.

**2. The deployed configuration is no longer a composition.** Board pose extraction + pruned
checkpoint + 24 fps + beam 4 measures **22.99 BLEU-4 / 43.43 ROUGE-L** (n=100). The composed estimate
addendum 3 reported was ~22.8, so the composition was close — but it is now a measurement and should
be quoted as one.

**3. Our central recommendation is independently reproduced on a different pose source.** 24 fps vs
source rate on *our own* keypoints is **+0.05 BLEU-4 [−1.11, +1.03]** — a tight null reproducing the
n=976 authors'-poses result of −0.07 [−0.53, +0.39]. Two pose sources, two clip sets, same answer:
**24 fps is free.** It also retires the n=30 artefact that read +1.97 for this comparison.

**4. Pruning is confirmed a third time.** −0.44 [−1.87, +0.94] on our keypoints at n=100, against
−0.46 [−3.82, +2.03] at n=30 and −0.28 [−0.63, +0.05] on the authors' at n=976. Same point estimate,
tightening interval, three independent measurements.

**5. Circumstantial support for L16 addendum 2's open question.** §2.5g found the >5.3 GB memory cliff
is the full checkpoint **plus a resident TensorRT engine** — `eval_openasl` holds no engine and ran
the same checkpoint at 4136 MB. Our standing hypothesis for the surviving 2.1–2.3× beam-width
discrepancy is that §2.9C measured the LM *standalone* while every end-to-end run has the pose engine
resident. A resident engine evidently changes the operating condition enough to move a memory cliff by
over a gigabyte, which makes it more plausible that it also moves energy. Still unverified; the clean
test remains re-running the §2.9C sweep in-process.

#### L16 addendum 5 (2026-10-03): run-to-run variation is large enough to explain the composition residual

§J8's C9 audit found that "3 runs, mean ± std" was never actually met for the energy rows: where
repeats exist they are **three passes inside one process**, not three processes. The five pose-engine
rows have no repeat at all.

**This matters directly to the frontier**, because `POSE_J_PER_S`, `LM_J` and the end-to-end runs come
from **three different processes**, and L16 addendum 2 already concluded the ~6.3 % composition
residual must be an across-run effect rather than a within-run accounting gap.

§2.2b measured that across-run term: between two runs of the *same* config, `imread` went 5.33 → 6.83
ms and `preprocess` 4.76 → 6.10 ms, both **×1.281**. Pricing that against M1's own per-frame
breakdown:

| | value |
|---|---|
| CPU stages per frame (imread + preprocess + post) | 12.39 ms — **49.4 %** of frame time |
| TRT per frame | 12.71 ms |
| frame time if CPU stages move ×1.281 | 25.10 → **28.58 ms (+13.9 %)** |
| pose share of system energy at source rate | **74.4 %** (32.3 of 43.4 J composed) |
| **resulting shift in system J** | **~10 %** |

**A ~10 % swing comfortably brackets the +6.3 % residual.** So the across-run hypothesis is not just
the remaining candidate by elimination — it is *sufficient in magnitude*, using a variation the pose
track measured directly rather than one we assumed.

This does not identify which run drifted, and it is not a correction to the composition's structure:
the frame-rate term still validates at −16.7 % measured against −17 % composed. What it says is that
**the absolute column should never have been quoted to better than ~10 %**, and that
`c9_process_repeats.sh` — three *separate* processes per config — is the measurement that will put a
real interval on it.

It also leaves L16 addendum 2's other open question untouched: the **2.1–2.3×** beam-width
discrepancy is a *systematic* sign-consistent error across four cells (greedy overestimated, beam 4
underestimated), which run-to-run noise does not produce. That one still needs the in-process §2.9C
re-run.

> *Two numbers corrected by the pose track, 2026-10-03, neither changing the conclusion.* The pose
> share is **74.4 %**, not 72 % — the row's own parenthetical (32.3 of 43.4 J) gives 74.4 %, and the
> resulting system-J shift is **10.3 %**, which brackets the 6.3 % residual slightly more comfortably
> than stated. And the beam-width discrepancy is **1.7–2.3×**, not 2.1–2.3×: §5.4 measures 4.65 J
> against a composed 2.04 at T=204 (2.28×) and 4.50 against 2.67 at T=263 (1.69×). The lower bound
> matters, because 1.69× at source rate is the weaker end of the effect and quoting the range from
> its top makes the systematic look more uniform than it is.

#### L16 addendum 6 (2026-10-03): the pose-share correction is accepted; the beam-width range correction is not

Two corrections were made to addendum 5. **The first is right and is kept.** The pose share is
**74.4 %** (32.3 of 43.4 J), not the 72 % we wrote — our own parenthetical contradicted our own
arithmetic — and the resulting system-J shift is **10.3 %**, bracketing the 6.3 % residual slightly
more comfortably than we claimed.

**The second is wrong, and the range is restored to 2.1–2.3×.** It rests on a composed value of
2.67 J at T=263 that `frontier.py` does not produce. Checked by running it:

```
T=204: beam4 10.878  greedy 8.833  penalty 2.044 J     <- agrees with 5.4's +2.04
T=263: beam4 11.040  greedy 8.880  penalty 2.160 J     <- 5.4 quotes +2.67
```

The cause is the clamp. `lm_energy` tops out at the measured grid's largest encoder length:

```python
grid = sorted({t for (_, t) in LM_J})      # 68, 137, 215
if frames >= grid[-1]:
    return at(grid[-1]), interp
```

**T=263 is beyond the grid, so the composition clamps to T=215 rather than extrapolating** — a
deliberate choice, since extrapolating an energy model past its measured range is how composed
numbers acquire fictitious precision. 2.67 looks like linear extrapolation past T=215, which the code
does not do.

With the clamped value the ratios are **4.65 / 2.044 = 2.27×** and **4.50 / 2.160 = 2.08×**, i.e.
**2.1–2.3×** as originally written. The substantive point behind the correction still stands and is
worth keeping: the discrepancy is *not* perfectly uniform across the two cells, and quoting a range
from its top would hide that. It is simply a narrower spread (2.08–2.27) than the proposed 1.69–2.28.

**Neither correction changes any conclusion.** The residual is still explained in magnitude by
across-run variation; the beam-width error is still a sign-consistent systematic that run-to-run noise
cannot produce; and the in-process §2.9C re-run is still the test that would settle it.

#### L16 addendum 7 (2026-10-03): addendum 5 was too generous; the sustained run validates the energy axis

§5.6 measured across **three separate processes** what addendum 5 could only estimate. Two results,
one against us and one for us.

**1. Our "~10 % comfortably brackets the residual" was too strong.** Addendum 5 priced §2.2b's ×1.281
CPU drift at a ~10 % system-J swing and concluded the across-run hypothesis was *sufficient in
magnitude*. Measured, the spread is **4.2 % CV on frame time**, which at a 74.4 % pose share is
**~3.1 % on system J at 1σ**:

| | system-J swing | in σ |
|---|---:|---:|
| addendum 5's estimate | ~10 % | 3.2σ |
| **measured, 3 processes** | **3.1 % (1σ)** | — |
| the residual being explained | 6.3 % | **~2.0σ** |

So the residual sits at about **two sigma**, which **three samples neither exclude nor establish**.
"Comfortably brackets" was wrong; "consistent with, but not demonstrated by" is the honest statement.
Two caveats pull in opposite directions: n=3 is a thin basis for a σ, and §2.2b's ×1.281 was a *real
single observation*, so the tail may be considerably fatter than this σ implies.

**What is now settled is the location.** All the spread is in `imread` and `preprocess` at ~9 % CV;
TRT reproduces at **0.2–0.3 % CV** in both precisions. **Any GPU-variability explanation is ruled
out.** And the jitter is specific to the configuration we ship: at FP32 everything is ≤0.7 %, because
FP32 holds the GPU 25.9 ms/frame against FP16's 13.9, leaving the CPU slack and letting the governor
settle. **FP16 is both the faster config and the jittery one, because it is the one the CPU bounds.**

**2. The frontier's energy axis survives realistic duty cycle, which was a genuine risk.** Every
latency and energy row behind the frontier came from an 8–25 s window with Tj never above 53 °C, and
nothing had tested sustained operation. A 30-minute run of **our deployable config** (pruned, 24 fps,
beam 4) over 254 consecutive sentences drifted **−0.56 %** — marginally *faster* at the end — with Tj
peaking at 51.75 °C, and works out to **38.48 J/sentence sustained against §5.4's 37.75 J, +1.9 %**.

So the short windows are not optimistic, there is no thermal throttling at 15 W, and **the frontier's
absolute energy column does not move under continuous load.** Combined with **this addendum's own**
revision — that the column should not be quoted to better than ~3 % — the axis is in better shape
than the open residual suggested.

> *Cross-reference corrected by the pose track, 2026-10-03.* The ~3 % bound is addendum **7**'s, not
> addendum 5's: addendum 5 said **~10 %**, and the measurement in §5.6 is what revised it down to
> ~3.1 % at 1σ. Attributing the tighter figure to the addendum that argued the looser one would make
> the estimate look like it had been corroborated rather than corrected, which is the opposite of what
> happened. Nothing else in the paragraph changes.

### 2.5h LM-track bootstrap on the n=400 board evals: the pose term is established, and the metrics swap roles

Computed by the LM track on the five raw eval JSONs pushed with §2.5's n=400 results, independently of
the pose track's own bootstraps, which were still running. `unisign/bootstrap_ci.py`, 400 paired clips
aligned by clip name, 2000 resamples.

**Pose substitution (authors' keypoints → ours), pruned checkpoint, 24 fps, beam 4, cap 64:**

| metric | ceiling | ours | delta | 95 % CI | verdict |
|---|---:|---:|---:|---|---|
| **BLEU-4** | 23.81 | 21.57 | **−2.24** | **[−3.50, −1.09]** | **sign established** |
| ROUGE-L | 42.27 | 41.17 | −1.11 | [−2.56, +0.35] | not established |

**1. The pose-substitution cost is now established, after moving against us three times.**

| n | delta BLEU-4 | 95 % CI | verdict |
|---:|---:|---|---|
| 30 (§2.5c) | −0.35 | [−4.23, +3.17] | not established — and we read it as "costs nothing detectable" |
| 100 (§2.5g) | −1.89 | [−4.26, +0.27] | not established, but only just |
| **400** | **−2.24** | **[−3.50, −1.09]** | **established** |

The point estimate moved monotonically away from zero as n grew, which is the signature of a real
effect that small samples were too noisy to see — not of a fluctuation. **The frontier's absolute
accuracy column sits about 2.2 BLEU-4 above what the shipped system delivers**, and `frontier.py`,
`plot_frontier.py` and the figure caption now say so with the established figure rather than a hedge.
Relative ordering across the nine cells is untouched: every cell uses the same poses.

**2. BLEU-4 establishes this effect and ROUGE-L does not — the reverse of L15/L16's pattern.**

This is the more interesting half. In the adaptation work, on identical clips with identical draws,
**ROUGE-L established both effects and BLEU-4 established neither.** Here, on identical clips with
identical draws, **BLEU-4 establishes and ROUGE-L does not.** The metrics have swapped roles on a
different family of effect.

So "ROUGE-L is more powerful than BLEU-4" — our reading when we had only the adaptation results — is
**too simple and is withdrawn**. The defensible statement is that the two metrics have different
sensitivity profiles, and which one resolves an effect depends on what kind of effect it is.

**Hypothesis for the mechanism, labelled as a hypothesis because it is untested.** Frame-rate thinning
and adaptation change fluency and structure — the sentence keeps its length and broad content and gets
better or worse at expressing it — which ROUGE-L's longest-common-subsequence recall tracks and
BLEU-4's n-gram precision averages away. Pose substitution instead changes *which content words
appear*, since different keypoints yield different nouns and names, and that is what n-gram precision
is sharp about. Testable by classifying the edits between prediction pairs in each family; not done.

**What this does to the publishable claim.** It strengthens it. The original observation — that the
field reports sub-1-point BLEU-4 differences at sample sizes where BLEU-4 cannot resolve them — stands.
The reversal adds that **reporting a single metric is unsafe in either direction**, because the metric
chosen may be the one blind to the effect under test. A paper on this now needs both metrics across
both families of effect.

#### Pose-track addendum to §2.5h (2026-10-03): reproduced, and the reversal is effect size rather than power

**Both figures reproduce exactly.** Independent run, `unisign/bootstrap_ci.py`, same 2000 resamples,
400 paired clips aligned by name: BLEU-4 **−2.24 [−3.50, −1.09]** established, ROUGE-L **−1.11
[−2.56, +0.35]** not established. Raw: `results/ci_n400_pose_sub_fps24.json`. The established
pose-substitution cost stands, and so does the ~2.2 BLEU-4 correction to the frontier's absolute
accuracy column.

**The scoping above is right** — §2.5h is careful that the original sub-1-point claim survives. What
this addendum refines is the phrase "the metrics swapped roles", which attributes the reversal to
power when the data say otherwise. Comparing the two metrics *within* each effect — same clips, same
draws, so the comparison is clean:

| effect | metric | delta | CI half-width | \|delta\| / half-width |
|---|---|---:|---:|---:|
| frame rate 30→16 (§2.9B, n=967) | BLEU-4 | −0.35 | **0.72** | 0.48 |
| | ROUGE-L | −1.34 | 1.06 | 1.26 |
| pose substitution (§2.5h, n=400) | BLEU-4 | −2.24 | **1.21** | 1.86 |
| | ROUGE-L | −1.11 | 1.46 | 0.76 |

**BLEU-4's interval is narrower in both cases — by 26 % and 17 %** — in the effect it fails to resolve
just as much as in the one it resolves. So precision did not reverse. What reversed is **effect
size**: the frame-rate effect is 3.8× larger on ROUGE-L (1.34 vs 0.35), the pose-substitution effect
2× larger on BLEU-4 (2.24 vs 1.11).

**"BLEU-4 is underpowered" conflates two separable things**, and separating them strengthens the claim:

* **Precision** — how wide the interval is. BLEU-4 wins both times, so it is the tighter estimator.
* **Sensitivity** — whether the effect registers on the metric at all. This is what varies by effect,
  and it is what decides whether anything is established.

So the defensible statement is not that one metric has more power, but that **BLEU-4 is precise and
partly blind to fluency-type degradation, while ROUGE-L is noisier and responds to it.** That promotes
§2.5h's mechanism hypothesis from a guess to the *explanation* of the pattern, because it predicts
exactly this shape: an effect that changes which words appear lands on n-gram precision, an effect
that leaves word choice alone and degrades ordering lands on longest-common-subsequence recall.

**Caveat on reading the table.** The two effects are at different n (967 and 400), so the half-widths
are **not** comparable *across* rows — only BLEU-4 against ROUGE-L *within* a row, which is the same
clips and the same draws. The quantity that does compare across rows is |delta| / half-width, since
the sample size cancels within each pair.

**What would test the mechanism** instead of restating it: score both effects with a precision-only
and a recall-only metric on the same draws — BLEU-4's precision components against ROUGE-L's recall.
If the split is really precision-versus-recall rather than BLEU-versus-ROUGE, the pattern should
follow the component and not the metric name. Not run.

#### 2.5h addendum 2 (LM track): the refinement is accepted, one figure corrected, and the pattern is broader

**Accepted in full, and it corrects our framing.** "The metrics swapped roles" conflated *precision*
(how wide the interval is) with *sensitivity* (how large the effect is). They do not both reverse.
BLEU-4 is the tighter estimator in both families; what reverses is effect size. The corrected claim —
**BLEU-4 is precise and partly blind to fluency-type degradation, ROUGE-L is noisier and responds to
it** — is better than ours, and it promotes §2.5h's mechanism hypothesis from a guess to the
explanation, since that mechanism predicts exactly this shape.

**One figure corrected: 32 % → 26 %.** Recomputed from `results/adapt_ci_dev.json`, the frame-rate
row's half-widths are BLEU-4 1.486 against ROUGE-L 2.012, so BLEU-4 is **26.1 %** narrower, not 32 %.
The pose-substitution figure of 17 % is right (2.409 vs 2.907 = 17.1 %), as are both effect-size
ratios (3.87× and 2.02×).

**And the pattern is stronger than two cases.** BLEU-4's interval is narrower in **all four**
comparisons we have on shared draws, not just the two quoted:

| comparison | BLEU-4 width | ROUGE-L width | BLEU-4 narrower by |
|---|---:|---:|---:|
| un-adapted: source → 16 fps | 1.486 | 2.012 | **26.1 %** |
| source rate: un-adapted → adapted | 1.327 | 1.476 | 10.1 % |
| 16 fps: un-adapted → adapted | 1.234 | 1.554 | 20.6 % |
| pose substitution (n=400) | 2.409 | 2.907 | **17.1 %** |

Four for four, across two datasets, two pose sources and two sample sizes. "BLEU-4 is the tighter
estimator" is not a two-point observation — it is the consistent direction everywhere we have measured
both on identical draws. That makes the precision-versus-sensitivity split harder to dismiss as an
artefact of which comparisons were picked.

> **Precision caveat, added after the pose track's noise analysis.** The last column is a ratio of two
> bootstrap-noisy widths and inherits noise from both. Bound noise of ≤0.07 — the spread §2.9B
> documents between runs differing only in draws — moves such a figure by about 5 percentage points,
> which is exactly how 31.6 % and 26.1 % arose for the same comparison. **None of the four should be
> read to better than ~±5 points, and they must not be ranked against each other.** What is stable is
> the sign, in all four; a 5-point wobble cannot flip any of them. The one-decimal figures above are
> retained only because they are what the artefacts contain.

The caveat about not comparing half-widths *across* rows (n=967 vs n=400) is right and is why the
table above is read down the last column rather than across. `RESEARCH-STORY-2026-10-03.md` §6 is
updated to the precision/sensitivity framing.

> **SUPERSEDED by the next addendum, 2026-10-04.** The precision/sensitivity split below is
> right about interval widths, but its reading of *which* effect each metric can see was built
> on one of five cells. ROUGE-L establishes pose substitution fine at source rate; the
> divergence is specific to 24 fps. See the following addendum.

#### §2.5h addendum 2 (2026-10-04): the full five-comparison set, and a correction to my own addendum

The pose track's own bootstrap chain finished all five comparisons. `results/ci_n400_*.json`, 400
paired clips, 2000 resamples each.

| comparison | metric | delta | 95 % CI | half-width | verdict |
|---|---|---:|---|---:|---|
| pose substitution @ **source** | BLEU-4 | −2.39 | [−3.49, −1.05] | 1.22 | **established** |
| | ROUGE-L | **−2.12** | **[−3.46, −0.78]** | 1.34 | **established** |
| pose substitution @ **24 fps** | BLEU-4 | −2.24 | [−3.50, −1.09] | 1.21 | **established** |
| | ROUGE-L | −1.11 | [−2.56, +0.35] | 1.46 | not established |
| pruning, our poses @ 24 fps | BLEU-4 | −0.51 | [−1.08, +0.05] | 0.57 | not established |
| | ROUGE-L | −0.51 | [−1.16, +0.08] | 0.62 | not established |
| 24 fps vs source, our poses | BLEU-4 | +0.13 | [−0.49, +0.74] | 0.61 | not established |
| | ROUGE-L | +0.45 | [−0.46, +1.46] | 0.96 | not established |
| 24 fps vs source, ceiling | BLEU-4 | −0.02 | [−0.66, +0.98] | 0.82 | not established |
| | ROUGE-L | −0.57 | [−1.72, +0.44] | 1.08 | not established |

**1. I have to correct my own previous addendum.** I wrote that "ROUGE-L is noisier and responds to
[fluency-type degradation]" while BLEU-4 is "partly blind" to it, and framed pose substitution as the
effect ROUGE-L cannot see. **At source rate ROUGE-L establishes pose substitution perfectly well**
(−2.12 [−3.46, −0.78]). The metric divergence exists at **24 fps only**, so it is not a property of
either metric.

**2. What is actually happening is an interaction with subsampling.** Going from source rate to 24 fps:

| | source | 24 fps | attenuation |
|---|---:|---:|---:|
| ROUGE-L response to pose substitution | −2.12 | −1.11 | **48 %** |
| BLEU-4 response to pose substitution | −2.39 | −2.24 | 6 % |

**Frame subsampling selectively destroys the ROUGE-L signal of pose-extraction quality and leaves the
BLEU-4 signal nearly intact.** Both metrics agree the effect exists at source rate; thinning to 24 fps
halves one of them.

This is the more useful finding, and it has a practical edge: **24 fps is free on accuracy and not free
on measurement sensitivity.** The same subsampling that costs nothing on either metric (rows 3 and 4
above, both tight nulls) halves our ability to *detect* pose-quality differences with ROUGE-L. Anyone
comparing pose front-ends on subsampled data is working with a blunted instrument, and would not know
it from the accuracy numbers.

*Mechanism, speculative and labelled as such:* ROUGE-L scores longest-common-subsequence recall, which
depends on sequence ordering; thinning frames removes ordering detail that distinguishes one pose
source from another, so the ROUGE-L contrast shrinks. Word-choice differences, which BLEU-4's n-gram
precision keys on, survive thinning. Untested — the test named in the previous addendum (precision-only
versus recall-only components on shared draws) would also discriminate this.

**3. 24 fps is confirmed free at n=400 on our own keypoints, on both metrics**: +0.13 [−0.49, +0.74]
BLEU-4 and +0.45 [−0.46, +1.46] ROUGE-L. Tight nulls on the deployed pose source, which is the
strongest form of this result we have.

**4. The n=100 refusal was vindicated.** §2.5g recorded ROUGE-L −1.30 [−2.71, −0.00] for 24 fps vs
source on the ceiling, flagged as sign-established by the script, and declined to promote it on the
grounds that n=976 disagreed and an upper bound of −0.00 is a threshold artefact. At n=400 it is
**−0.57 [−1.72, +0.44], not established.** Refusing a marginal result because a better-powered
measurement disagreed was the right call, and this is the check on it.

**5. Pruning is closer to established than it looks.** Both metrics give −0.51 with |delta|/half-width
of 0.90 and 0.82 — consistent point estimates on two metrics, intervals just straddling zero. It is
still **not** established and must not be quoted as a cost, but the honest expectation is that the full
931-clip split would establish a cost of roughly half a BLEU-4 point. That would not change the
deployment choice, since §5.4 prices pruning at −12 % system energy, half the peak memory and half the
load time.

#### Pose-track reply (2026-10-04): the 26 % correction is accepted, and the percentages need a noise caveat

**Accepted — 26.1 % is right and 32 % was mine.** I computed the ratio from the CI *as quoted in prose*
in §2.9B (BLEU-4 [−1.00, +0.45], ROUGE-L [−2.42, −0.30]) rather than from `results/adapt_ci_dev.json`
(BLEU-4 [−0.966, +0.521], ROUGE-L [−2.346, −0.334]). The raw artefact wins. **A derived quantity
should come from the JSON, not from a rounded figure in a sentence**, and that is the general lesson,
not just this instance.

**But the two are not a rounding difference, and that matters for how the percentages are quoted.**
They are two bootstrap runs of the same comparison differing only in draws. §2.9B already documents
that spread — four seeds, "CI bounds stable to ±0.06" — and the bound differences here are 0.034,
0.071, 0.074, 0.034, essentially within it.

| source | BLEU-4 width | ROUGE-L width | BLEU-4 narrower by |
|---|---:|---:|---:|
| prose CI quoted in §2.9B | 1.450 | 2.120 | **31.6 %** |
| `adapt_ci_dev.json` | 1.487 | 2.012 | **26.1 %** |

**Monte-Carlo noise of ≤0.07 on the bounds moves the derived statistic by 5.5 percentage points.** The
"narrower by X %" figures are ratios of two noisy widths, so they inherit noise from both and are far
less stable than the intervals they come from. None of the four should be read to better than roughly
±5 points.

**This strengthens the four-for-four argument rather than undermining it.** The individual magnitudes
(26.1, 10.1, 20.6, 17.1 %) are not stable enough to interpret one against another — it would be wrong
to say the frame-rate comparison shows a "bigger" precision gap than pose substitution. What is stable
is the **sign**: BLEU-4's interval is narrower in all four, across two datasets, two pose sources and
two sample sizes, and a 5-point wobble cannot flip any of them. The direction is the result; the
magnitudes are decoration.

**Suggested wording for the report**, so this does not get recomputed into a different number by
whoever writes it up: *"BLEU-4's interval is the narrower of the two in all four comparisons measured
on shared draws (by 10–26 %, each ±~5 points from bootstrap noise)."*

### 2.5i Where the −2.24 BLEU-4 pose gap is NOT: keypoints agree closely (2026-10-04)

`task1_rtmpose/10_pose_gap.py`. §2.5h established that substituting our keypoints for the authors'
costs **−2.24 BLEU-4 [−3.50, −1.09]**. The obvious next move is to swap pose models. Before spending
board time on that, this compares the two keypoint sets **directly** — we hold both for the same 400
clips, so the diagnosis is free and needs no GPU.

Both sets are in the OpenASL square-normalised frame (ours via `common/renorm_to_openasl.py`), so a
coordinate difference is directly interpretable. Distances in normalised units, 1.0 = the square
crop's side. Judged only where **both** sources score ≥0.3, since disagreement about a keypoint
neither can see is not an extraction-quality difference. 227 clips had identical frame counts and
could be compared point-for-point; 3.38 M keypoint pairs.

| group | mean | median | p95 | within 0.01 | within 0.02 | within 0.05 |
|---|---:|---:|---:|---:|---:|---:|
| body | 0.0099 | 0.0050 | 0.0345 | 81.2 % | 91.7 % | 96.9 % |
| left hand | 0.0122 | 0.0060 | 0.0340 | 73.7 % | 90.1 % | 96.7 % |
| right hand | 0.0122 | 0.0058 | 0.0339 | 74.1 % | 90.1 % | 96.8 % |
| face | 0.0047 | 0.0032 | 0.0074 | 98.1 % | 99.0 % | 99.4 % |
| **consumed (69 kpts)** | **0.0098** | **0.0047** | **0.0273** | **81.4 %** | **92.7 %** | **97.5 %** |

**Mean signed offset over the consumed keypoints: dx +0.0016, dy +0.0032, magnitude 0.0036.** Near
zero, so **this is not a misaligned coordinate system** — §2.5d's frame fix did its job and no
systematic framing error remains. What is left is per-keypoint scatter.

**So a different pose model is unlikely to be the answer**, and that is the practical conclusion. Our
keypoints already sit a median of **0.47 % of the frame** from the authors', with no bias, and C2
established that their poses came from the *same* 256×192 RTMW/RTMPose family we are using. A
replacement would have to beat an agreement that is already this close. Note also the ranking: **hands
are the worst group and the face the best**, which matches §P4's finding that hand error is argmax
instability in the pose head — unchanged by precision, and so not fixable by choosing a number format
or, probably, a sibling model.

**What this does not establish.** Close agreement in *mean* coordinate distance does not mean the
residual is harmless. Hand p95 is 0.034 — on a 500 px square that is ~17 px, which is enough to change
a handshape, and handshape is the signal. A small, hand-concentrated error is exactly the kind that
could cost 2 BLEU-4 while looking negligible in this table. **The diagnosis here is where the gap is
not, not where it is.**

### A truncated-download bug, found by the same comparison

One clip in the n=400 set, `N29DdjIj5rw-00:00:11.400-00:00:19.200`, holds **4 frames for a 7.8 s
utterance** — its own `meta.json` records `n_frames=4`, so `data/openasl_fetch.py` produced a
truncated clip and nothing checked it. A scan of all 931 clip metas for effective rate
(`n_frames / duration_s`) finds **exactly two** below 15 fps:

| clip | n_frames | duration | effective rate |
|---|---:|---:|---:|
| `N29DdjIj5rw-00:00:11.400-00:00:19.200` | 4 | 7.80 s | 0.51 fps |
| `RT2ZoSSsP1Y-00:01:38.300-00:01:51.000` | 8 | 12.70 s | 0.63 fps |

The first is in the n=400 evaluation; the second is not. Its effect is visible and instructive — our
pipeline, fed 4 frames, predicts **"No."**, while the authors' poses for the same clip produce
*"Keep everything fresh and alive, I often get bored easily, so that s w…"* against a reference of
*"It keeps things fresh for me, I mean -- I get bored easily, that's why"*.

**It does not explain the gap.** Removing it moves the deployable row 21.57 → 21.67 and the pose
substitution gap **−2.24 → −2.19**, i.e. this artefact accounts for 0.05 of 2.24, about 2 %. The
published figures are left as measured, with this noted.

**The lesson is the same one as §5.5**, one layer up: `openasl_pose_fetch.py` had an unchecked
truncation path and it never fired; `openasl_fetch.py` has one and it **did**, twice. An effective-rate
assertion at fetch time (`n_frames / duration_s` within a sane band of the declared `fps`) is the
one-line guard both needed.

### 2.5j Hand swap: the pose gap is not localised to the hands (2026-10-04)

`jetson/p12_hybrid_hands.sh`, `task1_rtmpose/11_make_hybrid.py`, raw `results/eval_hybrid_*.json`,
CIs `results/ci_hyb_*.json`. §2.5i showed our keypoints agree with the authors' to a median of 0.47 %
of the frame with no systematic offset, but that hands are the worst group and the face the best.
Since handshape is the signal in sign language, a hand-only error was the leading hypothesis for
§2.5h's −2.24 BLEU-4. This tests it directly by exchanging **only** COCO-WholeBody indices 91–132
between the two sources, keypoints and scores together, frame for frame.

One matched 227-clip set for all four rows — the clips where both sources have identical frame counts.
Pruned checkpoint, 24 fps, beam 4, cap 64, batch 1.

| row | keypoints | BLEU-4 | ROUGE-L |
|---|---|---:|---:|
| `theirs_m` | authors' (ceiling) | 23.07 | 42.58 |
| `ours_theirhands` | our body+face, **their hands** | 21.67 | 42.09 |
| `theirs_ourhands` | their body+face, **our hands** | 21.47 | 41.85 |
| `ours_m` | ours (baseline) | 21.31 | 41.28 |

Paired bootstrap, 2000 resamples, same 227 clips:

| comparison | BLEU-4 | 95 % CI | verdict |
|---|---:|---|---|
| total gap (theirs → ours) | −1.75 | [−3.45, −0.36] | established |
| **degrade hands only** | **−1.60** | **[−2.78, −0.57]** | established |
| **degrade body+face only** | **−1.40** | **[−2.91, −0.06]** | established, *marginally* |
| **repair hands only** (ours → ours+their hands) | **+0.35** | **[−0.69, +1.61]** | **not established** |

**1. Both groups carry the damage, and the effect is strongly sub-additive.** Degrading hands alone
costs 1.60 and body+face alone costs 1.40, but degrading **both** costs only 1.75. The two individual
effects sum to 3.00 against a combined 1.75, so each group alone already accounts for most of what
both together cost. That is the signature of a degradation the model can absorb up to a point and then
not — not of one group carrying the signal.

**2. So repairing the hands is not the fix.** Giving our pipeline the authors' hands recovers
**+0.35 [−0.69, +1.61]**, which is **not established** and is at most a fifth of the gap. **The
hand-only hypothesis is rejected as the explanation**, and with it the case for choosing a pose model
on hand accuracy specifically. A model swap aimed at hands would be board time spent on ~20 % of the
problem, at best.

**3. The body+face row is marginal and is not quoted as a clean result.** Its upper bound is −0.06.
Per §2.5g's rule — a bound that rounds to zero is a threshold artefact — it is recorded as
*marginally* established and nothing is built on its exact value. The sub-additivity argument does not
depend on it: it holds on the point estimates and on the hands row alone, which is comfortably
established.

**4. ROUGE-L establishes none of the four.** All four ROUGE-L intervals include zero, at n=227. This is
consistent with the §2.5h addendum's finding that pose-source effects register on BLEU-4 and not
ROUGE-L — and note it is another instance of the two metrics disagreeing about the same data, so
reporting one alone would have given the opposite impression here too.

**What this says about where the gap is.** Combined with §2.5i, the picture is a **diffuse,
distribution-wide difference** rather than a localised failure: coordinates agree closely everywhere,
no group is the culprit, and damage saturates. That is what a *distribution shift* looks like — the
model was trained on the authors' keypoints and never saw our extractor's noise — and it points at
adaptation rather than at a better pose model.

### 2.9D / J6: pose energy across the crop range — the ~7 % bias is really ~3 % (2026-10-04)

`jetson/j6_crop_range.sh`, raw `results/j6_crop/`. Every pose energy number in this document comes
from one clip whose 644×720 crop sits at the **83.4th percentile** of crop area across the 931-clip
split. `unisign/frontier.py` flags that as a **~7 % pessimistic** bias on the absolute energy column
and it is the last caveat on that axis. Five clips spanning **50,660 → 766,080 px (15.1×)**, three
*separate processes* each per §5.6's protocol, FP16 engine, 15 W.

| crop area | wall ms/frame | TRT ms | CPU (imread+pre) ms | avg W | mJ/frame |
|---:|---:|---:|---:|---:|---:|
| 50,660 | 17.01 ± 0.06 | 9.82 ± 0.04 | 5.99 ± 0.05 | 5.317 | **105.7 ± 1.2** |
| 351,360 | 25.32 ± 0.82 | 13.66 ± 0.09 | 10.20 ± 0.71 | 4.879 | 133.0 ± 4.5 |
| 394,560 | 25.07 ± 0.96 | 13.65 ± 0.02 | 9.95 ± 0.89 | 4.893 | 132.1 ± 2.4 |
| 439,200 | 25.84 ± 0.91 | 13.71 ± 0.11 | 10.63 ± 0.81 | 4.869 | 134.6 ± 2.9 |
| 766,080 | 27.41 ± 1.03 | 13.84 ± 0.09 | 12.10 ± 0.87 | 4.848 | **141.9 ± 4.2** |

**15.1× more crop area costs only 1.34× the energy — the cost is 85 % fixed.** Linear fit:
**mJ/frame = 109.7 + 49.5 per Mpx**.

| clip | area | predicted mJ/frame | vs the measured clip |
|---|---:|---:|---|
| the measured clip (83.4th pct) | 463,680 | 132.6 | — |
| **median clip** | 394,560 | 129.2 | **−2.6 %** |
| p10 clip | 184,116 | 118.8 | −10.4 % |

**So the caveat tightens from ~7 % to ~2.6 %**, and the old figure was an overestimate by about 3×.
The ~7 % came from attributing all ~11.1 ms of per-frame CPU time to area scaling; measured, only
about 4.5 ms of it does — CPU stages run 5.99 ms at the smallest crop and 12.10 at the largest, so
roughly 6 ms varies across a 15× area range and the rest is fixed. **At ~2.6 % this is no longer the
dominant uncertainty on the absolute energy axis** — it is now the same order as the ±3 % run-to-run
term L16 addendum 7 settled on, and smaller than the composition's open beam-width error.

**TRT is flat above ~350k px, as it must be** (13.65–13.84 ms at a fixed 256×192 input), which is a
useful internal check: the engine does not see the crop, only the resize of it. The one departure is
the smallest clip at **9.82 ms**, 28 % below the others at the same input size. That is not explained
here and should not be read as an area effect in the engine; the likeliest candidates are DVFS
(that run also drew the most power, 5.317 W, with the GPU busier relative to a shorter CPU stage) or
clip-specific decode behaviour. Flagged, not resolved.

**Small clips are genuinely cheaper** — the p10 clip is 10.4 % below the measured one — so a
deployment dominated by tight crops would do better than our table says. But the median correction is
small, and the relative orderings the frontier is built on were never affected by this at all.

---

### L17 Scaling probe: how much adaptation data this harness needs (`colab_probe_scale.py`, 2026-10-05)

J9 proposes adapting the pose encoder to our own extractor's keypoints, trained on the ~920-clip dev
split, because the train split would need 20,000 YouTube videos through the board extractor. L15's
runs — the ones that worked — trained on **20,000** clips. Nobody had sampled the gap between 300
(moved nothing, by design) and 20,000, so the board extraction was about to be spent on a coin flip.

This measures it on the axis where the 20,000-clip answer is already known. Same adaptation as L15.3
(16 fps, `lr 1e-5`, `ls 0.0`, warmup 0.1, 1 epoch, seed 42, pruned checkpoint), at a **nested** ladder
of training-set sizes — `--limit N` takes the first N clips, so each set is a subset of the next and a
difference between rungs cannot be a draw effect. Every rung is scored on all **967 dev clips** and
paired-bootstrapped against **one** un-adapted baseline re-scored in the same environment.

| n train | Δ BLEU-4 | Δ ROUGE-L |
|---:|---|---|
| 500 | +0.08 [−0.31, +0.48] | +0.40 [−0.13, +0.89] |
| **920** | **+0.44 [+0.01, +0.90]** | +0.40 [−0.20, +0.96] |
| 2000 | +0.38 [−0.11, +0.85] | **+0.73 [+0.10, +1.35]** |
| 5000 | +0.25 [−0.33, +0.80] | +0.54 [−0.21, +1.24] |
| 20000 | +0.41 [−0.24, +1.02] | **+1.02 [+0.27, +1.75]** |

#### The anchor reproduces, so the curve is trustworthy

The 20,000 rung exists as a validity check against L15.3, which ran in a Colab runtime that was later
wiped. It reproduces almost exactly:

| | L15.3 (Block 4) | this run | difference |
|---|---|---|---|
| Δ BLEU-4 | +0.464 [−0.170, +1.064] | +0.41 [−0.24, +1.02] | **−0.054** |
| Δ ROUGE-L | +1.044 [+0.247, +1.802] | +1.02 [+0.27, +1.75] | **−0.024** |

This is also the project's **first independent replication of L15.3**, on a different runtime with a
retrained checkpoint, and GPU reductions are non-deterministic so it was never going to be bit-exact.

**The un-adapted baseline, however, *is* bit-exact across the two runtimes** — BLEU-4 22.787836 and
ROUGE-L 41.589883 in both, to a difference of 0.00e+00. No training is involved in that cell, so beam
search over the same pruned checkpoint is deterministic. That pins down where the −0.054 / −0.024 at
the 20,000 rung comes from: **the retrained checkpoint, not the environment.** Everything the two runs
share reproduces exactly, and only the thing that was retrained differs.

#### ROUGE-L scales with data; BLEU-4 does not move at all

The ROUGE-L column climbs — +0.40 at 500 and 920, +1.02 at 20,000 — and **n=920 reaches 39 % of the
full-scale gain with an interval that includes zero.** The BLEU-4 column is flat and non-monotonic
(+0.08, +0.44, +0.38, +0.25, +0.41): 5,000 clips scores *below* 920, which is not a scaling curve.

**The n=920 BLEU-4 row should not be read as establishing anything.** Its lower bound is **+0.01**, it
sits inside the band measured one rung below it, and across ten intervals one grazing zero is what
chance produces. The honest reading is the ROUGE-L one: **at dev scale this harness delivers roughly
40 % of what it delivers at 20,000 clips, and not measurably more than zero.**

#### The probe cannot answer J9 directly, and this is the important caveat

The two shifts register on **different metrics**, which §2.5h and its addenda have now shown four
times:

| shift | BLEU-4 | ROUGE-L |
|---|---|---|
| frame-rate, n=967 dev (what this probe scales) | −0.35 [−0.97, +0.52] *not established* | **−1.34 [−2.35, −0.33] established** |
| pose-source, n=400 test (what J9 targets) | **−2.24 [−3.50, −1.09] established** | −1.11 [−2.56, +0.35] *not established* |

So the scaling curve exists on a **ROUGE-L** effect, while J9 will be judged on **BLEU-4** — the column
where this probe shows no scale signal, because the frame-rate shift barely registers there at any n.
The probe bounds the data-volume question; it does not transfer to J9's axis.

#### Power: J9 at dev scale is underpowered on the test set as it stands

If J9's adaptation behaves like this curve, dev scale buys ~39 % of full recovery:

| | |
|---|---:|
| pose gap to recover | 2.24 BLEU-4 |
| 39 % of it | **+0.88** |
| half-width of the n=400 paired interval (§2.5h) | **±1.20** |

**+0.88 does not clear ±1.20.** A real effect of that size would come back "not established" — and the
agreed framing then licenses nothing. Running J9 as specified risks buying a null that is about
sample size rather than about adaptation.

**The fix is cheaper than the thing it protects.** The n=400 test pose set is 400 of 976 clips.
Extracting the remaining 576 narrows the interval by about `sqrt(400/976) = 0.64`, to **±0.77**, which
**+0.88 does clear**. That is 576 clips of board extraction against the 967 the training set needs, so
it is the smaller job and it is the one that decides whether J9 can report anything at all.

**Deviations from protocol.** One seed per rung, not three — the rungs are a curve, not a headline, and
L10's seed spread (0.05 BLEU-4 / 0.15 ROUGE-L) is small against these intervals. Run on Colab, not the
rig. `n_boot` 1000, matching `adapt_ci_dev.json` so the anchor is comparable.
