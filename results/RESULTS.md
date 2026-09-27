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
48 past K/V; 25 outputs). Synthetic verify at T=264, 12 steps: encoder max diff 4.1e-6, per-step
logits ≤ 2.2e-5, tokens identical. Real-pose verify (`unisign/onnx_decode.py`, Bitcoin clip, greedy):
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

