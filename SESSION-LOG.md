# Session Log — Phase 1 (RTMPose on Jetson) — 2026-09-15/16

Continues `jetson-setup-handoff.md`. Read that first for the environment facts. This file records what
was done after it, what changed in the code, the numbers, and where to pick up.

---

## 0. Constraints learned this session

- **No sudo on the Jetson, permanently.** The board is course-owned. Consequences: power mode is fixed at
  15W (mode 0), no `jetson_clocks`, no `nvpmodel -m`. The 7W-vs-15W axis is dropped. A "power budget"
  is framed as a measured average-watts constraint, not a hardware cap. DVFS state (clocks, fan) is
  logged with each run instead of pinned.
- **No `tegrastats` inside the container.** Power is read straight from the INA3221 hwmon sysfs, which
  the container can see. `common/power_logger.py` got a `sysfs` backend and it is now the default.
- **The Mac hosts the MMPose toolchain.** MMPose/mmcv do not build cleanly on the Jetson. ONNX export
  and the PyTorch reference are produced on the Mac (conda env `mmpose`, Python 3.10, torch 2.1.2,
  mmcv 2.2.0, mmpose 1.3.2, numpy 1.26). Only the TensorRT engine build runs on the board.
- **Container facts:** Python 3.12.3, torch 2.8.0a0 (nv25.06), TensorRT 10.11.0.33, CUDA 12.9,
  L4T R36.4.7, numpy 1.26.4, onnx 1.17. `opencv-python-headless` was pip-installed. No onnxruntime.

## 1. Mac environment setup (done once)

```bash
conda create -n mmpose python=3.10 -y && conda activate mmpose
pip install "torch==2.1.2" torchvision onnx onnxruntime opencv-python-headless "numpy<2"
pip install "setuptools<81" wheel
pip install --no-build-isolation mmcv==2.2.0          # setuptools>=81 removed pkg_resources
pip install --no-build-isolation chumpy               # broken build script
pip install cython && pip install --no-build-isolation --no-cache-dir xtcocotools
mim install "mmengine>=0.10" "mmpose>=1.3"
pip install mmdet                                     # mmpose imports it even for RTMPose
# relax mmdet's mmcv<2.2.0 assert:
sed -i '' "s/mmcv_maximum_version = '2.2.0'/mmcv_maximum_version = '2.3.0'/" \
  "$(python -c "import importlib.util as u; print(u.find_spec('mmdet').origin)")"
pip install "numpy<2"                                 # something bumped it to 2.x; torch 2.1 needs 1.x
pip install yt-dlp imageio-ffmpeg                     # OpenASL clip download; no system ffmpeg needed
```

## 2. Model weights

RTMPose-x, COCO-WholeBody (133 kpts), 384x288, from the OpenMMLab model zoo. `mim download` does not
index it (it lives in mmpose `projects/`), so fetched by URL:

- config: `weights/rtmpose-x_8xb32-270e_coco-wholebody-384x288.py`
  (raw.githubusercontent.com/open-mmlab/mmpose/main/projects/rtmpose/rtmpose/wholebody_2d_keypoint/)
- weights: `weights/rtmpose-x_simcc-coco-wholebody_pt-body7_270e-384x288-401dfc90_20230629.pth`
  (download.openmmlab.com/mmpose/v1/projects/rtmposev1/, 227 MB)

## 3. Test data

One OpenASL **test-split** clip, chosen for length 7–10 s and having an OpenASL bbox:

- `Ads-4j06eJY-00:07:37.233-00:07:47.200`, sentence: *"The tweets appeared to solicit donations as part
  of a Bitcoin scam."*
- Downloaded only that segment with `yt-dlp --download-sections`, 720p. First candidate
  (`RNk46VzamZI`) was private; expect more of those.
- Cut to 299 frames at 29.97 fps, each **cropped to the OpenASL signer bbox** (mirrors the dataset's
  own preprocessing), saved as `data/test_frames/f_%04d.jpg`. Metadata in `data/test_frames_meta.json`.
- Purpose: validate the harness and produce first per-frame latency/power. Not for accuracy. Accuracy
  needs the full test split (976 clips) on Colab/GPU rig.

OpenASL repo (tsv + bbox json) was cloned to the scratchpad; re-clone from
`github.com/chevalierNoir/OpenASL` when more clips are needed.

## 4. Export and reference (Mac)

```bash
python task1_rtmpose/01_export_onnx.py --config weights/*.py --checkpoint weights/*.pth --out models/rtmpose-x.onnx
python task1_rtmpose/05_make_reference.py --config weights/*.py --checkpoint weights/*.pth \
  --frames data/test_frames --limit 20 --onnx models/rtmpose-x.onnx --out results/rtmpose_reference.npz
```

- ONNX: input `(1,3,384,288)`, outputs `simcc_x (1,133,576)`, `simcc_y (1,133,768)`. Batch axis dynamic.
  The keypoint axis exported as a symbolic dim; `preproc.json` `num_keypoints` was hand-set to 133
  (nothing downstream reads it).
- Checks that passed on the Mac: standalone preprocess vs mmpose pipeline max diff **0.0**; ONNX
  Runtime vs PyTorch simcc max diff **8.0e-6**.
- Visual overlay on frame 11: face, hands, shoulders correct; lower-body joints scattered (out of frame,
  expected).
- Copied to the Jetson with `scp -r models results data fatisri@192.168.1.70:~/atisri-cv-jetson/`.
  **The code folders had vanished from the Jetson** at that point (unknown cause); re-copied
  `common task1_rtmpose task2_mt5_baseline task3_mt5_onnx README.md requirements-jetson.txt probe_device.sh`.

## 5. Code changes made this session

| File | Change | Why |
|---|---|---|
| `common/trt_runner.py` | `config.clear_flag(trt.BuilderFlag.TF32)` when not fp16; print size via `os.path.getsize` | TRT enables TF32 by default on Ampere; "FP32" engine was off by 1.5e-3. `len(IHostMemory)` is gone in TRT 10.11 |
| `task1_rtmpose/06_compare_trt.py` | `--min-score` (default 0.3): px error judged only on keypoints the reference is confident about; reports worst frame/kpt, per-element simcc mean, NaN check | Out-of-frame joints have flat heatmaps whose argmax flips on 1e-3 noise (336 px "error" that meant nothing) |
| `common/power_logger.py` | New `_SysfsBackend` (INA3221 hwmon, thermal zones, GPU load); `--power-backend auto|sysfs|tegrastats|jtop`, default `auto` | No tegrastats binary in the container |
| `models/preproc.json` | `num_keypoints: 0 -> 133` | Cosmetic |

All three source files were copied to the Jetson after editing. The Jetson copy of `02_build_engine.py`
run that produced the FP32 engine used the *old* runner (TF32 on); the engine was **rebuilt** with the
fixed runner before the passing comparison below.

## 6. Jetson runs (container `lpcv-work`, cwd `/workspace`)

```bash
python3 task1_rtmpose/02_build_engine.py --onnx models/rtmpose-x.onnx              # ~158 s, 231 MB engine
python3 task1_rtmpose/06_compare_trt.py --engine models/rtmpose-x_fp32.engine --reference results/rtmpose_reference.npz
python3 task1_rtmpose/03_infer_frames.py --engine models/rtmpose-x_fp32.engine --frames data/test_frames --out results/rtmpose_trt_fp32.json
python3 task1_rtmpose/04_infer_power.py --engine models/rtmpose-x_fp32.engine --frames data/test_frames \
  --repeat 3 --power-json results/rtmpose_fp32_15W_power.json --power-csv results/rtmpose_fp32_15W_power.csv
```

### Correctness (Phase 1 gate)

| TensorRT FP32 engine vs PyTorch reference, 20 frames | simcc max diff | kpt px (2416 confident) | Result |
|---|---|---|---|
| TF32 on (TensorRT default) | 1.55e-3 | (336 px on an out-of-frame joint) | FAIL |
| TF32 off | **7.2e-6** | **0.0** | **PASS** |

### FP32 baseline, 15W, 3 x 299 frames, batch 1

| Stage | mean ms | p95 ms |
|---|---|---|
| preprocess (CPU: JPEG decode + affine + normalize) | 10.6 | 11.0 |
| TensorRT | 44.4 | 44.7 |
| postprocess (SimCC argmax -> xy, scores) | 1.5 | 1.6 |
| **total** | **56.4** | 57.1 |

| Power (VDD_IN = board total) | |
|---|---|
| idle (3 s before run) | 3.73 W |
| running avg / peak | 11.62 W / 13.33 W |
| VDD_CPU_GPU_CV avg | 7.22 W |
| VDD_SOC avg | 1.84 W |
| energy / frame, total / above idle | 754 mJ / 512 mJ |
| GPU load, Tj | 68 %, 56 C |

Reading: 56 ms/frame ≈ 18 fps, below the 30 fps source rate, so FP32 RTMPose-x cannot run real time.
Preprocess is 19 % of frame time and is CPU work that GPU quantization will not touch. Latency jitter
<2 ms, no throttling. One 10 s sentence ≈ 225 J of pose extraction.

Also in `results/RESULTS.md` (the running results table).

## 7. Rubric gap check for the FP32 baseline

| Rubric | Status |
|---|---|
| Baseline clarity | Pose stage done; mT5 stage still needs the same treatment |
| Deployment validation | Done at 15W; 7W impossible (no sudo) — reframed as measured budget |
| Empirical rigor | Single run, one clip. Needs 3 runs mean±std, several signers, 30-min sustained |
| Compression justification | No accuracy metric yet. Add (a) keypoint agreement vs FP32 on the Jetson for pose ablations, (b) OpenASL BLEU on Colab for end-to-end |

## 8. Where to pick up

Pending outputs the user was asked to paste:

1. FP16 engine, compare, power:
   ```bash
   python3 task1_rtmpose/02_build_engine.py --onnx models/rtmpose-x.onnx --fp16
   python3 task1_rtmpose/06_compare_trt.py --engine models/rtmpose-x_fp16.engine --reference results/rtmpose_reference.npz --simcc-atol 0.05 --kpt-atol-px 2
   python3 task1_rtmpose/04_infer_power.py --engine models/rtmpose-x_fp16.engine --frames data/test_frames \
     --repeat 3 --power-json results/rtmpose_fp16_15W_power.json --power-csv results/rtmpose_fp16_15W_power.csv
   ```
2. Sysfs paths readable in the container for GPU clock / CPU clock / fan, to add to the logger:
   ```bash
   cat /sys/class/devfreq/*/cur_freq 2>/dev/null; ls /sys/class/devfreq/
   cat /sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1 /sys/class/hwmon/hwmon*/pwm1 2>/dev/null
   cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq /sys/devices/system/cpu/online
   ```
3. Copy Jetson results back to the Mac:
   ```bash
   scp "fatisri@192.168.1.70:~/atisri-cv-jetson/results/rtmpose_*" ~/atisri-cv-jetson/results/
   ```

Then: Phase 2 (mT5 plain PyTorch on Jetson, Task 2; needs Uni-Sign OpenASL checkpoint from
`huggingface.co/ZechengLi19/Uni-Sign` + `transformers sentencepiece` in the container), Phase 3 (full
pipeline once on the 299-frame clip), then the accuracy metric. INT8, vocab pruning, variant/frame-rate
sweeps, thermal, LLM correction all remain deferred until Phase 3 works.

## 9. Gotchas worth remembering

- `scp` from the Mac needs a password (no key installed), so Claude Code cannot copy to the Jetson; the
  user runs every scp.
- macOS has no `timeout`; zsh does not word-split unquoted `$var` (use bash for loops).
- Reference `.npz` keys: `names, inputs, simcc_x, simcc_y, keypoints, scores, centers, scales`.
- `05_make_reference.py` uses the full frame as the person bbox; that is why frames are pre-cropped.

---

# Session 2 — 2026-09-17/18 — LM track (Atisri), Mac + Colab

Ownership changed: Atisri owns the language-model track, teammate owns pose + Jetson (`WORKSPLIT.md` v2).
Constraint learned: **no sudo on the Jetson, ever** (course-owned). 15 W only; DVFS logged, not pinned.

Done (all numbers in `results/RESULTS.md`):
- L1 reproduced OpenASL pose-only BLEU-4 22.53 (paper 22.67) on Colab T4 via the authors' repo.
- Metric gotcha: the authors' `out/*_pres.txt` lines carry a `sample: ..., prediction: ` prefix; raw scoring gives 58.7.
- `data/openasl_pose_fetch.py`: pulls any clip's pose pkl from the 32 GB HF archive by HTTP range (18 MB for one clip;
  all 976 test pkls now in `data/openasl_test_pose/`, 555 MB).
- C2: the released poses come from a **256x192 RTMW/RTMPose model** (archive folder `pose-rtmpose-192`, model-space
  distances). Jetson baseline pose model → RTMW-l-m 256x192; RTMPose-x 384x288 is the oversized row.
- C3 `common/pose_to_unisign.py` (bit-exact vs their loader). C5 `unisign/` standalone package (no mmpose/deepspeed),
  strict checkpoint load, CPU or CUDA. C7 `unisign/eval_openasl.py`: 23.16 BLEU-4 on all 976 clips, Mac CPU, 16 min.
- L5 vocab pruning: 587.7M → 243.6M params, 1187 → 571 MB, -0.28 BLEU-4 [CI -0.63, +0.05]. `weights/openasl_pose_only_slt_pruned.pth`,
  HF dir `weights/mt5-base-openasl-pruned/`.
- L6 beam curve: greedy -2.00 (2.7x faster), beam 2 -0.81 (1.7x), beam 4 ref. L12 `unisign/bootstrap_ci.py`.
- L3 CPU timings: decoder 60–80 % of LM time; lever order beam > pruning > encoder.
- L8 ONNX half: three graphs with explicit KV cache verified vs PyTorch (tokens identical, logprob diff 3e-5) on real poses.
  `unisign/onnx_decode.py` (ORT loop), `unisign/trt_decode.py` (TRT loop, untested, for the board).

Environments on the Mac: conda `mmpose` (py3.10, torch 2.1.2, transformers 4.44.2, mmpose 1.3.2, rtmlib) for pose
export + eval; conda `unisign` (py3.11, torch 2.14, transformers 4.57.6, onnx, onnxruntime) for the ONNX export.

Open: C0 git init (user decision pending). Jetson queue for the teammate (`WORKSPLIT.md` §5): J1 keypoints JSON,
J2 on-board LM run full + pruned, J4 TRT engines + `trt_decode`. Next on the LM track: L9 weight-only INT8, C8 training
harness (needs the 97 K train pose pkls, ~30 GB via `openasl_pose_fetch.py --split train`), L13 LLM correction.
