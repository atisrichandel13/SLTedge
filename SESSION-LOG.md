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

---

# Session 3 — 2026-09-18 — Pose track (Tushar), P0 onboarding on the Jetson

Board: `jetson-lpcv-03` = `192.168.1.73` on the lab LAN, reachable only through WireGuard (hostname
DNS is flaky; use the IP). Login `tgoyal`, password auth, **no sudo**, docker group. This is a shared
course board: 24 student homes and other students' `lpcv:fall2026` containers running. Atisri's
account `fatisri` is on the same host but their home is unreadable, so the Phase-1 artefacts
(`data/test_frames/`, `models/rtmpose-x.onnx`, `results/rtmpose_reference.npz`, `results/rtmpose_*`
raw outputs) are **not** available to the pose owner until Atisri copies them (J1).

Done:
- Own container per `Jetson-access.md`: `jetson/Dockerfile` (base `nvcr.io/nvidia/pytorch:25.06-py3`,
  already on the board, arm64) + `jetson/run.sh` (`build|probe|run|shell|up|exec|down`) +
  `jetson/README.md`. Image `slt-jetson:25.06`, runs as the host UID so bind-mounted files stay
  user-owned. Stack identical to Session 1: Python 3.12.3, torch 2.8.0a0 nv25.06, TensorRT 10.11.0.33,
  CUDA 12.9, numpy 1.26.4; added cv2 4.11, onnx, onnxruntime (CPU), transformers 4.57.6, sentencepiece.
  NGC banner parts removed from the entrypoint. Build ≈ 1 min (pip layer 41 s).
- Repo synced to `~/sign-lang-project` on the board (rsync over ssh; `.git` excluded).
- Checks passed in the container: GPU visible (Orin), `03_infer_frames.py --help`, `power_logger.py`
  self-test reads INA3221 sysfs without root (VDD_IN 3.3 W idle, Tj 45 °C), `rtmpose_utils.py`
  self-test, `import unisign.unisign_infer / unisign.trt_decode / common.trt_runner`.
- `common/envinfo.py`: power mode now read from `/var/lib/nvpmodel/status` (bind-mounted read-only
  by `run.sh`) when the `nvpmodel` binary is absent → reports `mode 0 (15W)`.
- Host facts: 6 CPUs, 7.4 GiB RAM, 40 GB free, host TensorRT 10.3 + `trtexec`, `tegrastats` and
  `jtop` exist on the host (not needed; sysfs backend is used). Internet works from the board (PyPI,
  GitHub, Hugging Face); `download.openmmlab.com` root answers 403 (file URLs untested).

Blocked / next:
- **J1** needs Atisri: copy `results/rtmpose_trt_fp32.json`, `results/rtmpose_fp32_15W_power.{json,csv}`,
  `results/rtmpose_reference.npz`, `models/rtmpose-x.onnx`, `data/test_frames/` (299 jpg) to the shared
  Drive or to `/home/tgoyal/` on the board (their `fatisri` home is mode 700).
- **P1** (RTMW-l-m 256×192 baseline) can start without J1 if the test frames are regenerated from
  `data/test_frames_meta.json` (yt-dlp section download + crop, SESSION-LOG §3) and the ONNX is fetched
  via `rtmlib` on the board; reference would be ONNX Runtime CPU instead of mmpose PyTorch.
- J2 / J4 need Atisri's `weights/` (mT5-base, `openasl_pose_only_slt*.pth`, `models/mt5_pruned_onnx/`).

### Session 3, later: wrong base image variant (found 2026-09-18 evening)

The first `slt-jetson:25.06` image was built on `nvcr.io/nvidia/pytorch:25.06-py3`, which was already
pulled on the board. Torch saw the GPU, but every TensorRT build failed with
`Error Code 9: API Usage Error (Target GPU SM 87 is not supported by this TensorRT release.)`, also
with `trtexec` on a 2 KB ONNX, also as root with the stock entrypoint. Cause: that tag is the **SBSA
(server ARM) build** (`docker history` shows `L4T=0`, CUDA runtime from `targets/sbsa-linux`). The
course image `lpcv:fall2026` is built from the **iGPU variant** (`L4T=1`), tag
`nvcr.io/nvidia/pytorch:25.06-py3-igpu`; same torch / TensorRT version strings, different binaries.
Fix: `jetson/Dockerfile` now uses `pytorch:25.06-py3-igpu` (pulled 2026-09-18). Lesson: on Jetson,
always check `docker history <image> | grep L4T=` or run `trtexec` on a tiny ONNX before trusting a
container's TensorRT.

Large files staged on the board for the queue (to be deleted after results are pulled back, see
`jetson/run.sh clean-large`): `data/test_frames/`, `models/rtmpose-x.onnx`,
`results/rtmpose_reference.npz`, `weights/{mt5-base,mt5-base-openasl-pruned,openasl_pose_only_slt*.pth}`,
`models/mt5_pruned_onnx/`, `data/openasl_ref_pose/`. Source: `model-data-lpcv/` on the Mac (Atisri's
large-file clone, 7.9 GB, gitignored in this repo). Missing from it: the three raw Phase-1 outputs
(`results/rtmpose_trt_fp32.json`, `results/rtmpose_fp32_15W_power.{json,csv}`); they will be
regenerated here by rerunning the FP32 pipeline (a reproduction on a second board).

P1 prep: RTMW-l-m 256x192 ONNX fetched from the OpenMMLab SDK zip (rtmlib "lightweight" pose model)
to `models/rtmw/rtmw-l-m_256x192.onnx` (129 MB, opset 11, input `input` 1x3x256x192 dynamic batch,
outputs `simcc_x` 133x384, `simcc_y` 133x512) with `models/rtmw/preproc.json` written from the zip's
`pipeline.json` (same mean/std as RTMPose-x, padding 1.25, simcc_split_ratio 2). Reference for it =
`task1_rtmpose/05b_make_reference_ort.py` (ONNX Runtime CPU; no mmpose checkpoint for this export).

### Session 3 checkpoint (2026-09-18, ~22:15 board time) — where to pick up

- Image `slt-jetson:25.06` rebuilt on `pytorch:25.06-py3-igpu`; `trtexec` tiny build **PASSED**,
  `envinfo` shows Orin / TRT 10.11.0.33 / mode 0 (15 W). All large inputs for J1-reproduction, J2, J4
  and P1 are staged on the board under `~/sign-lang-project/` (6.4 GB) + `models/rtmw/` (129 MB).
- First real engine build (`02_build_engine.py --onnx models/rtmpose-x.onnx`, default
  `--workspace-gb 2`) ran 157 s then failed: `Cuda Runtime (out of memory)` when the builder
  requested 512 MB. Board had 5.7 GB available afterwards and the other student's container was idle,
  so it was a transient peak (builder tactic timing + 2 GB workspace + page cache from 18 GB of
  transfers just before). **Next step: retry with `--workspace-gb 1`**, then `06_compare_trt.py`,
  `03_infer_frames.py`, `04_infer_power.py --repeat 3`, pull `results/rtmpose_*` back, record in
  `results/RESULTS.md` as the second-board reproduction, then J2, J4, P1.
- Logs on the board: `results/logs/build_rtmposex_fp32.log`, `results/logs/compare_rtmposex_fp32.log`.

## Session 4 (2026-09-18, evening) — J1 reproduction on lpcv-03; the 15W/25W finding

- Engine build retry with `--workspace-gb 1` succeeded: 158 s wall, 140 s engine generation, TRT
  allocator peak 384 MiB GPU, `models/rtmpose-x_fp32.engine` 259 MB. The 2 GB-workspace OOM was
  contention on the shared 8 GB unified memory, not a hard limit.
- `06_compare_trt.py` vs the Mac PyTorch reference: simcc max 6.9e-6, all 2416 confident keypoints at
  0.0 px, PASS. `03_infer_frames.py` -> `results/rtmpose_trt_fp32.json`; `04_infer_power.py --repeat 3`
  -> `results/rtmpose_fp32_15W_power.{json,csv}`. All three now in the repo (Atisri's C4 input).
- **Numbers do not match Atisri's Phase-1 row**: TRT 66.7 ms vs 44.4 ms, 8.2 W vs 11.6 W. Root cause:
  lpcv-03's GPU is DVFS-capped at 612 MHz (`/sys/class/devfreq/17000000.gpu/max_freq`), which is the
  nvpmodel mode 0 = 15W cap in `nvpmodel_p3767_0003_super.conf`. Mode 1 = 25W caps at 918 MHz and is
  the conf's DEFAULT; 66.7 x 612/918 = 44.5 ms. So the Phase-1 row was taken at the 25W cap. The
  status file alone is not enough evidence of the power mode; the clock must be logged.
- Fixes: `common/power_logger.py` sysfs backend now samples `gpu_MHz`, `cpu0_MHz`, `fan_pwm`,
  `fan_rpm` every 100 ms (CSV columns + `aux_avg`); `common/envinfo.py` reports `gpu_max_MHz`,
  `gpu_min_MHz`, `cpu_max_MHz` and has the correct mode table (0=15W, 1=25W, 2=MAXN_SUPER, 3=7W).
  Power run repeated with the new logger: GPU 612 MHz on every busy sample. RESULTS.md Task 1 table
  now has a Board / GPU MHz column and a discrepancy note; the lpcv-03 mode-0 row is the FP32 baseline.
- `jetson-lpcv-04` (192.168.1.74, same login) checked at the user's suggestion: also mode 0 / 612 MHz,
  no containers, home has only the course assignment1 files. No engine there; nothing to gain.
- Still on the board: engine (259 MB), ONNX, frames, weights for J2/J4/P1. Delete after those stages.

### Session 4 checkpoint — where to pick up

J1 reproduction is done and recorded. Next in the queue: **J2** (Uni-Sign on-board run, full then
pruned checkpoint, `unisign.unisign_infer ... --repeat 3`), then **J4** (mT5 TensorRT engines +
`trt_decode`), then **P1** (RTMW-l-m 256x192: `05b_make_reference_ort.py`, build, compare, latency,
power). Inputs for all three are already staged on lpcv-03.

## Session 5 (2026-09-18, evening) — J2 on-board LM run done; J4 engines

- **J2 = L4 done** on `jetson-lpcv-03` (mode 0, GPU 612 MHz). `unisign.unisign_infer --device cuda
  --repeat 3` with the full and the pruned checkpoint on the Bitcoin clip: text, token ids and
  log-probs identical to the Mac (`c5_authors_pose_Ads-4j06eJY.json`). Full: GCN ~85 / encoder 57 /
  decoder 1653–1664 ms (24 tokens, 69 ms/token), peak GPU 2.41 GB. Pruned: decoder 1330 ms (55
  ms/token), peak 1.03 GB. Files `results/l4_jetson_{full,pruned}_fp32.json`, logs `results/logs/j2_*`.
- Finding: the board decoder is 5–7× slower than the Mac CPU while the encoder is faster. HF
  `generate()` is launch/Python-bound on the 15W ARM cores; this is the number the TensorRT KV-cache
  engines (J4) have to beat. Pruned decoder drifted 1240 → 1330 ms over three runs (noise ±5%).
- Three failures fixed in `unisign/model.py` (all only reachable on CUDA / the 8 GB board):
  1. `PrunedTokenizer.decode` indexed a CPU tensor with CUDA ids → `ids.cpu()`.
  2. Full FP32 model OOMed in `.to("cuda")` twice (NvMap error 12, then CUDA OOM): 2.3 GB CPU copy +
     2.3 GB GPU copy with ~1.8 GB free outside page cache. Now `from_pretrained(device_map=device)`
     for the full checkpoint, checkpoint `torch.load(mmap=True)` and freed before the device copy.
  3. `prune_vocab` now indexes with `keep` on the weights' device. The pruned path deliberately still
     builds on CPU and slices first (1.03 GB peak vs 2.57 GB when sliced on the GPU).
  Mac (CPU) behaviour is unchanged: `device_map` is only passed for non-CPU devices.
- **J4 FP32 = PASS.** Engines `models/mt5_pruned_onnx/{encoder,decoder_init,decoder_step}_fp32.engine`
  (341/616/558 MB, built in 15/23/13 s, builder peak ≤ 698 MiB) with `--enc-len 1,264,512
  --dec-len 1,1,128 --workspace-gb 1`. `unisign.trt_decode`: tokens identical to PyTorch, max
  |Δlogprob| 5.6e-4, decoder 460 ms / 24 tokens = **19 ms/token vs 55 in PyTorch**; TRT FP32
  encoder 82 ms is slower than PyTorch's 57 (TF32 off → FFMA kernels). `results/l8_jetson_trt_fp32.json`.
- **Board memory finding (root cause of every OOM today)**: nvmap (the Jetson GPU allocator) only
  takes from `MemFree` and does not reclaim page cache. Decoder engine builds failed on 57–613 MB
  allocations with 2.7 GB free and 3.9 GB cached; engine deserialisation failed the same way after
  the PyTorch reference had filled the cache. No root, so `jetson/drop_file_cache.py` evicts our own
  files with `posix_fadvise(DONTNEED)`; `03_build_engines.py` and `trt_decode.py` call it on the board
  automatically, and `trt_decode.py` now frees the PyTorch reference before loading engines (it also
  prints MemFree before/after: engines take 2.07 GB). Documented in `jetson/README.md`.
- **J4 FP16 = numerically broken** (mT5 FP16 overflow: every step token 0, logprob −ln 26078). The
  FP16 encoder is 3–4× faster than FP32 TRT, the FP16 decoder step is not faster at all (18 vs 19
  ms/token) → the step loop is host-bound. Added `--bf16` to the runner/build/decode scripts; BF16
  engines + decode run next. `results/l8_jetson_trt_fp16.json` kept as the negative result.
- **J4 BF16**: builds and decodes coherently but diverges from FP32 greedy at step 3 (22 vs 24
  tokens, max |Δlogprob| 1.1); 21.8 ms/token, no faster. Conclusion recorded in RESULTS.md L8.2: the
  deployable LM engine set on this board is FP32; a real FP16 path needs mixed precision (FP32
  residual/FFN-out). **J4 done.** mT5 engines deleted from the board (rebuild ≈ 1 min each from the
  ONNX, which stays with the weights until the user decides).
- **P1 done**: RTMW-l-m 256×192 FP32 on `jetson-lpcv-03` (15W). `05b_make_reference_ort.py` (ORT CPU,
  20 frames, 8 s) → `02_build_engine.py --workspace-gb 1` (158 MB, 123 s) → `06_compare_trt.py`
  PASS (0.0 px, simcc 2.0e-5) → `03_infer_frames.py` → `04_infer_power.py --repeat 3` (897 frames):
  pre 6.1 / TRT 25.6 / post 1.2 / total 33.0 ms, 6.53 W avg, 3.62 W idle, 275 mJ/frame (123 dyn),
  GPU 60 %, Tj 51 C. Busy-sample GPU clock 598 MHz mean (governor 408–612). 2.6× faster and 2.7×
  less energy than RTMPose-x on the same board. Row + paragraph in RESULTS.md Task 1.
- Board cleanup at this checkpoint: all engines deleted (RTMPose-x FP32, RTMW FP32, mT5 ×9; each
  rebuilds in 15 s–2.5 min from the ONNX). Kept on the board for P2/later stages: `models/rtmpose-x.onnx`,
  `models/rtmw/`, `models/mt5_pruned_onnx/*.onnx` (1.5 GB), `weights/` (4.8 GB), `data/`, reference npz.
  Ask the user before wiping weights/ONNX (`jetson/run.sh clean-large`).

### Session 5 checkpoint — where to pick up

J2, J4, P1 are done and recorded (RESULTS.md L4, L8.2, Task 1; PROJECT-GUIDE 1.3, 1.4, 2.1).
Next in the Teammate queue: **P2** (FP16 engines for RTMW-l-m and RTMPose-x: `02_build_engine.py
--fp16 --workspace-gb 1`, gate with `--simcc-atol 0.05 --kpt-atol-px 2`, then `04_infer_power.py
--repeat 3` → `results/*_fp16_15W_power.json` rows), then the rest of section 2 of the guide. Run
`python3 jetson/drop_file_cache.py` on the board before every build/engine load. Nothing is running
on the board. Uncommitted: see `git status` (all of today's work).

## Session 6 (2026-09-18, night) — P2 FP16 pose engines

- **RTMW-l-m FP16** (`02_build_engine.py --fp16 --workspace-gb 1`, 534 s, 68.7 MB): 13.8 ms TRT,
  19.7 ms total (40 fps end to end), 4.93 W, **127 mJ/frame**, Tj 50 C. Gate at `--simcc-atol 0.05
  --kpt-atol-px 2`: simcc 0.039 (inside), keypoints max 6.50 px (outside) → FAIL. But every error is
  an exact multiple of one simcc bin (2.17 image px here), 88.5 % of 2458 confident keypoints are
  bit-exact, 99.8 % within 5 px. Re-gated at `--min-score 0.7`: worst error exactly one bin, 100 %
  within 5 px. The max-px criterion is the wrong instrument for argmax decoding.
- **RTMPose-x plain FP16 is unusable**: simcc 0.649, keypoint p90 130 px, 383 px worst.
  New `task1_rtmpose/08_fp16_range_scan.py` (ONNX on ORT CPU with every intermediate exposed, peak
  |activation| vs the FP16 max) found one cause in 376 tensors: the head's ScaleNorm sum of squares
  `/mlp/mlp.0/ReduceSum` peaks at **142 293**, with `/mlp/mlp.0/Pow` at 23 565. The same scan over
  RTMW's 398 tensors reports **zero** overflowing tensors, which is why its FP16 engine is fine.
- **Mixed precision fixes it**: new `--fp32-layers <regex>` in `02_build_engine.py` /
  `common/trt_runner.py` (sets `OBEY_PRECISION_CONSTRAINTS`, pins matching compute layers to FP32,
  skips shape/constant/cast layers). `--fp32-layers 'mlp\.0'` pinned **7 of 465** layers: simcc
  0.649 → 0.023, within-2 px 67.7 → 96.1 %, within-5 px 71.7 → 99.5 %, for 23.3 vs 22.7 ms and 264 vs
  254 mJ/frame. Engine 132.3 MB. This is the RTMPose-x FP16 row to use.
- **Clock caveat, new**: the governor drops the GPU as work gets lighter — FP32 RTMW ran at 598 MHz
  mean over busy samples, FP16 RTMW at **310 MHz**, FP16 RTMPose-x at 413 MHz. Cross-precision
  latency ratios are therefore understated; energy per frame is the robust comparison.
- **Incident**: a bare `rsync board:results/ results/` overwrote `results/RESULTS.md` with the
  board's stale copy and reverted the J2/J4/P1 write-ups. Recovered by replaying the edits out of the
  session transcript. Added `jetson/pull_results.sh` (excludes `*.md` and `*.npz`) — use it, never a
  bare rsync of `results/`. Root cause is that the board carries a full copy of the repo.
- **Committed** `a33443d` (71 files): all of J2/J4/P1/P2-so-far plus the jetson/ tooling. Before
  this the whole day existed only in one uncommitted working tree.

### Session 6 checkpoint — where to pick up

P2 is done and recorded (RESULTS.md Task 1 rows + the "P2: FP16 pose engines" note; PROJECT-GUIDE
2.2). Pose stage now: **RTMW-l-m FP16, 19.7 ms/frame, 127 mJ/frame, 40 fps at 15 W**, vs 79.2 ms and
733 mJ for the original RTMPose-x FP32 baseline.

Next in the Teammate queue: **P3** (guide 2.3) — `data/openasl_fetch.py`, ≥5 test clips from distinct
signers and ≥300 calibration frames from ≥5 signers into `data/calib_frames/`. Everything after it
(P4 keypoint agreement, C7 pose→BLEU, INT8 calibration) is blocked on having more than 20 frames of
one clip. Run `python3 jetson/drop_file_cache.py` before every board build/engine load, and pull with
`jetson/pull_results.sh`.

## Session 7 (2026-09-26) — P3 data

- New `data/openasl_fetch.py`. OpenASL is distributed by reference (YouTube id, time range, signer
  bbox, sentence), so frames have to be fetched per clip. The script joins the release tsv with
  `bbox-v1.0.json`, picks one clip per YouTube video (distinct video = signer proxy; OpenASL has no
  signer field), downloads only that section with yt-dlp, crops and writes JPEG frames, and records
  every skip with its reason.
- **Crop recipe**: normalised bbox scaled to the frame and **clamped** to it, native resolution,
  native fps. This matches `data/test_frames_meta.json` from the baseline clip, *not* OpenASL's own
  `prep/crop_video.py`, which squares the box, black-pads the overflow and resizes to 224. We keep
  the extra pixels because the pose models do their own affine and black padding would change the
  input. Documented in the script header: change it for one clip and the rows stop comparing.
- **Test clips**: 5/5 from 5 distinct videos, **1510 frames**, 62 MB, 0 failures — `UoU3ZSuTef4`,
  `ImwA3Ctckfk`, `ZdEwfVNtSmw`, `y7KIrON1uco`, `ixq65EiuJ_c` (all 1280x720, ~30 fps). With the
  baseline `Ads-4j06eJY` clip that is **6 signers**. Visual check on two clips: signer centred,
  hands in frame, one studio and one home setting, no black bars.
- **Calibration**: 6/6 train videos, 60 strided frames each = **360 frames**, 16 MB, in
  `data/calib_frames/` with `calib_index.json`. Ready for INT8 calibration.
- **yt-dlp version matters**: the 6-month-old build in the `yt-crawl` env fails every section
  download with `ffmpeg exited with code 8` / HTTP 403, because `--download-sections` hands ffmpeg a
  URL bound to the player client that yt-dlp used. A current yt-dlp (2026.08.19) in an isolated
  scratchpad venv works with no cookies and no sign-in. Pass it with `--yt-dlp <path>`. The script
  also takes `--cookies-from-browser` if YouTube ever demands sign-in; not needed today.
- **Note**: the Mac `mmpose` conda env from Session 1 no longer exists, and no env on this Mac has
  cv2. The fetch script therefore shells out to ffmpeg instead of using cv2. Anything that needs
  mmpose again (new ONNX exports, PyTorch references) will have to rebuild that env.

### Session 7 checkpoint — where to pick up

P3 done (guide 2.3). Next: **P4** (guide 2.4) `task1_rtmpose/07_kpt_agreement.py` — % of confident
keypoints within 1 / 2 / 5 px of the FP32 engine, per keypoint group (body / face / hands), over all
1510 frames rather than the 20-frame gate. This is the measurement that replaces the max-px gate
that failed both FP16 engines on a single argmax flip. Then C7, the pose→BLEU path.

Board runs need the frames pushed (78 MB) and the engines rebuilt from the staged ONNX; delete both
afterwards. Pull results with `jetson/pull_results.sh`, never a bare rsync.

## Session 8 (2026-09-26) — P4 keypoint agreement

The board was unreachable for this session: the lab WireGuard tunnel is a full tunnel
(`AllowedIPs = 0.0.0.0/0`), so bringing it up moves the default route into the lab and Claude Code's
own connection dies with it. Sockets opened *before* the tunnel comes up keep using `en0` because
macOS caches the route on the socket, which is why it looked like the two could coexist if started in
the right order. The fix is a split tunnel: `AllowedIPs = 192.168.1.0/24` and no `DNS =` line (we
address the board by IP with `HostKeyAlias`, so lab DNS is not needed). Until that is set, board and
Claude access are mutually exclusive.

P4 did not need the board. The 299-frame keypoint dumps from the P1/P2 board runs are already in
`results/` (`rtmw_trt_fp32`, `rtmw_trt_fp16`, `rtmpose_trt_fp32`, `rtmposex_trt_fp16`,
`rtmposex_trt_fp16mixed`), which is 36 k confident keypoints per config against the 20 frames the
gate used.

- **`task1_rtmpose/07_kpt_agreement.py`** (new): pairs two keypoint dumps, Euclidean px error per
  keypoint, masked by the reference score, reported per COCO-WholeBody group plus `unisign_used` (the
  69 keypoints `common/pose_to_unisign.py` keeps). Also reports score-threshold crossings, since
  Uni-Sign zeroes joints under 0.3 and a crossing changes the LM's input at 0 px of motion. Pure
  numpy — the four Uni-Sign index slices are mirrored locally with an assert against the real module,
  because `pose_to_unisign` imports torch and no env on this Mac has torch any more. `--ref`/`--test`
  take several per-clip dumps and pair them on the clip id parsed out of `<config>__<vid>.json`.
- **Results** (RESULTS.md "P4", three `results/kpt_agreement_*.json`): both engines the max-px gate
  failed agree with FP32 on 99.5 % (RTMW FP16) and 99.3 % (RTMPose-x FP16 mixed) of consumed
  keypoints within 5 px; plain FP16 RTMPose-x sits at 86.5 %, its face branch at 49.6 % with a median
  error of 108 px. So the gate's single-keypoint verdict was hiding the difference between a bin flip
  and a real overflow.
- **Two findings worth carrying.** (1) The ≤1 px and ≤2 px columns are one simcc bin apart — 2.17 px
  for RTMW, 1.45 px for RTMPose-x — so for RTMW they are identical by construction and only the 5 px
  column compares across models. (2) Hand agreement is *identical* in the broken and repaired
  RTMPose-x engines (98.87 % vs 98.93 %), so hand error is argmax instability on low-confidence
  joints, not FP16 precision, and no fix so far touches it. Hands are the accuracy floor in every
  config while the face is essentially exact — the wrong way round for sign language.
- **`jetson/p4_clips.sh`** (new): board driver for the five-signer version. Rebuilds each engine from
  the staged ONNX (the board keeps none between sessions), runs `03_infer_frames.py` per clip per
  config into `results/kpts/<config>__<vid>.json`, drops the page cache before every build and load,
  and skips dumps that already exist so it survives an interruption. Those dumps are also P5's input.

**Session 8 checkpoint — where to pick up.** Committed; working tree clean. Nothing running, nothing
left on the board (it was never reached). Next, in order: (a) set the WireGuard split tunnel, (b) push
`data/clips/` (62 MB) to the board and run `jetson/p4_clips.sh`, then `jetson/pull_results.sh` and
re-run `07_kpt_agreement.py` across all five clips into
`results/kpt_agreement_<config>_5signers.json` — this is the only part of P4 still outstanding, and
it tests generalisation across signers rather than precision damage; (c) **C7/P5** (guide 2.5), the
pose→BLEU path, which consumes the same `results/kpts/` dumps through
`common/pose_to_unisign.py` → `unisign/eval_openasl.py --poses`. Note that BLEU needs torch, which
this Mac no longer has: rebuilding an env is a prerequisite for 2.5 either way.


## Session 9 (2026-09-26) — split tunnel, the 5-signer P4 run, and a crop bug

**The WireGuard split tunnel works.** A second tunnel `lab-split` with `AllowedIPs = 192.168.1.0/24`
and no `DNS =` line gives board access and Claude access at once: `192.168.1.73/32 → utun12` while the
default route stays on `en0`. The original profile was a full tunnel (`AllowedIPs = 0.0.0.0/0`), which
moved the default route into the lab. This is why board access previously seemed to depend on the
*order* of operations — sockets opened before the tunnel came up survived, because macOS caches the
route on the socket, so an already-running Claude session kept working while a new one could not start.

**P4 is complete over five signers.** Pushed `data/clips/` + `data/calib_frames/` (78 MB) and ran
`jetson/p4_clips.sh` on `jetson-lpcv-03` at 15 W (`pmode:0000`): 4 engines built, 20 dumps, ~35 min.
`results/kpt_agreement_{rtmw_fp16,rtmposex_fp16mixed}_5signers.json`. The verdict from the single-clip
run holds and improves: **99.81 % (RTMW FP16) and 99.75 % (RTMPose-x FP16 mixed)** of Uni-Sign-consumed
keypoints within 5 px of FP32, over 1510 frames and ~104 k confident keypoints per config.

**The finding worth keeping: the simcc bin is bbox-dependent, so the ≤1/≤2 px columns are not a
measurement of accuracy.** The smallest non-zero error in each clip *is* one bin, and it matches
`max(crop_w, crop_h × 0.75) × 1.25 / (input_w × simcc_split_ratio)` to two decimals — 1.71–2.45 px for
RTMW across the five clips, 1.14–1.63 px for RTMPose-x. RTMW's ≤2 px column therefore reads 99.2–99.4 %
on the three clips whose bin is under 2 px and 86.5–87.9 % on the two whose bin is over it: a 13-point
swing from signer framing alone. The ≤5 px column is the only comparable one, because 5 px is "within
two bins" for every clip in both models (largest bin 2.45 px). Any future gate must be stated in bins.

Hands stay the accuracy floor (99.69 % / 99.61 % within 5 px, lowest of every group) and the worst
confident keypoint is a hand joint in both engines — 122.7 px at ref score 0.337 for RTMW, 229.1 px at
0.470 for mixed. Both maxima roughly doubled versus the single clip, which is what a tail driven by
rare low-confidence argmax flips does as the sample grows: more evidence it is instability, not
precision. Also noted: RTMPose-x produces only 30 confident foot keypoints against RTMW's 1527 out of
9060. Feet are not among the 69 keypoints Uni-Sign consumes, so nothing downstream changes.

**Bug found and fixed: every clip's recorded crop was 1 px too large.** ffmpeg's `crop` filter rounds
width and height *down* to even for yuv420p chroma subsampling, so all five clips with an odd dimension
delivered JPEGs 1 px smaller than `meta.json` claimed. `pose_to_unisign.load_keypoints_json` normalises
by exactly those numbers, so every coordinate would have been scaled by e.g. 754/755. Fixed in three
places: `crop_xywh` now rounds with `& ~1` and records why, the five `meta.json` files and `index.json`
were corrected in place, and `common/dumps_to_pkl.py` asserts normalised keypoints land within
[0, 1.05] rather than trusting the metadata. The error is ~0.1 % and largely absorbed by Uni-Sign's own
renormalisation, but nothing would have flagged it.

**The authors' own poses are now a ceiling row.** `data/openasl_pose_fetch.py` pulled the released
`pose-rtmpose-192` pkls for exactly our five test clips (35 MB, by HTTP range request out of the 32 GB
8-part archive) into `data/openasl_5clip_pose/`. 2.5 can therefore compare our engines against *the
authors' extractor on the same five sentences*, which isolates the cost of our pose front-end from the
cost of the checkpoint. Caveat to carry into the write-up: BLEU over 5 sentences is far too noisy to
read as an absolute score — the signal is the ordering against that ceiling plus the count of
token-identical predictions between configs.

Both dump sets are gitignored (`results/kpts/` 46 MB, `data/openasl_5clip_pose/` 35 MB); the derived
`results/kpt_agreement_*.json` are tracked, and `jetson/p4_clips.sh` regenerates the dumps.

**2.5 pose→BLEU is done too, and it decides the front-end.** All four configs plus the authors' released
poses as a ceiling row, scored on the board through the same checkpoint (`results/eval_*.json`).
Headline: **RTMW FP16 produces 5/5 token-identical predictions to its own FP32 engine**, so FP16 costs
exactly nothing in output text — the max-px gate is now refuted twice over, once on keypoints and once
on the translation. RTMPose-x mixed precision is *not* neutral (3/5), which is the same ordering P4's
≤5 px figure gave (99.81 % RTMW vs 99.75 % mixed), and the first sign that the keypoint metric predicts
downstream behaviour.

RTMW-l-m also beats RTMPose-x by 11.5 ROUGE-L (43.39 vs 31.94) and is the only engine to reproduce a
ceiling prediction verbatim (1/5 vs 0/5) — **but Tushar caught that this conflicts with the paper**,
which specifies RTMPose-x for the 133 keypoints, the same source as our index groups. Checked it two
ways. (1) Measured the simcc grid in the authors' own released pkls: bbox span / bin is scale-free, our
two engines recover 288/192 = 1.500 to within 2.6 % which validates the estimator, and the authors' poses
come out at 1.008x our 192-wide engine and 0.655x our 288-wide one — so the *archive* the checkpoint
consumes is 192-wide, agreeing with its `pose-rtmpose-192` folder name. The ratio conflates input width
with split_ratio and bbox padding though, so a 288 model padded ~1.9x would also fit: best reading, not
proof. Likely the paper describes their pipeline (CSL-News) while the distributed OpenASL poses were made
at 192. (2) Ruled out a bug in our RTMPose-x: its FP32 engine matched PyTorch to 8e-6 at P1 and its
per-group confidences track RTMW's (body 0.627 vs 0.715; face 100 % both; hands 98.4/99.1 vs 99.9/99.9).
The 0.3 % vs 16.9 % feet gap is both models failing on feet either side of the 0.3 gate.

**So the front-end is provisionally RTMW-l-m FP16, carried mainly on cost** (1.8x faster, 2x less energy
than RTMPose-x FP16 mixed), not on a 5-sentence accuracy claim. I had written "front-end decided" off
1/5 vs 0/5 ceiling matches, which cannot carry that weight in a report. The FP16 half is independent and
solid. Settling it needs ~25 more clips: the authors' poses for all 976 test clips are already in
`data/openasl_test_pose/`, so only the videos need fetching plus one board pass.

Two things to hold on to. (1) **BLEU on 5 sentences is not usable**: the authors' own poses score 16.23
BLEU-4 on this subset against 22.53–22.67 on the full 976-clip split with the same checkpoint and metric
code, so the subset alone moves BLEU-4 by 6 points. Only the identical-prediction counts are exact.
(2) **A 6.3 BLEU-4 / 2.8 ROUGE-L gap to the ceiling remains**, with only 1/5 predictions shared. The
leading untested cause is the crop recipe: P3 crops the bbox at native resolution, the authors
square/pad/resize to 224, and our confident keypoints span ~0.01–0.99 of the frame where theirs span
~0.06–0.89. Re-cropping the five clips the authors' way needs no new model and would lift every config
at once — highest-value next experiment on this track.

Board lessons from this run, both now fixed in the repo: `eval_openasl.py` died at import five times on
missing `portalocker` and `rouge` (now in `jetson/Dockerfile`), and then five more times with
`NvMapMemAllocInternalTagged: error 12` (ENOMEM) inside `from_pretrained`, because 3.1 GB sat in page
cache and nvmap only allocates from `MemFree`. `jetson/drop_file_cache.py` before each eval took MemFree
2.8 → 5.0 GB and fixed it. **The rule generalises beyond TensorRT builds: drop the page cache before
loading any large model on the board.** I had treated it as a build-time-only step, which cost a full
failed round of five evals.

**The crop experiment: hypothesis tested, not supported, and one of my results retracted.** Re-fetched the
same five clips with OpenASL's own recipe (`--crop-style openasl`: square the bbox, black-pad outside the
frame, resize to 224), validated the geometry against the authors' released keypoints first
(`data/verify_openasl_crop.py`, all five clips pass, y-maxima within 0.01–0.03 of the predicted content
edge), then re-ran all four engines and the four evals.

Their frames are **15–50 % black padding** — the OpenASL bboxes run far outside the frame, e.g.
`UoU3ZSuTef4` y from −141 to 1082 in a 720-tall frame. The shift landed as intended: our confident
keypoint spans moved from ~0.01–0.99 to ~0.14–0.88 against the authors' ~0.10–0.91.

**But it did not close the gap.** RTMW FP32 BLEU-4 rose 9.89 → 12.38 while ROUGE-L fell 43.39 → 38.87, and
ceiling-identical predictions stayed at 1/5. RTMPose-x got worse on both. So **crop convention is not the
cause of the ceiling gap.** The direction matters: their 224 crop *discards* resolution (our native crops
are 502–754 px wide), so our pose input is better-resolved and theirs still translates better — which rules
out pose sharpness too. The live hypothesis is now their extractor's keypoint placement conventions, learned
by the frozen ST-GCN, which no re-crop can fix.

**Retraction: "RTMW FP16 is translation-neutral, 5/5 identical" does not hold.** On the openasl crop the
same engine pair gives 3/5. I checked whether this was a real resolution effect by re-running the agreement
metric in **bins** rather than pixels — the only cross-crop-valid unit — and it is not: ≤2 bins is 99.39 %
at native and 99.76 % at 224, i.e. unchanged or slightly better. So the 5/5 was a favourable five-sentence
coin flip. The robust claim is the keypoint agreement; **sentence-level neutrality is unproven at n=5** and
must not be repeated in a report until it is re-tested on ~25 clips. FP16 stays adopted on the keypoint
evidence and on cost (1.8× faster, 2× less energy).

Two things gained. An ordering that now reproduces on two independent crops: RTMW agrees with its own FP32
better than RTMPose-x mixed does (≤2 bins 99.39 % vs 96.58 % native, 99.76 % vs 98.72 % openasl) — much
stronger than the single-crop version. And the sharpest illustration yet of why pixel thresholds are
unusable: the same engine pair reads 99.81 % within 5 px at native crop and 99.99 % at 224, which looks
like the small crop is safer, but 5 px is ~2 bins at native and ~7 bins at 224.

**Session 9 checkpoint — where to pick up.** Committed. Board: repo + both ONNX + 4 engines +
`data/clips/` + `data/calib_frames/` + `data/openasl_5clip_pose/` + `data/poses_*` + LM weights, 7.2 G,
30 G free; container `slt-work` up; nothing running. **P4 and 2.5 are both closed.** Next, in order:
(a) **fetch ~25 more clips.** This is now the blocking item for three separate questions, all of which are
n=5-limited: the extractor choice (which contradicts the paper), sentence-level FP16 neutrality (retracted
above), and the size of the ceiling gap. One fetch plus one board pass answers all three; the crop question
is already closed and needs no repeat. (b) **5.1 C6/M1**, first on-device end-to-end translation (pose → pkl → LM in
one process) — the Stage-0 baseline the week-4 gate (5.3) depends on; note it can proceed on RTMW FP16
regardless, since a later extractor switch changes one engine path and not the harness; (c) **the
author-style crop experiment**, cheap and would move every 2.5 row — and worth doing *before* (a), since a
crop mismatch may not penalise the two extractors equally; (d) **2.6 P5 INT8**, judged on hand agreement specifically (the group with no headroom) and on the
5/5-identical-predictions bar RTMW FP16 just set, calibrated from the 360 frames already on the board.
Then `jetson/run.sh clean-large`. Still outstanding from session 8: the C9 results protocol (3 runs
mean ± std, ≥3 clips/signers, one 30-min sustained run) has never been applied retroactively to the
existing latency rows, which are 3×299 frames of a single clip.

### Session 9 addendum: the two fps numbers, and a CPU/GPU power-budget coupling (2026-09-26)

Tushar asked why the RTMW FP16 row said 19.7 ms and 40 fps when 1000/19.7 = 51 and every other row
obeyed 1000/total. **It was an error in my briefing table**: that one cell held `fps_end_to_end` from
the harness while the rest held computed compute-fps. Both numbers were real, mixed as one.

Instrumented the two previously untimed stages in `task1_rtmpose/03_infer_frames.py` (`imread_ms`,
`collect_ms`, plus `fps_compute`, `wall_ms_per_frame`, `residual_ms`) and re-ran both RTMW engines with
power logging. The gap is JPEG decode (5.3 ms) + result serialisation (0.16 ms); residual 0.01 ms, so
the accounting closes. `total_ms` reproduced the published P2 rows to <1 %.

Unasked-for finding: `imread` and `preprocess` both inflate by **exactly ×1.281** between the FP16 and
FP32 runs, with `cpu0_MHz` 1045 → 897 (×1.165). A hungrier engine starves the CPU under the 15 W cap,
so CPU-stage timings are a property of the configuration, not of the code — which retires an old
RESULTS.md claim that preprocess "does not shrink with GPU quantization" (it shrinks 22 %).

Process fix: Tushar caught that I had not verified the shared board was idle before timing — I had
checked `docker ps` and `free -m` (memory safety) but not other users or GPU clients (timing validity).
Added **`jetson/run.sh whoelse`** (users, load, non-mine CPU hogs, containers, `/dev/nvidia*` holders,
MemFree, power mode) and used it for these runs. Retroactively, the earlier run was clean: 0 other
users, no foreign process >1 % CPU, and the latencies reproduce the published ones.

Docs: RESULTS.md new §2.2b + three corrected fps claims; BRIEFING §3.1 table now carries both fps
columns and CPU MHz; PROJECT-GUIDE rows 2.2 and A5.

### Session 9 addendum 2: n=30 pose→BLEU, and the extractor question answered (2026-09-26, late)

Fetched 25 more test clips (30 total, 30 distinct videos, 7299 frames) plus the authors' released poses
for all 30, ran 4 pose configs × 30 clips on the board (120 dumps), then 7 evals with paired bootstrap
CIs. The three questions that were n=5-limited are now answered, and two of my earlier conclusions were
wrong in the same direction — both drawn from five clips.

**Answered.** RTMW beats RTMPose-x by **7.13 BLEU-4 / 7.73 ROUGE-L**, sign established — the front-end
choice was cost-led and turns out to be accuracy-correct, so every "provisional" hedge is removed.
**FP16 is free**: −0.13 BLEU-4, CI [−1.48, +1.21], which properly replaces the retracted 5/5 claim.
**The ceiling gap is +1.61 BLEU-4, not established** (the n=5 figure of 6.33 was a subset artifact).

**Mechanism found for what remained.** Diffing our poses against theirs in the encoder's own units, the
*body* group (the only one in absolute coordinates) was worse by 2–4× than the root-relative hand and
face groups — the signature of a global scale error. Cause: they normalise over a square-padded-224 crop,
we over the bbox; predicted `square_side/crop_width` vs measured shoulder ratio correlate at **0.968**.
`common/renorm_to_openasl.py` fixes it by exact arithmetic (body disagreement 0.1328 → 0.0055, 24×) and
the best config, RTMW FP16 + frame fix at BLEU-4 18.52, is indistinguishable from the ceiling's 18.17.
Confidence gating was ruled out cleanly (2.8 % vs 2.8 % of consumed joints zeroed).

**Two mistakes worth keeping.** (1) I reported the crop hypothesis as "tested, not supported" and
retracted it on five clips where BLEU-4 had actually risen 9.89 → 12.38 — a real signal read as noise.
(2) I queued a second eval behind a `pgrep` guard that did not hold, ran two PyTorch processes at once,
and both died on the NVML allocator assert, losing an eval. Replaced with `jetson/p5_all_evals.sh`
(strictly sequential, one process, MemFree floor) and `jetson/after_evals.sh` (detached post-run CI
analysis, waits on a log marker rather than a pgrep).

Also caught in my own pipeline: the ceiling eval scored n=40 because `data/openasl_pose` holds 40
reference poses, so its raw 16.79 is not comparable to our n=30 rows. Restricted to the shared 30 it is
18.17. `bootstrap_ci.py` intersects by clip name so the CIs were unaffected — that name-recording change
earned itself back the same day.

### Session 10: M1 end-to-end, and INT8 dropped (2026-09-28)

**INT8 dropped** (Tushar's call, guide row 3.2 marked ❌). The premise did not hold: L8.2 had already
measured mT5 FP16 as numerically broken and BF16 as diverging, and W8A16 keeps activations in FP16 so it
inherits that; the decoder is also host-bound, so INT8 buys memory not latency. The frontier does not
need it — its accuracy axis is measured (beam 4 / 2 / greedy, established signs), so beam width is the
knob and only board energy is outstanding.

**M1 done** — `unisign/e2e_translate.py`, pose TRT + LM in one process, 255-frame clip: **9052.8 ms ±
335.6 per sentence** (pose 6640.7 + convert 25.7 + LM 2386.5), **50.37 J/sentence**, peak GPU 2.577 GB,
LM load 64.4 s. **1.06× slower than real time**, and the LM is why — pose alone is 0.78×.

Three things worth keeping:
1. **The NVML assert is an out-of-memory in disguise.** `NVML_SUCCESS == r INTERNAL ASSERT FAILED at
   CUDACachingAllocator.cpp:1017` is always preceded by `NvMapMemAllocInternalTagged ... error 12`.
   nvmap fails, PyTorch tries to report an OOM, NVML is partly unsupported on Jetson, and the assert
   replaces the real error. This explains the five evals lost on 09-26 as one cause, not two.
2. **fadvise is not enough on a shared board.** With another user's desktop session active, MemFree sat
   at 100 MB with 6.2 GB cached and `drop_file_cache.py` freed 14 MB. Added `--target-free-mb=N`, which
   allocates anonymous memory to force kernel reclaim then releases it (MemFree → 4618 MB). First
   version sized the allocation to the deficit, which evicts nothing — it must be sized to the target.
3. **Batch size changes output text.** M1 disagreed with the offline eval on one clip; the cause was
   batching (eval pads a batch of 8, M1 runs 1). `--batch-size 1` reproduces M1 exactly. 2/30 clips
   differ, BLEU-4 18.52 → 18.64. Every offline BLEU number carries this, so quote the batch size.

Also corrected a prediction of mine: I expected end-to-end to exceed pose-alone + LM-alone because of
the 15 W CPU/GPU coupling (§2.2b). It does not, because this implementation is sequential — all frames,
then the LM — so the stages never overlap. A streaming design would contend and is unmeasured.

FP32 end-to-end does not fit (OOM in KV-cache concat), so FP16 is a fitting requirement, not only an
energy choice.

### Session 10 part 2: P8 + L7 (2026-09-28)

`jetson/p8_l7_sweep.sh` (detached) + `jetson/p8_after.sh` (detached analysis). Three parts: pose energy
at 30/24/16/12/8 fps from real runs, accuracy at the same rates from subsampled keypoints, and the LM's
beam × encoder-length matrix.

**16 fps is the operating point**: pose energy 0.53× with Δ BLEU-4 −0.15, CI [−3.20, +2.89]. Only 8 fps
is an established loss (−7.18, CI [−12.55, −2.13]). 12 fps is *unknown*, not free. 24 fps scoring above
30 fps is noise.

**L7 answered, and it is a negative result worth stating**: the encoder is 39–50 ms against a decoder of
1062–1538 ms, so cutting T from 256 to 68 saves the LM only 12 %. Reduced frame rate is a pose saving,
not an LM saving. Beam width is the LM's real lever (greedy 8.88 J vs beam 4 11.04 J for 2.00 BLEU-4,
established at n=976) — and that is the whole frontier, which is why INT8 was droppable.

System effect: ~38 % energy saving at 16 fps for no measured accuracy cost.

Two failures in the first sweep pass, both mine: part C got the already-pruned mT5 dir together with the
pruned checkpoint (keep_ids index the original 250k vocab, so it needs the full mT5), and the 30 fps
eval hit the masked OOM at MemFree 4215 MB (floor raised to 5200). Both fixed and re-run.

Third silent board failure of the project: the board's CI step produced nothing because the board still
had the Sep-18 `bootstrap_ci.py` with no `--out` flag. Pushed the current one. **Board-generated
summaries need verifying, not trusting** — that is now three for three.

### Session 11: Atisri's report corrects three of my claims (2026-09-30)

Read `lm-track-report-2026-09-30.md`. Three corrections, all of them right.

1. **"16 fps is free" is retracted.** My n=30 curve (Δ −0.15, CI [−3.20, +2.89]) is superseded by her
   n=976/967: 30→16 fps costs **ROUGE-L −1.63 (test) / −1.30 (dev)**, replicating on both splits, while
   BLEU-4 cannot resolve it at n≈1000. She also showed that at n=300 the same model scored +1.10 BLEU-4
   *higher* at 16 fps — the opposite sign — so n=30 was far inside the unreliable regime. I wrote "not
   established" (correct) and then reported it as "no measured cost" (not correct). **Operating point is
   now 24 fps**, free on both metrics for 21 % less pose energy.
2. **A real bug she found independently:** OpenASL is not one frame rate. 76.5 % of our 931 clips are
   ~30 fps, 22.2 % ~24, 7 are 59.94 — and `subsample_pkl.py` / `--keep-fps` assumed 29.97, so a 24 fps
   clip labelled "16 fps" was subsampled to ~12.8. 6 of my 30 clips were affected. Now derived per clip
   from frames ÷ duration.
3. **My INT8 justification was wrong.** I argued W8A16 inherits mT5's fp16 failure; their implementation
   is W8A32 (fp32 activations, matmul and accumulation). The guide row said W8A16 until 09-28, but I
   should have checked the implementation, not the label. Her disproof is cleaner: the INT8 config scores
   22.79 BLEU-4, impossible under the fp16 failure mode. My second reason — host-bound decoder — was
   correct and sufficient. Right decision, wrong justification.
4. **"Beam width is the frontier" was LM-local.** Per BLEU-4 point: 30→24 fps costs 103.5 J, beam 4→2
   costs 1.0 J. Beam is the LM's lever and the *worst* system lever, since the LM is only ~25 % of system
   energy. Frame rate is 10–100× more efficient.

Sent back to her: (a) her test figures carry her own §8.1 vocabulary-leakage caveat, which reaches my
restatement of her pruning and decode rows — 53 keep-ids occur only in test, including `▁Bitcoin`, our
demo clip's key word; (b) her composed frontier's absolute system-J is anchored to my single clip, whose
crop is at the 83rd percentile, so a median clip is ~7 % cheaper per frame; (c) "memory is not binding at
2.577 GB peak" is true in steady state but not at load, where the pruned path peaks ~4.2 GB; (d) her 6 %
composition gap is probably not contention, since M1 found none — the stages are sequential.

Jetson still unreachable: tunnel up, routes installed, but no host on the lab subnet answers.
