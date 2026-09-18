# cv-jetson: two-stage pose + mT5 pipeline on Jetson Orin Nano 8GB

Target: JetPack 6.2.1, TensorRT 10.3, PyTorch 2.8.0a0, CUDA 12.6, Python 3.10.
Power modes: `sudo nvpmodel -m 3` (7 W) / `sudo nvpmodel -m 0` (15 W).
Rails: `VDD_IN` (module total, used for watts/mJ), `VDD_CPU_GPU_CV`, `VDD_SOC`.

```
common/            envinfo.py (stack versions), power_logger.py (tegrastats/jtop -> W, mJ/frame),
                   trt_runner.py (TRT 10 v3 API runner + ONNX->engine builder, torch tensors as memory)
task1_rtmpose/     01 export ONNX -> 02 build engine -> 03 latency loop -> 04 + power
                   05 PyTorch reference (.npz, host) -> 06 TRT vs reference (device)
task2_mt5_baseline/01 mT5 plain PyTorch generate + latency + power
task3_mt5_onnx/    01 optimum export | 02 manual export with KV-cache I/O (+ --verify) | 03 TRT engines
probe_device.sh    confirm versions / nvpmodel / rails on the device
```

## Setup
```
./probe_device.sh
pip3 install -r requirements-jetson.txt        # see comments inside for optional parts
mkdir -p models results data/test_frames        # drop ~200 jpg/png frames in data/test_frames
```

## Task 1: RTMPose-x
```
# on a host with mmpose (or on the Jetson if mmcv installs):
mim download mmpose --config rtmpose-x_8xb32-270e_coco-wholebody-384x288 --dest weights/
python task1_rtmpose/01_export_onnx.py --config weights/*.py --checkpoint weights/*.pth --out models/rtmpose-x.onnx
python task1_rtmpose/05_make_reference.py --config weights/*.py --checkpoint weights/*.pth \
    --frames data/test_frames --limit 20 --onnx models/rtmpose-x.onnx --out results/rtmpose_reference.npz
# copy models/rtmpose-x.onnx, models/preproc.json, results/rtmpose_reference.npz to the Jetson, then:
python task1_rtmpose/02_build_engine.py --onnx models/rtmpose-x.onnx
python task1_rtmpose/06_compare_trt.py --engine models/rtmpose-x_fp32.engine --reference results/rtmpose_reference.npz
python task1_rtmpose/03_infer_frames.py --engine models/rtmpose-x_fp32.engine --frames data/test_frames --out results/rtmpose_trt.json
sudo nvpmodel -m 3
python task1_rtmpose/04_infer_power.py --engine models/rtmpose-x_fp32.engine --frames data/test_frames \
    --repeat 3 --power-json results/rtmpose_power_7W.json --power-csv results/rtmpose_power_7W.csv
```
Pass criteria in 06: simcc max|diff| <= 1e-3 and keypoints within 1 px (FP32).

## Task 2: mT5 baseline
```
python task2_mt5_baseline/01_mt5_torch_baseline.py --model google/mt5-base --seq-len 64 \
    [--unisign-ckpt weights/unisign.pth --unisign-prefix mt5_model.] --power-json results/mt5_power_7W.json
```
`--seq-len` / `--batch` are placeholders for the confirmed Uni-Sign encoder input shape (B, T, 768).

## Task 3: mT5 ONNX + TensorRT scaffold
```
python task3_mt5_onnx/02_export_manual.py --model google/mt5-base --out models/mt5_onnx_manual --encoder-input embeds --verify
python task3_mt5_onnx/03_build_engines.py --onnx-dir models/mt5_onnx_manual --which encoder      # one per process on 8 GB
python task3_mt5_onnx/03_build_engines.py --onnx-dir models/mt5_onnx_manual --which decoder_init
python task3_mt5_onnx/03_build_engines.py --onnx-dir models/mt5_onnx_manual --which decoder_step --enc-len 1,64,128 --dec-len 1,1,128
# route A (optimum), for comparison / decoders only:
python task3_mt5_onnx/01_export_optimum.py --model google/mt5-base --out models/mt5_onnx_optimum
```
Read the `VERIFY-BY-HAND LIST` at the top of `02_export_manual.py` before trusting the graphs.
The export path was exercised on a tiny random mT5 (transformers 4.57, torch 2.2, CPU): ONNX greedy
decode with explicit KV cache matched PyTorch logits to ~1e-5 over 12 steps at encoder lengths
different from the export sample. Not yet run on mt5-base or on TensorRT.

## What was NOT run on the Jetson
Everything TensorRT- or mmpose-dependent was written against the TRT 10.3 / MMPose 1.x APIs but only
syntax-checked here. `common/power_logger.py` parses the exact tegrastats line format from the device
(self-test in `__main__`); `task1_rtmpose/rtmpose_utils.py` has a self-test for the affine/SimCC math.
