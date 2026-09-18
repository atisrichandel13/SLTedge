#!/usr/bin/env bash
# Run once on the Jetson to confirm the environment the scripts were written for.
set -u
echo "== JetPack / L4T";  dpkg-query --showformat='${Version}\n' --show nvidia-jetpack 2>/dev/null; cat /etc/nv_tegra_release 2>/dev/null | head -1
echo "== TensorRT";       dpkg -l | grep -E "^ii\s+(tensorrt|libnvinfer10)\s" | awk '{print $2, $3}'
echo "== CUDA";           /usr/local/cuda/bin/nvcc --version | tail -1
echo "== Python / torch"; python3 - <<'PY'
import sys; print(sys.version.split()[0])
import torch; print("torch", torch.__version__, "cuda", torch.version.cuda, "ok", torch.cuda.is_available())
import tensorrt as trt; print("tensorrt", trt.__version__)
for m in ("transformers", "onnx", "onnxruntime", "optimum", "mmpose", "cv2", "jtop"):
    try: print(m, __import__(m).__version__)
    except Exception as e: print(m, "MISSING")
PY
echo "== nvpmodel (expect: mode 3 = 7W, mode 0 = 15W)"; sudo -n nvpmodel -q 2>/dev/null || nvpmodel -q 2>/dev/null || echo "run: sudo nvpmodel -q"
echo "== tegrastats rails (expect VDD_IN, VDD_CPU_GPU_CV, VDD_SOC)"; timeout 2 tegrastats --interval 500 | head -1 | grep -o "VDD_[A-Z_]* [0-9]*mW/[0-9]*mW" || echo "no rails: try sudo tegrastats"
