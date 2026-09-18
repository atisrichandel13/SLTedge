#!/usr/bin/env python3
"""Print the on-device software stack so every log is self-describing.

Expected on the target device (from the project brief):
    JetPack 6.2.1+b38 | TensorRT 10.3.0.30 | PyTorch 2.8.0a0 | CUDA 12.6 | Python 3.10.12
    nvpmodel: mode 3 = 7W, mode 0 = 15W
    tegrastats rails: VDD_IN, VDD_CPU_GPU_CV, VDD_SOC
"""
import platform
import re
import subprocess
import sys


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return ""


def jetpack_version():
    out = _run(["dpkg-query", "--showformat=${Version}", "--show", "nvidia-jetpack"])
    return out or "unknown (nvidia-jetpack package not found)"


def l4t_version():
    try:
        with open("/etc/nv_tegra_release") as f:
            m = re.search(r"R(\d+).*REVISION: ([\d.]+)", f.read())
            return f"R{m.group(1)}.{m.group(2)}" if m else "unknown"
    except FileNotFoundError:
        return "n/a (not a Jetson)"


def nvpmodel_mode():
    # `nvpmodel -q` may need sudo on some images; we only read, never set.
    out = _run(["nvpmodel", "-q"]) or _run(["sudo", "-n", "nvpmodel", "-q"])
    return " ".join(out.split()) if out else "unknown (try: sudo nvpmodel -q)"


def collect():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jetpack": jetpack_version(),
        "l4t": l4t_version(),
        "nvpmodel": nvpmodel_mode(),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"import failed: {e}"
    try:
        import tensorrt as trt
        info["tensorrt"] = trt.__version__
    except Exception as e:  # noqa: BLE001
        info["tensorrt"] = f"import failed: {e}"
    nvcc = _run(["nvcc", "--version"]) or _run(["/usr/local/cuda/bin/nvcc", "--version"])
    m = re.search(r"release ([\d.]+), V([\d.]+)", nvcc)
    info["nvcc"] = f"V{m.group(2)}" if m else "unknown"
    return info


def print_env(prefix="[env] "):
    for k, v in collect().items():
        print(f"{prefix}{k}: {v}")


if __name__ == "__main__":
    print_env()
