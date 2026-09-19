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


# Orin Nano 8GB with JetPack 6.2 "super" conf (/etc/nvpmodel/nvpmodel_p3767_0003_super.conf):
# 0 = 15W (GPU <= 612 MHz), 1 = 25W (GPU <= 918 MHz, the conf's DEFAULT), 2 = MAXN_SUPER (uncapped),
# 3 = 7W (GPU <= 408 MHz). The course boards are pinned to mode 0.
_NVP_MODES = {"0": "15W", "1": "25W", "2": "MAXN_SUPER", "3": "7W"}


def gpu_clock_caps():
    """DVFS caps from devfreq (readable in the container). Latency scales ~1/gpu_max_MHz."""
    out = {}
    for k, path in (("gpu_max_MHz", "/sys/class/devfreq/17000000.gpu/max_freq"),
                    ("gpu_min_MHz", "/sys/class/devfreq/17000000.gpu/min_freq"),
                    ("cpu_max_MHz", "/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq")):
        try:
            with open(path) as f:
                v = float(f.read().strip())
            out[k] = v / 1e6 if k.startswith("gpu") else v / 1e3
        except (OSError, ValueError):
            pass
    return out


def nvpmodel_mode():
    # `nvpmodel -q` may need sudo on some images; we only read, never set.
    out = _run(["nvpmodel", "-q"]) or _run(["sudo", "-n", "nvpmodel", "-q"])
    if out:
        return " ".join(out.split())
    # Inside a container (no nvpmodel binary) the host's status file can be bind-mounted read-only:
    # `pmode:0000` -> mode 0. jetson/run.sh mounts it.
    try:
        with open("/var/lib/nvpmodel/status") as f:
            m = re.search(r"pmode:(\d+)", f.read())
        if m:
            mode = str(int(m.group(1)))
            return f"mode {mode} ({_NVP_MODES.get(mode, '?')}) from /var/lib/nvpmodel/status"
    except OSError:
        pass
    return "unknown (try: sudo nvpmodel -q)"


def collect():
    info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "jetpack": jetpack_version(),
        "l4t": l4t_version(),
        "nvpmodel": nvpmodel_mode(),
    }
    info.update(gpu_clock_caps())
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
