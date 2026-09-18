#!/usr/bin/env python3
"""Power logging for Jetson via tegrastats (default) or jtop.

Usage as a context manager:

    from common.power_logger import PowerLogger
    with PowerLogger(interval_ms=100) as pl:
        pl.mark("idle_start"); time.sleep(3); pl.mark("idle_end")
        pl.mark("run_start"); run_inference(); pl.mark("run_end")
    summary = pl.summarize(n_frames=N, window=("run_start", "run_end"),
                           baseline=("idle_start", "idle_end"))
    print(json.dumps(summary, indent=2))

Rails (confirmed on the target Orin Nano, JetPack 6.2.1):
    VDD_IN          -> total module input power (this is "watts" for the report)
    VDD_CPU_GPU_CV  -> CPU + GPU + CV accelerators
    VDD_SOC         -> SoC (memory controller, misc)

tegrastats prints one line per interval like:
    ... VDD_IN 3335mW/3335mW VDD_CPU_GPU_CV 521mW/521mW VDD_SOC 1125mW/1125mW
where the first number is the instantaneous reading and the second is the
running average since tegrastats started. We use the instantaneous value and
integrate it ourselves.

NOTE: on some images tegrastats needs root to read the INA3221 rails. If the
parsed samples contain no VDD_* keys, re-run with `sudo=True` (uses `sudo -n`,
so configure passwordless sudo for tegrastats or run the whole script as root).
"""
import json
import re
import shutil
import subprocess
import threading
import time
from collections import defaultdict

RAIL_RE = re.compile(r"(VDD_[A-Z0-9_]+) (\d+)mW/(\d+)mW")
GPU_RE = re.compile(r"GR3D_FREQ (\d+)%")
RAM_RE = re.compile(r"RAM (\d+)/(\d+)MB")
TEMP_RE = re.compile(r"(gpu|cpu|soc\d?|tj)@([\d.]+)C")


def parse_tegrastats_line(line):
    """Return {rail: mW, ...} plus a few aux fields from one tegrastats line."""
    sample = {}
    for name, inst, _avg in RAIL_RE.findall(line):
        sample[name] = float(inst)
    m = GPU_RE.search(line)
    if m:
        sample["GR3D_FREQ_pct"] = float(m.group(1))
    m = RAM_RE.search(line)
    if m:
        sample["RAM_used_MB"] = float(m.group(1))
    for name, val in TEMP_RE.findall(line):
        sample[f"temp_{name}_C"] = float(val)
    return sample


class _TegrastatsBackend:
    def __init__(self, interval_ms, sudo=False):
        exe = shutil.which("tegrastats") or "/usr/bin/tegrastats"
        self.cmd = (["sudo", "-n"] if sudo else []) + [exe, "--interval", str(interval_ms)]
        self.proc = None

    def start(self, on_sample):
        self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     text=True, bufsize=1)
        self._on_sample = on_sample
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        for line in self.proc.stdout:
            s = parse_tegrastats_line(line)
            if s:
                self._on_sample(time.perf_counter(), s)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        # tegrastats can leave a daemon behind when started with sudo.
        subprocess.run(self.cmd[:-2] + ["--stop"], capture_output=True)


class _SysfsBackend:
    """Read the INA3221 rails straight from hwmon sysfs. Works inside a Docker container
    (no tegrastats binary needed) as long as /sys is visible. Power per rail is
    in<i>_input (mV) * curr<i>_input (mA) / 1000 -> mW, same quantity tegrastats reports.
    Orin Nano: /sys/bus/i2c/drivers/ina3221/1-0040/hwmon/hwmon*/ with labels
    VDD_IN, VDD_CPU_GPU_CV, VDD_SOC."""

    GLOB = "/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*"

    @classmethod
    def available(cls):
        return bool(cls.discover())

    @classmethod
    def discover(cls):
        import glob, os
        rails = {}
        for d in glob.glob(cls.GLOB):
            for lab in glob.glob(os.path.join(d, "in*_label")):
                i = os.path.basename(lab)[2:-6]
                name = open(lab).read().strip()
                v, c = os.path.join(d, f"in{i}_input"), os.path.join(d, f"curr{i}_input")
                if name.startswith("VDD_") and os.path.exists(v) and os.path.exists(c):
                    rails[name] = (v, c)
        return rails

    def __init__(self, interval_ms):
        import glob
        self.interval = interval_ms / 1000.0
        self.rails = self.discover()
        if not self.rails:
            raise RuntimeError(f"no INA3221 rails under {self.GLOB}")
        # optional extras: thermal zones and GPU load, best effort
        self.temps = {}
        for tz in glob.glob("/sys/class/thermal/thermal_zone*"):
            try:
                t = open(tz + "/type").read().strip().lower()
                for key in ("gpu", "cpu", "soc", "tj"):
                    if t.startswith(key):
                        self.temps[f"temp_{t.split('-')[0].split('_')[0]}_C"] = tz + "/temp"
            except OSError:
                pass
        self.gpu_load = None
        for cand in ("/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load"):
            try:
                open(cand).read(); self.gpu_load = cand; break
            except OSError:
                pass
        self._stop = threading.Event()

    @staticmethod
    def _read(path):
        with open(path) as f:
            return float(f.read().strip())

    def sample(self):
        s = {}
        for name, (v, c) in self.rails.items():
            s[name] = self._read(v) * self._read(c) / 1000.0  # mV*mA -> mW
        for k, path in self.temps.items():
            try: s[k] = self._read(path) / 1000.0
            except OSError: pass
        if self.gpu_load:
            try: s["GR3D_FREQ_pct"] = self._read(self.gpu_load) / 10.0  # load is in 0..1000
            except OSError: pass
        return s

    def start(self, on_sample):
        self._on_sample = on_sample
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        nxt = time.perf_counter()
        while not self._stop.is_set():
            self._on_sample(time.perf_counter(), self.sample())
            nxt += self.interval
            dt = nxt - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            else:
                nxt = time.perf_counter()

    def stop(self):
        self._stop.set()
        self._t.join(timeout=2)


class _JtopBackend:
    """Requires `sudo pip3 install jetson-stats` and a reboot (jtop service)."""

    def __init__(self, interval_ms):
        from jtop import jtop  # noqa: F401
        self.interval = interval_ms / 1000.0
        self._stop = threading.Event()

    def start(self, on_sample):
        from jtop import jtop
        self._jt = jtop(interval=self.interval)
        self._jt.start()
        self._on_sample = on_sample
        self._t = threading.Thread(target=self._reader, daemon=True)
        self._t.start()

    def _reader(self):
        while not self._stop.is_set() and self._jt.ok():
            p = self._jt.power  # jtop>=4: {'rail': {name: {'power': mW, ...}}, 'tot': {...}}
            s = {}
            for name, d in p.get("rail", {}).items():
                s[name] = float(d.get("power", 0.0))
            if "tot" in p and "VDD_IN" not in s:
                s["VDD_IN"] = float(p["tot"].get("power", 0.0))
            s["GR3D_FREQ_pct"] = float(self._jt.stats.get("GPU", 0.0))
            self._on_sample(time.perf_counter(), s)
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        self._jt.close()


class PowerLogger:
    def __init__(self, interval_ms=100, backend="tegrastats", sudo=False, total_rail="VDD_IN"):
        self.interval_ms = interval_ms
        self.total_rail = total_rail
        self.samples = []  # list of (t, dict)
        self.marks = {}
        self._lock = threading.Lock()
        if backend == "auto":
            backend = "sysfs" if _SysfsBackend.available() else "tegrastats"
        self.backend_name = backend
        if backend == "jtop":
            self.backend = _JtopBackend(interval_ms)
        elif backend == "sysfs":
            self.backend = _SysfsBackend(interval_ms)
        else:
            self.backend = _TegrastatsBackend(interval_ms, sudo=sudo)
        print(f"[power] backend={backend} interval={interval_ms}ms")

    def _on_sample(self, t, s):
        with self._lock:
            self.samples.append((t, s))

    def __enter__(self):
        self.backend.start(self._on_sample)
        # Wait for the first sample so the run window is fully covered.
        t0 = time.perf_counter()
        while not self.samples and time.perf_counter() - t0 < 3.0:
            time.sleep(0.05)
        if not self.samples:
            raise RuntimeError(f"no {self.backend_name} samples within 3 s; is tegrastats on PATH "
                               "(try sudo=True) or the jtop service running?")
        return self

    def __exit__(self, *exc):
        time.sleep(self.interval_ms / 1000.0 * 2)  # catch the tail of the run
        self.backend.stop()

    def mark(self, name):
        self.marks[name] = time.perf_counter()

    # ------------------------------------------------------------------ analysis
    def _window(self, window):
        with self._lock:
            samples = list(self.samples)
        if window is None:
            return samples
        a = self.marks[window[0]] if isinstance(window[0], str) else window[0]
        b = self.marks[window[1]] if isinstance(window[1], str) else window[1]
        return [(t, s) for (t, s) in samples if a <= t <= b]

    @staticmethod
    def _integrate_mJ(samples, key):
        """Trapezoidal integral of mW over seconds -> mJ."""
        if len(samples) < 2:
            return 0.0
        e = 0.0
        for (t0, s0), (t1, s1) in zip(samples[:-1], samples[1:]):
            if key in s0 and key in s1:
                e += 0.5 * (s0[key] + s1[key]) * (t1 - t0)
        return e

    def summarize(self, n_frames=None, window=None, baseline=None):
        samples = self._window(window)
        if not samples:
            return {"error": "no samples in window"}
        keys = sorted({k for _, s in samples for k in s})
        duration_s = samples[-1][0] - samples[0][0]
        out = {"n_samples": len(samples), "duration_s": round(duration_s, 3), "rails_mW_avg": {},
               "rails_mW_max": {}}
        for k in keys:
            vals = [s[k] for _, s in samples if k in s]
            if k.startswith("VDD_"):
                out["rails_mW_avg"][k] = round(sum(vals) / len(vals), 1)
                out["rails_mW_max"][k] = round(max(vals), 1)
            else:
                out.setdefault("aux_avg", {})[k] = round(sum(vals) / len(vals), 2)
        total = self.total_rail
        e_mJ = self._integrate_mJ(samples, total)
        out["total_rail"] = total
        out["avg_watts"] = round(out["rails_mW_avg"].get(total, 0.0) / 1000.0, 3)
        out["energy_mJ"] = round(e_mJ, 1)
        if baseline is not None:
            base = self._window(baseline)
            if base:
                bvals = [s[total] for _, s in base if total in s]
                base_mW = sum(bvals) / len(bvals)
                out["baseline_watts"] = round(base_mW / 1000.0, 3)
                out["dynamic_energy_mJ"] = round(e_mJ - base_mW * duration_s, 1)
        if n_frames:
            out["n_frames"] = n_frames
            out["mJ_per_frame"] = round(e_mJ / n_frames, 2)
            if "dynamic_energy_mJ" in out:
                out["dynamic_mJ_per_frame"] = round(out["dynamic_energy_mJ"] / n_frames, 2)
        return out

    def dump_csv(self, path):
        keys = sorted({k for _, s in self.samples for k in s})
        with open(path, "w") as f:
            f.write("t," + ",".join(keys) + "\n")
            for t, s in self.samples:
                f.write(f"{t:.4f}," + ",".join(str(s.get(k, "")) for k in keys) + "\n")


def add_power_args(parser):
    parser.add_argument("--power-backend", choices=["auto", "sysfs", "tegrastats", "jtop"], default="auto",
                        help="auto = sysfs INA3221 if visible (works in Docker), else tegrastats")
    parser.add_argument("--power-interval-ms", type=int, default=100)
    parser.add_argument("--power-sudo", action="store_true", help="run tegrastats via sudo -n")
    parser.add_argument("--idle-seconds", type=float, default=3.0,
                        help="idle baseline measured before the run (0 to skip)")
    parser.add_argument("--power-csv", default=None, help="dump raw samples to CSV")
    parser.add_argument("--power-json", default=None, help="write summary JSON")


def run_with_power(args, fn, n_frames_getter):
    """Run fn() under power logging; returns (fn_result, summary)."""
    with PowerLogger(args.power_interval_ms, args.power_backend, args.power_sudo) as pl:
        if args.idle_seconds > 0:
            pl.mark("idle_start")
            time.sleep(args.idle_seconds)
            pl.mark("idle_end")
        pl.mark("run_start")
        result = fn()
        pl.mark("run_end")
    n = n_frames_getter(result)
    summary = pl.summarize(n_frames=n, window=("run_start", "run_end"),
                           baseline=("idle_start", "idle_end") if args.idle_seconds > 0 else None)
    if args.power_csv:
        pl.dump_csv(args.power_csv)
    if args.power_json:
        with open(args.power_json, "w") as f:
            json.dump(summary, f, indent=2)
    return result, summary


if __name__ == "__main__":
    # Self-test of the parser with the exact rail string from the device.
    line = ("09-15-2026 10:00:00 RAM 2345/7620MB (lfb 10x4MB) SWAP 0/3810MB (cached 0MB) CPU [3%@729,"
            "2%@729,0%@729,0%@729,0%@729,0%@729] GR3D_FREQ 0% cpu@45.5C soc2@44C soc0@45C gpu@44.5C "
            "tj@45.5C soc1@44C VDD_IN 3335mW/3335mW VDD_CPU_GPU_CV 521mW/521mW VDD_SOC 1125mW/1125mW")
    print(parse_tegrastats_line(line))
    assert parse_tegrastats_line(line)["VDD_IN"] == 3335.0
    print("parser OK")
