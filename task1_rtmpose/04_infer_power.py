#!/usr/bin/env python3
"""RTMPose TensorRT inference loop wrapped with tegrastats/jtop power logging.

    sudo nvpmodel -m 3    # 7W   (mode 0 = 15W)   -- set BEFORE running, needs root
    python task1_rtmpose/04_infer_power.py --engine models/rtmpose-x_fp32.engine \
        --frames data/test_frames --power-json results/rtmpose_power_7W.json \
        --power-csv results/rtmpose_power_7W.csv [--power-sudo]

Reports average watts on VDD_IN (module input), energy (mJ) integrated over the
timed window, mJ/frame, and dynamic mJ/frame (idle baseline subtracted).
For stable numbers use >= 200 frames or --repeat so the window is >= 20 s.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from common.envinfo import collect, print_env  # noqa: E402
from common.power_logger import add_power_args, run_with_power  # noqa: E402
from common.trt_runner import TrtRunner  # noqa: E402
import importlib  # noqa: E402

infer = importlib.import_module("03_infer_frames")


def main():
    ap = argparse.ArgumentParser()
    infer.add_args(ap)
    add_power_args(ap)
    ap.add_argument("--repeat", type=int, default=1, help="loop over the folder N times")
    args = ap.parse_args()
    print_env()

    runner = TrtRunner(args.engine)  # engine load + CUDA init happen outside the power window
    runner.describe()

    def job():
        n_total, last = 0, None
        for _ in range(args.repeat):
            last = infer.run_folder(args, runner=runner, quiet=True)
            n_total += last["n_frames"]
        last["n_frames_total"] = n_total
        return last

    summary, power = run_with_power(args, job, lambda s: s["n_frames_total"])
    report = {"env": collect(), "latency": summary, "power": power}
    print(json.dumps(report, indent=2))
    if args.power_json:
        with open(args.power_json, "w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
