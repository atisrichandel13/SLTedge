#!/usr/bin/env python3
"""Tabulate the P8 / L7 sweep: pose energy vs rate, accuracy vs rate, LM energy matrix.

    python3 tools/p8_tables.py            # prints; the wrapper tees it to a file

Reads only JSON the sweep already wrote, so it is safe to run repeatedly and reports what is present
rather than failing on what is missing.
"""
import glob
import json
import os
import re

RATES = [30, 24, 16, 12, 8]


def pose_table():
    print("=" * 78)
    print(" A. Pose stage vs capture rate (real runs, 3 repeats, 15 W)")
    print("=" * 78)
    rows = []
    for f in sorted(glob.glob("results/p8_pose_fps*.json")):
        d = json.load(open(f))
        lat, pw, aux = d["latency"], d["power"], d["power"]["aux_avg"]
        fps = float(re.search(r"fps(\d+)", f).group(1))
        n = lat["n_frames"]
        sec_video = n / fps
        rows.append((fps, n, lat["total_ms"]["mean"], pw["mJ_per_frame"],
                     pw["dynamic_mJ_per_frame"], pw["avg_watts"], aux.get("gpu_MHz", -1),
                     aux.get("cpu0_MHz", -1), pw["mJ_per_frame"] * n / 1000 / sec_video))
    if not rows:
        print("  (no pose runs found)\n"); return
    rows.sort(key=lambda r: -r[0])
    base = rows[0][-1]
    print(f"{'fps':>4s} {'frames':>6s} {'ms/frame':>8s} {'mJ/frame':>8s} {'dyn mJ':>7s} {'W':>5s} "
          f"{'gpuMHz':>7s} {'cpuMHz':>7s} {'J/s video':>10s} {'vs 30fps':>9s}")
    for fps, n, ms, mj, dmj, w, g, c, jsv in rows:
        print(f"{fps:4.0f} {n:6d} {ms:8.2f} {mj:8.1f} {dmj:7.1f} {w:5.2f} {g:7.0f} {c:7.0f} "
              f"{jsv:10.2f} {jsv / base:8.2f}x")
    print("\n  mJ/frame is flat by construction -- each frame costs the same. What scales is how many")
    print("  frames exist per second of video, so J per second of video is the comparable column.")
    print("  Absolute mJ/frame is NOT comparable to other runs: each of the 3 repeats pays 20 warm-up")
    print("  inferences, so clips with fewer frames amortise that overhead less.\n")


def accuracy_table():
    print("=" * 78)
    print(" B. Translation accuracy vs capture rate (30 clips, beam 4, batch 1, released ckpt)")
    print("=" * 78)
    rows = []
    for fps in RATES:
        f = f"results/eval_30clip_fps{fps}.json"
        if not os.path.exists(f):
            continue
        d = json.load(open(f))
        rows.append((fps, d["n"], d["bleu"]["bleu4"], d["rouge_l"]))
    if not rows:
        print("  (no evals found)\n"); return
    base4 = rows[0][2] if rows[0][0] == 30 else None
    print(f"{'fps':>4s} {'n':>4s} {'BLEU-4':>7s} {'ROUGE-L':>8s} {'dBLEU-4':>8s}")
    for fps, n, b, r in rows:
        d = f"{b - base4:+8.2f}" if base4 is not None else "       -"
        print(f"{fps:4d} {n:4d} {b:7.2f} {r:8.2f} {d}")
    print("\n  Poses subsampled from the full-rate keypoints, which is exact: pose extraction is")
    print("  per-frame independent, so a kept frame gets the keypoints it would have had at 30 fps.")
    print("  These are UN-ADAPTED: the frozen encoder never saw a reduced rate. C8 adaptation is the")
    print("  separate question, and this is the baseline any adaptation gain is measured against.\n")


def lm_table():
    print("=" * 78)
    print(" C. LM latency + energy: beam width x encoder length (pruned checkpoint)")
    print("=" * 78)
    f = "results/lm_sweep_pruned.json"
    if not os.path.exists(f):
        print("  (lm_sweep_pruned.json not found)\n"); return
    rows = json.load(open(f))["rows"]
    print(f"{'beams':>5s} {'len':>4s} {'used':>5s} {'ms':>7s} {'enc':>6s} {'dec':>7s} "
          f"{'J/sent':>7s} {'dynJ':>6s} {'W':>5s} {'gpuMHz':>7s}")
    for r in rows:
        print(f"{r['beams']:5d} {r['requested_length']:4d} {r['frames_used'][0]:5d} "
              f"{r['total_ms']['mean']:7.0f} {r['encoder_ms']:6.0f} {r['decoder_ms']:7.0f} "
              f"{r['J_per_sentence']:7.2f} {r['dyn_J_per_sentence']:6.2f} {r['avg_W']:5.2f} "
              f"{r['gpu_MHz']:7.0f}")
    print("\n  Pair with the accuracy axis from RESULTS.md L6.1 (976 clips, established signs):")
    print("    beam 4 -> 22.87 | beam 2 -> 22.06 (-0.81, CI [-1.29, -0.39]) | greedy -> 20.88")
    print("    (-2.00, CI [-2.63, -1.41]).  Those two together are the accuracy-energy frontier.\n")


if __name__ == "__main__":
    pose_table()
    accuracy_table()
    lm_table()
