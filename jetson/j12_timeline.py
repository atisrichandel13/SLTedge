#!/usr/bin/env python3
"""Reconstruct a board extraction run's timeline from pkl mtimes.

WHY THIS EXISTS. J12 extracted 528 clips and its log recorded five lines of it: the driver's stdout
was a pipeline whose reader exited two minutes in, and since `set -e` is deliberately off in
j12_rest_of_split.sh (a batch shortfall must not kill the run) every later echo failed with EPIPE in
silence. The pkls themselves still carry when they were written, so the run is recoverable from the
filesystem. RESULTS.md 2.5k is built from this script's output.

The method is VALIDATED, not assumed: on the n=100 rung it reproduces 2.5g's separately recorded
figures -- 24,365 frames exactly, 36.3 vs 36.0 frames/s, 11.2 vs 11.3 min -- and the rung frame
totals sum to the same 203,562 that n_frames sums to across the 931 meta.json files.

Runs ON THE BOARD, where the pkls and the clip metas both live:

    python3 jetson/j12_timeline.py
    python3 jetson/j12_timeline.py --pkl results/pkl_dev_rtmw_fp16 --clips data/clips_dev

Rungs are split on gaps longer than --gap-min, so a run separated from the previous one by a break
shows up as its own row. That is a heuristic about human scheduling, not a recorded boundary: two
extractions less than --gap-min apart will merge into one row, so check the clip counts against what
you expected to run rather than trusting the split blindly.
"""
import argparse
import datetime as dt
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", default="results/pkl_split_rtmw_fp16")
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--gap-min", type=float, default=30.0,
                    help="a gap longer than this starts a new rung")
    a = ap.parse_args()

    items = sorted((os.path.getmtime(p), os.path.basename(p)[:-4])
                   for p in glob.glob(os.path.join(a.pkl, "*.pkl")))
    if not items:
        raise SystemExit(f"[timeline] no pkls under {a.pkl!r}")

    rungs = [[items[0]]]
    for it in items[1:]:
        if it[0] - rungs[-1][-1][0] > a.gap_min * 60:
            rungs.append([it])
        else:
            rungs[-1].append(it)

    def stamp(t):
        return dt.datetime.utcfromtimestamp(t).strftime("%m-%dT%H:%M:%SZ")

    tot_clips = tot_frames = 0
    print(f"{'rung':>4} {'clips':>6} {'frames':>9} {'wall':>9} {'frames/s':>9}  window (UTC)")
    for i, r in enumerate(rungs, 1):
        frames = nometa = 0
        for _, vid in r:
            try:
                with open(os.path.join(a.clips, vid, "meta.json")) as fh:
                    frames += json.load(fh)["n_frames"]
            except Exception:
                nometa += 1
        span = r[-1][0] - r[0][0]
        rate = f"{frames / span:9.1f}" if span > 0 else "       NA"
        print(f"{i:>4} {len(r):>6} {frames:>9,} {span / 60:>8.1f}m {rate}  "
              f"{stamp(r[0][0])} -> {stamp(r[-1][0])}"
              + (f"   [{nometa} clip(s) with no readable meta]" if nometa else ""))
        tot_clips += len(r)
        tot_frames += frames

    print(f"{'all':>4} {tot_clips:>6} {tot_frames:>9,}")
    print("\nframes/s is an AGGREGATE over the window, so a batched run includes its staging rsyncs\n"
          "and reclaim passes and reads below the GPU-only rate. Cross-check the total against\n"
          "sum(n_frames) over the clip metas before quoting any of it.")


if __name__ == "__main__":
    main()
