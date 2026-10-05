#!/usr/bin/env python3
"""Delete frames only for clips that already have BOTH pose normalisations on disk.

Runs on the board. The old J12 cleanup was `find data/clips -name frames -exec rm -rf {} +`, which
deletes the frames of clips that failed extraction too -- on 2026-10-05 a batch wrote 3 pkls of 90
and then had all 90 clips' frames removed, so the only copy of the work queue was the pose-track Mac.
Deleting strictly what is already captured means a failed clip keeps its frames and the next run
retries it without another 7 MB push.

Clip directory names contain colons, so this is Python rather than a shell loop: no quoting to get
wrong.

    python3 jetson/j12_reclaim.py [--clips data/clips] [--pkl results/pkl_split_rtmw_fp16] [--dry-run]
"""
import argparse
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--clips", default="data/clips")
ap.add_argument("--pkl", default="results/pkl_split_rtmw_fp16")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()


def have(d):
    return {f[:-4] for f in os.listdir(d) if f.endswith(".pkl")} if os.path.isdir(d) else set()


done = have(a.pkl) & have(a.pkl + "_raw")
freed = kept = n = 0
for vid in sorted(os.listdir(a.clips)):
    fdir = os.path.join(a.clips, vid, "frames")
    if not os.path.isdir(fdir):
        continue
    size = sum(os.path.getsize(os.path.join(fdir, f)) for f in os.listdir(fdir))
    if vid in done:
        n += 1
        freed += size
        if not a.dry_run:
            shutil.rmtree(fdir)
    else:
        kept += 1
        print(f"[reclaim] KEEPING {vid}: no pkl pair yet ({size / 1e6:.1f} MB), so it can be retried")

print(f"[reclaim] {'would free' if a.dry_run else 'freed'} {freed / 1e9:.2f} GB from {n} captured "
      f"clip(s); kept frames for {kept} clip(s) still awaiting a pkl")
