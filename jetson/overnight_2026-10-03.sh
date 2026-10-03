#!/usr/bin/env bash
# Overnight chain: J8 (C9 sustained) then the n=400 accuracy pass. Launch detached:
#
#     setsid nohup ./jetson/overnight_2026-10-03.sh > results/logs/overnight.log 2>&1 < /dev/null &
#
# WHY CHAINED AND NOT PARALLEL. The two jobs share no results -- J8 is latency/power/thermal, n=400
# is accuracy -- but they cannot run together. Two PyTorch/TensorRT processes on this board kill each
# other with the CUDACachingAllocator assert, which has happened twice (p5_all_evals.sh header). A
# sustained thermal measurement is also meaningless with a second GPU consumer present: the whole
# point of J8 is to see whether the board throttles under a KNOWN load.
#
# ORDER. J8 first, and on an otherwise idle board:
#   * it is the shorter job (~30 min vs ~2 h) and the one with a hard cleanliness requirement
#   * the 2.22 GB of n=400 frames are pushed from the Mac BEFORE this script starts, so no disk I/O
#     competes with the thermal window either
# n=400 is resumable per clip and per eval, so if it is interrupted it costs at most one unit.
set -uo pipefail
cd "$(dirname "$0")/.."

echo "=============================================================="
echo "[ovn] start $(date -u +%FT%TZ)"
echo "[ovn] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"
who | sed 's/^/[ovn]   /'
df -h ~ | tail -1 | sed 's/^/[ovn]   /'
echo "=============================================================="

# ---------------------------------------------------------------- J8 / C9
# Phases 1 and 1b (sustained pose, FP16 and FP32) are already done, 2026-09-28; the script skips
# them. Phase 2, the sustained end-to-end run, has never been done and is the gap J8 names.
echo "[ovn] ---- J8: C9 sustained runs ----"
bash jetson/c9_sustained.sh 2>&1 | sed 's/^/[ovn-c9] /'
echo "[ovn] c9 exit=$? at $(date -u +%FT%TZ)"

# Let the board shed the thermal load and release its mappings before the next GPU consumer starts.
echo "[ovn] cooling 120 s"
sleep 120

# ---------------------------------------------------------------- J8 part b
# The other half of C9: "every final row = 3 runs, mean +- std". The audit (c9_process_repeats.sh
# header) found DVFS logged everywhere but repeats only ever WITHIN one process, while 2.2b shows
# process-to-process variance is the larger term. Short stage, so it goes before the long one.
echo "[ovn] ---- J8b: process-level repeats ----"
bash jetson/c9_process_repeats.sh 2>&1 | sed 's/^/[ovn-c9r] /'
echo "[ovn] c9r exit=$? at $(date -u +%FT%TZ)"
sleep 60

# ---------------------------------------------------------------- n=400
echo "[ovn] ---- n=400 accuracy pass ----"
n_clips=$(find data/clips -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
n_frames=$(find data/clips -mindepth 2 -maxdepth 2 -type d -name frames -not -empty | wc -l | tr -d ' ')
n_ref=$(ls data/openasl_pose_split/*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "[ovn] clips=${n_clips}  with frames=${n_frames}  ref poses=${n_ref}"
if [ "$n_ref" -lt 400 ]; then
  echo "[ovn] ABORT: only ${n_ref} reference poses; the ceiling must be scored on the same clip set" >&2
  exit 1
fi
SLT_TAG=n400 bash jetson/p10_split.sh 2>&1 | sed 's/^/[ovn-n400] /'
echo "[ovn] n400 exit=$? at $(date -u +%FT%TZ)"

# ---------------------------------------------------------------- reclaim the board
# Shared scratch: once a clip has both pkl normalisations its frames are redundant, and the Mac holds
# the originals. The end-to-end clip is kept -- 5.1/5.3/5.4 and C9 phase 2 all run on it.
KEEP_CLIP="ixq65EiuJ_c-00:03:47.633-00:03:56.133"
del=0
for d in data/clips/*/; do
  c=$(basename "$d")
  [ "$c" = "$KEEP_CLIP" ] && continue
  if [ -s "results/pkl_split_rtmw_fp16/$c.pkl" ] && [ -s "results/pkl_split_rtmw_fp16_raw/$c.pkl" ] \
     && [ -d "data/clips/$c/frames" ]; then
    rm -rf "data/clips/$c/frames"; del=$((del+1))
  fi
done
echo "[ovn] reclaimed frames for ${del} clips"
du -sh data/clips | sed 's/^/[ovn]   /'
df -h ~ | tail -1 | sed 's/^/[ovn]   /'

echo "[ovn] ALL DONE $(date -u +%FT%TZ)"
