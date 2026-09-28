#!/usr/bin/env bash
# C9: the 30-minute sustained runs that have never been done. Launch detached:
#
#     setsid nohup ./jetson/c9_sustained.sh > results/logs/c9_sustained.log 2>&1 < /dev/null &
#
# Every latency and energy row in RESULTS.md comes from an 8-25 s window with Tj never above 53 C.
# If the board throttles under sustained load, those rows are optimistic and the accuracy-energy
# frontier moves. Two phases, pose first because it is the safe one:
#   1  pose engine only, 30 min   -- the stage that runs continuously in deployment
#   2  full pose->text loop, 30 min -- the real duty cycle, both models resident
# Phase 2 is the more honest test and the more likely to hit memory trouble over hundreds of
# iterations, so phase 1 banks a usable thermal result first.
#
# Skips a phase whose output already exists, so it resumes.
set -uo pipefail
cd "$(dirname "$0")/.."

CLIP="${SLT_CLIP:-ixq65EiuJ_c-00:03:47.633-00:03:56.133}"
ENGINE="${SLT_ENGINE:-models/rtmw/rtmw-l-m_256x192_fp16.engine}"
CKPT="${SLT_CKPT:-weights/openasl_pose_only_slt.pth}"
MT5="${SLT_MT5:-weights/mt5-base}"
DUR="${SLT_DURATION_S:-1800}"
R=(jetson/run.sh exec-batch)

prep_mem() {
    python3 jetson/drop_file_cache.py --target-free-mb=5200 2>&1 | tail -1
    local f; f=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
    [ "$f" -lt 2500 ] && { echo "[c9] ABORT: MemFree ${f} MB too low" >&2; return 1; }
    return 0
}

echo "[c9] start $(date)  duration ${DUR}s per phase  clip $CLIP"
echo "[c9] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"
echo "[c9] occupancy:"; who | sed 's/^/[c9]   /'; uptime | sed 's/^/[c9]   /'

# ---- phase 1: pose only
if [ -f results/c9_sustained_pose_iters.json ]; then
    echo "[c9] have phase 1"
else
    prep_mem || exit 1
    echo "[c9] PHASE 1: sustained pose, ${DUR}s  $(date +%H:%M:%S)"
    "${R[@]}" python3 -m unisign.sustained_run --mode pose --duration-s "$DUR" \
        --engine "$ENGINE" --frames "data/clips/$CLIP/frames" \
        --power-json results/c9_sustained_pose.json --power-csv results/c9_sustained_pose.csv \
        --out results/c9_sustained_pose_iters.json > results/logs/c9_pose.log 2>&1
    grep -E '^\[c9\] VERDICT|throttled|tj_C_max_overall|drift' results/logs/c9_pose.log | tail -5 \
        | sed 's/^/[c9]   /'
fi

# ---- phase 2: full pipeline
if [ -f results/c9_sustained_e2e_iters.json ]; then
    echo "[c9] have phase 2"
else
    prep_mem || exit 1
    echo "[c9] PHASE 2: sustained end-to-end, ${DUR}s  $(date +%H:%M:%S)"
    "${R[@]}" python3 -m unisign.sustained_run --mode e2e --duration-s "$DUR" \
        --engine "$ENGINE" --frames "data/clips/$CLIP/frames" --meta "data/clips/$CLIP/meta.json" \
        --ckpt "$CKPT" --mt5 "$MT5" --num-beams 4 \
        --power-json results/c9_sustained_e2e.json --power-csv results/c9_sustained_e2e.csv \
        --out results/c9_sustained_e2e_iters.json > results/logs/c9_e2e.log 2>&1
    grep -E '^\[c9\] VERDICT|throttled|tj_C_max_overall|drift' results/logs/c9_e2e.log | tail -5 \
        | sed 's/^/[c9]   /'
fi

echo "[c9] ALL DONE $(date)"
