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

# A 30-minute run reads its frames hundreds of times and leaves the page cache full, and the previous
# process needs a moment to release its mappings before fadvise can do anything. Settle, then retry:
# a single attempt straight after phase 1 left MemFree at 1659 MB and the LM load died of a masked OOM.
prep_mem() {
    local want="${1:-5200}" f=0 attempt
    sleep 15
    for attempt in 1 2 3; do
        python3 jetson/drop_file_cache.py --target-free-mb="$want" 2>&1 | tail -1
        f=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
        [ "$f" -ge "$want" ] && break
        echo "[c9]   attempt ${attempt}: MemFree ${f} MB < ${want} MB, retrying"
        sleep 10
    done
    echo "[c9]   MemFree before load: ${f} MB"
    [ "$f" -lt 3500 ] && { echo "[c9] ABORT: MemFree ${f} MB too low to load both models" >&2; return 1; }
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

# ---- phase 1b: pose at FP32. C9 asks for a sustained run for FP32 *and* the best compressed config,
# so the FP16 result above needs its uncompressed counterpart to be comparable.
if [ -f results/c9_sustained_pose_fp32_iters.json ]; then
    echo "[c9] have phase 1b"
else
    prep_mem 4500 || exit 1
    echo "[c9] PHASE 1b: sustained pose FP32, ${DUR}s  $(date +%H:%M:%S)"
    "${R[@]}" python3 -m unisign.sustained_run --mode pose --duration-s "$DUR" \
        --engine models/rtmw/rtmw-l-m_256x192_fp32.engine --frames "data/clips/$CLIP/frames" \
        --power-json results/c9_sustained_pose_fp32.json \
        --power-csv results/c9_sustained_pose_fp32.csv \
        --out results/c9_sustained_pose_fp32_iters.json > results/logs/c9_pose_fp32.log 2>&1
    grep -E '^\[c9\] VERDICT' results/logs/c9_pose_fp32.log | tail -2 | sed 's/^/[c9]   /'
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
