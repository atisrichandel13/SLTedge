#!/usr/bin/env bash
# P8 + L7: energy and accuracy versus capture rate, plus the LM's own latency/energy matrix.
# Runs ON the Jetson. Launch detached so it survives the ssh session and the laptop:
#
#     setsid nohup ./jetson/p8_l7_sweep.sh > results/logs/p8_l7.log 2>&1 < /dev/null &
#
# Three parts, strictly sequential -- never two model-holding processes at once, which is what killed
# two evals on 2026-09-26:
#   A  pose-stage latency + power at 30 / 24 / 16 / 12 / 8 fps (real runs; energy needs real runs)
#   B  translation accuracy at the same rates, by subsampling the keypoints we already have. Pose
#      extraction is per-frame independent, so this is exact, not an approximation -- see
#      common/subsample_pkl.py. Saves ~40 min of board time versus re-extracting.
#   C  LM latency + power across beam width and encoder length, one process (model load is ~64 s)
#
# Every step is skipped if its output exists, so the script resumes after an interruption.
set -uo pipefail
cd "$(dirname "$0")/.."

RATES="${SLT_RATES:-30 24 16 12 8}"
CLIP="${SLT_CLIP:-ixq65EiuJ_c-00:03:47.633-00:03:56.133}"
ENGINE="${SLT_ENGINE:-models/rtmw/rtmw-l-m_256x192_fp16.engine}"
POSES="${SLT_POSES:-results/pkl_30clip_rtmw_fp16_sqnorm}"
CKPT_FULL="${SLT_CKPT:-weights/openasl_pose_only_slt.pth}"
MT5_FULL="${SLT_MT5:-weights/mt5-base}"
CKPT_PRUNED="${SLT_CKPT_PRUNED:-weights/openasl_pose_only_slt_pruned.pth}"
MT5_PRUNED="${SLT_MT5_PRUNED:-weights/mt5-base-openasl-pruned}"
LABELS="${SLT_LABELS:-data/openasl_labels/labels.test}"
REPEAT="${SLT_REPEAT:-3}"
MIN_FREE_MB="${SLT_MIN_FREE_MB:-4000}"

R=(jetson/run.sh exec-batch)

# nvmap allocates from MemFree only and never reclaims page cache. On a shared board fadvise alone is
# not enough (another user's desktop session holds the cache), so ask for a target and let the tool
# force kernel reclaim. Skipping this produces "NVML_SUCCESS == r INTERNAL ASSERT FAILED", which is an
# out-of-memory wearing a misleading hat -- see RESULTS.md 5.1.
prep_mem() {
    python3 jetson/drop_file_cache.py --target-free-mb="$MIN_FREE_MB" 2>&1 | tail -2
    local free_mb; free_mb=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
    if [ "$free_mb" -lt 2500 ]; then
        echo "[sweep] ABORT: MemFree ${free_mb} MB is too low to load a model safely" >&2
        return 1
    fi
    return 0
}

echo "[sweep] start $(date)  rates: $RATES  clip: $CLIP"
echo "[sweep] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"
echo "[sweep] board occupancy before starting:"
who | sed 's/^/[sweep]   /' || true
uptime | sed 's/^/[sweep]   /'

# ---------------------------------------------------------------- A: pose energy vs capture rate
for fps in $RATES; do
    out="results/p8_pose_fps${fps}.json"
    if [ -f "$out" ]; then echo "[sweep] have $out"; continue; fi
    prep_mem || exit 1
    echo "[sweep] A: pose at ${fps} fps  $(date +%H:%M:%S)"
    "${R[@]}" python3 task1_rtmpose/04_infer_power.py \
        --engine "$ENGINE" --frames "data/clips/$CLIP/frames" \
        --keep-fps "$fps" --repeat "$REPEAT" \
        --power-json "$out" --power-csv "results/p8_pose_fps${fps}.csv" \
        > "results/logs/p8_pose_fps${fps}.log" 2>&1
    grep -E '"(n_frames|keep_fps|mJ_per_frame|avg_watts|total_ms)"' "results/logs/p8_pose_fps${fps}.log" \
        | head -6 | sed 's/^/[sweep]   /'
done

# ---------------------------------------------------------------- B: accuracy vs capture rate
for fps in $RATES; do
    sub="${POSES}_fps${fps}"
    out="results/eval_30clip_fps${fps}.json"
    if [ -f "$out" ]; then echo "[sweep] have $out"; continue; fi
    if [ ! -d "$sub" ]; then
        "${R[@]}" python3 common/subsample_pkl.py --in "$POSES" --keep-fps "$fps" --out "$sub" \
            2>&1 | sed 's/^/[sweep]   /'
    fi
    prep_mem || exit 1
    echo "[sweep] B: eval at ${fps} fps  $(date +%H:%M:%S)"
    "${R[@]}" python3 unisign/eval_openasl.py --ckpt "$CKPT_FULL" --mt5 "$MT5_FULL" \
        --poses "$sub" --labels "$LABELS" --num-beams 4 --batch-size 1 --out "$out" \
        > "results/logs/eval_fps${fps}.log" 2>&1
    grep -E '"(bleu4|rouge_l|n)"' "results/logs/eval_fps${fps}.log" | head -4 | sed 's/^/[sweep]   /'
done

# ---------------------------------------------------------------- C: LM latency + power matrix
out=results/lm_sweep_pruned.json
if [ -f "$out" ]; then
    echo "[sweep] have $out"
elif [ ! -f "$CKPT_PRUNED" ]; then
    echo "[sweep] SKIP C: no pruned checkpoint at $CKPT_PRUNED" >&2
else
    prep_mem || exit 1
    echo "[sweep] C: LM matrix (pruned)  $(date +%H:%M:%S)"
    "${R[@]}" python3 -m unisign.lm_sweep --ckpt "$CKPT_PRUNED" --mt5 "$MT5_PRUNED" \
        --poses "$POSES" --n-clips 3 --beams 1 2 4 --lengths 256 205 137 103 68 \
        --repeat "$REPEAT" --out "$out" > results/logs/lm_sweep_pruned.log 2>&1
    tail -20 results/logs/lm_sweep_pruned.log | sed 's/^/[sweep]   /'
fi

echo "[sweep] ALL DONE $(date)"
