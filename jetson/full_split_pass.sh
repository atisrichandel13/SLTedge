#!/usr/bin/env bash
# Phase A stages 3-4: keypoints + pkl + evals for the full test split. Launch detached:
#
#     setsid nohup ./jetson/full_split_pass.sh > results/logs/full_split.log 2>&1 < /dev/null &
#
# RESUMABILITY, which is the whole design constraint:
#   * extraction is per-clip: a clip whose pkl exists is skipped (09_batch_clips.py), and the pkl is
#     written to a .part file and renamed, so it never exists half-written
#   * each eval is skipped when its output JSON exists
#   * so an interrupted run, a closed laptop or a killed process costs at most one clip
# Re-running is therefore always safe and always cheap.
#
# The engine is built once per configuration and every clip reuses it: p4_clips.sh spent ~9 s of docker
# exec + engine load + CUDA init per clip, which at 974 clips is 2.4 h of pure overhead per config.
set -uo pipefail
cd "$(dirname "$0")/.."

CONFIGS="${SLT_CONFIGS:-rtmw_fp16 rtmw_fp32 rtmposex_fp32 rtmposex_fp16mixed}"
CLIPS_DIR="${SLT_CLIPS:-data/clips}"
TAG="${SLT_TAG:-full}"
CKPT="${SLT_CKPT:-weights/openasl_pose_only_slt.pth}"
MT5="${SLT_MT5:-weights/mt5-base}"
LABELS="${SLT_LABELS:-data/openasl_labels/labels.test}"
REF_POSE="${SLT_REF_POSE:-data/openasl_pose}"
BEAMS="${SLT_BEAMS:-4}"
BATCH="${SLT_BATCH:-8}"
DROP_FRAMES="${SLT_DROP_FRAMES:-0}"   # 1 = delete frames once every config has its pkls

R=(jetson/run.sh exec-batch)

prep_mem() {
    local want="${1:-5200}" f=0 a
    sleep 5
    for a in 1 2 3; do
        python3 jetson/drop_file_cache.py --target-free-mb="$want" 2>&1 | tail -1
        f=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
        [ "$f" -ge "$want" ] && break
        sleep 10
    done
    echo "[full]   MemFree ${f} MB"
    [ "$f" -lt 3000 ] && { echo "[full] ABORT: MemFree ${f} MB too low" >&2; return 1; }
    return 0
}

engine_for() {
    case "$1" in
        rtmw_fp32)          echo models/rtmw/rtmw-l-m_256x192_fp32.engine ;;
        rtmw_fp16)          echo models/rtmw/rtmw-l-m_256x192_fp16.engine ;;
        rtmposex_fp32)      echo models/rtmpose-x_fp32.engine ;;
        rtmposex_fp16mixed) echo models/rtmpose-x_fp16mixed.engine ;;
        *) echo "" ;;
    esac
}

n_clips=$(find "$CLIPS_DIR" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
echo "[full] start $(date)  clips on board: ${n_clips}  configs: ${CONFIGS}"
echo "[full] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"
who | sed 's/^/[full]   /'
df -h ~ | tail -1 | sed 's/^/[full]   /'

# ---- stage 3: keypoints -> pkl, one process per config
for cfg in $CONFIGS; do
    eng=$(engine_for "$cfg")
    if [ -z "$eng" ] || [ ! -f "$eng" ]; then
        echo "[full] SKIP $cfg: no engine at '${eng}'" >&2; continue
    fi
    pkl="results/pkl_${TAG}_${cfg}"
    have=$(ls "$pkl"/*.pkl 2>/dev/null | wc -l | tr -d ' ')
    if [ "$have" -ge "$n_clips" ]; then echo "[full] have all ${have} pkls for $cfg"; continue; fi
    prep_mem 3500 || exit 1
    echo "[full] STAGE3 $cfg ($have/$n_clips done)  $(date +%H:%M:%S)"
    "${R[@]}" python3 task1_rtmpose/09_batch_clips.py --engine "$eng" --clips-dir "$CLIPS_DIR" \
        --config "$cfg" --pkl-out "$pkl" --square-norm --progress-every 25 \
        2>&1 | tail -4 | sed 's/^/[full]   /'
done

# ---- optionally reclaim the frames: the board is shared scratch and the pkls are what the evals need
if [ "$DROP_FRAMES" = "1" ]; then
    ok=1
    for cfg in $CONFIGS; do
        [ -z "$(engine_for "$cfg")" ] && continue
        have=$(ls "results/pkl_${TAG}_${cfg}"/*.pkl 2>/dev/null | wc -l | tr -d ' ')
        [ "$have" -lt "$n_clips" ] && ok=0
    done
    if [ "$ok" = "1" ]; then
        echo "[full] every config has $n_clips pkls; deleting frames to free the board"
        find "$CLIPS_DIR" -mindepth 2 -maxdepth 2 -type d -name frames -exec rm -rf {} +
        df -h ~ | tail -1 | sed 's/^/[full]   /'
    else
        echo "[full] not deleting frames: some config is incomplete"
    fi
fi

# ---- stage 4: evals
for cfg in $CONFIGS; do
    [ -z "$(engine_for "$cfg")" ] && continue
    pkl="results/pkl_${TAG}_${cfg}"
    out="results/eval_${TAG}_${cfg}.json"
    [ -f "$out" ] && { echo "[full] have $out"; continue; }
    [ -d "$pkl" ] || continue
    prep_mem 5200 || exit 1
    echo "[full] STAGE4 eval $cfg  $(date +%H:%M:%S)"
    "${R[@]}" python3 unisign/eval_openasl.py --ckpt "$CKPT" --mt5 "$MT5" --poses "$pkl" \
        --labels "$LABELS" --num-beams "$BEAMS" --batch-size "$BATCH" --out "$out" \
        > "results/logs/eval_${TAG}_${cfg}.log" 2>&1
    grep -E '"(bleu4|rouge_l|n|missing)"' "results/logs/eval_${TAG}_${cfg}.log" | head -4 \
        | sed 's/^/[full]   /'
done

# ---- ceiling: the authors' released poses through the same checkpoint
out="results/eval_${TAG}_authors_ceiling.json"
if [ -f "$out" ]; then
    echo "[full] have $out"
elif [ -d "$REF_POSE" ]; then
    prep_mem 5200 || exit 1
    echo "[full] STAGE4 eval authors_ceiling  $(date +%H:%M:%S)"
    "${R[@]}" python3 unisign/eval_openasl.py --ckpt "$CKPT" --mt5 "$MT5" --poses "$REF_POSE" \
        --labels "$LABELS" --num-beams "$BEAMS" --batch-size "$BATCH" --out "$out" \
        > "results/logs/eval_${TAG}_ceiling.log" 2>&1
    grep -E '"(bleu4|rouge_l|n)"' "results/logs/eval_${TAG}_ceiling.log" | head -3 | sed 's/^/[full]   /'
fi

echo "[full] ALL DONE $(date)"
