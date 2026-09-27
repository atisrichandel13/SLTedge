#!/usr/bin/env bash
# Every pose->BLEU eval for the 30-clip set, STRICTLY ONE AT A TIME, in a single process.
#
# Two PyTorch processes on this board kill each other: TensorRT/torch hit
# "NVML_SUCCESS == r INTERNAL ASSERT FAILED ... CUDACachingAllocator.cpp:1017" and both die. That has
# now happened twice -- once from treating drop_file_cache as build-time-only, once from launching a
# second eval behind a pgrep guard that did not hold. Hence: no backgrounding, no concurrency guard,
# one loop. Every eval is skipped if its output exists, so re-running after a crash resumes.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG=30clip
CKPT=weights/openasl_pose_only_slt.pth
MT5=weights/mt5-base
LABELS=data/openasl_labels/labels.test
MIN_FREE_MB=${SLT_MIN_FREE_MB:-1200}

R=(jetson/run.sh exec-batch)

eval_one() {  # $1 = pose dir, $2 = output name
    local poses="$1"
    local name="$2"
    local out="results/eval_${TAG}_${name}.json"
    if [ -f "$out" ]; then echo "[p5] have $out"; return 0; fi
    if [ ! -d "$poses" ]; then echo "[p5] MISSING pose dir $poses, skipping $name" >&2; return 0; fi
    python3 jetson/drop_file_cache.py >/dev/null 2>&1 || true
    # nvmap allocates only from MemFree and never reclaims page cache; below ~1.2 GB the LM load
    # fails mid-way and takes the run with it, so stop instead of producing a half-run
    local free_mb; free_mb=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
    if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
        echo "[p5] ABORT: MemFree ${free_mb} MB < ${MIN_FREE_MB} MB after dropping cache" >&2
        return 1
    fi
    echo "[p5] eval $name  (MemFree ${free_mb} MB)  $(date +%H:%M:%S)"
    local t0=$SECONDS
    "${R[@]}" python3 unisign/eval_openasl.py --ckpt "$CKPT" --mt5 "$MT5" --poses "$poses" \
        --labels "$LABELS" --num-beams 4 --out "$out"
    echo "[p5] done $name in $((SECONDS - t0)) s"
}

for cfg in rtmw_fp32 rtmw_fp16 rtmposex_fp32 rtmposex_fp16mixed; do
    pkl="results/pkl_${TAG}_${cfg}"
    n=$(ls results/kpts/${cfg}__*.json 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ] && { [ ! -d "$pkl" ] || [ "$(ls "$pkl"/*.pkl 2>/dev/null | wc -l)" -ne "$n" ]; }; then
        echo "[p5] convert $cfg: $n dumps -> $pkl"
        "${R[@]}" python3 common/dumps_to_pkl.py --dumps results/kpts --config "$cfg" \
            --clips data/clips --out "$pkl"
    fi
    eval_one "$pkl" "$cfg"
done

# the ceiling: the authors' own released poses through the same frozen checkpoint
eval_one data/openasl_pose authors_ceiling

# the frame-correction test: our own poses, re-expressed in the authors' square frame
# (common/renorm_to_openasl.py). Everything else is held fixed, so a delta here is the frame alone.
for cfg in rtmw_fp32 rtmw_fp16; do
    sq="results/pkl_${TAG}_${cfg}_sqnorm"
    if [ ! -d "$sq" ] && [ -d "results/pkl_${TAG}_${cfg}" ]; then
        "${R[@]}" python3 common/renorm_to_openasl.py --in "results/pkl_${TAG}_${cfg}" \
            --clips data/clips --out "$sq"
    fi
    eval_one "$sq" "${cfg}_sqnorm"
done

echo "[p5] ALL DONE $(date +%H:%M:%S)"
