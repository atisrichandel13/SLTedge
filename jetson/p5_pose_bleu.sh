#!/usr/bin/env bash
# Pose -> BLEU for every pose config over the clips in data/clips, plus the authors' poses as the
# ceiling row. Runs ON the Jetson from ~/sign-lang-project. Feeds guide row 2.5.
#
#     ./jetson/p5_pose_bleu.sh                        # 4 configs + ceiling
#     ./jetson/p5_pose_bleu.sh rtmw_fp32 rtmw_fp16    # a subset, no ceiling row
#     SLT_TAG=30clip ./jetson/p5_pose_bleu.sh         # names the output files
#
# Each config: dumps -> Uni-Sign pkl (common/dumps_to_pkl.py) -> eval (unisign/eval_openasl.py).
# Already-finished evals are skipped, so this is safe to re-run after an interruption.
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${SLT_TAG:-30clip}"
DUMPS="${SLT_KPTS:-results/kpts}"
CLIPS="${SLT_CLIPS:-data/clips}"
CKPT="${SLT_CKPT:-weights/openasl_pose_only_slt.pth}"     # released checkpoint = the FP32 reference
MT5="${SLT_MT5:-weights/mt5-base}"
LABELS="${SLT_LABELS:-data/openasl_labels/labels.test}"
REF_POSE="${SLT_REF_POSE:-data/openasl_pose}"             # authors' released poses = ceiling
BEAMS="${SLT_BEAMS:-4}"

R=(jetson/run.sh exec-batch)
# nvmap allocates only from MemFree and never reclaims page cache. The LM is ~1.2 GB and five evals
# were lost to NvMapMemAllocInternalTagged error 12 before this was done per-eval rather than per-build.
drop_cache() { python3 jetson/drop_file_cache.py >/dev/null 2>&1 || true; }

CONFIGS=("$@")
CEILING=1
if [ "${#CONFIGS[@]}" -gt 0 ]; then CEILING=0; else CONFIGS=(rtmw_fp32 rtmw_fp16 rtmposex_fp32 rtmposex_fp16mixed); fi

echo "[p5] tag=$TAG  clips=$(ls -d "$CLIPS"/*/ 2>/dev/null | wc -l)  beams=$BEAMS  ckpt=$CKPT"
echo "[p5] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"

run_eval() {  # $1 = pose dir, $2 = output tag
    local poses="$1" name="$2" out="results/eval_${TAG}_${name}.json"
    if [ -f "$out" ]; then echo "[p5] skip $out"; return; fi
    drop_cache
    local t0=$SECONDS
    "${R[@]}" python3 unisign/eval_openasl.py --ckpt "$CKPT" --mt5 "$MT5" --poses "$poses" \
        --labels "$LABELS" --num-beams "$BEAMS" --out "$out"
    echo "[p5] eval $name  $((SECONDS - t0)) s"
}

for cfg in "${CONFIGS[@]}"; do
    n=$(ls "$DUMPS"/${cfg}__*.json 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then echo "[p5] no dumps for $cfg in $DUMPS, skipping" >&2; continue; fi
    pkl="results/pkl_${TAG}_${cfg}"
    if [ ! -d "$pkl" ] || [ "$(ls "$pkl"/*.pkl 2>/dev/null | wc -l)" -ne "$n" ]; then
        echo "[p5] $cfg: $n dumps -> $pkl"
        "${R[@]}" python3 common/dumps_to_pkl.py --dumps "$DUMPS" --config "$cfg" \
            --clips "$CLIPS" --out "$pkl"
    fi
    run_eval "$pkl" "$cfg"
done

# The ceiling is the authors' own released poses through the same checkpoint: it bounds what any pose
# front-end could score on these clips, so our rows are only interpretable next to it.
if [ "$CEILING" -eq 1 ] && [ -d "$REF_POSE" ]; then
    run_eval "$REF_POSE" "authors_ceiling"
fi

echo "[p5] done. On the Mac: pull results/eval_${TAG}_*.json, then"
echo "     python -m unisign.bootstrap_ci results/eval_${TAG}_rtmw_fp32.json \\"
echo "         results/eval_${TAG}_authors_ceiling.json -n 5000"
