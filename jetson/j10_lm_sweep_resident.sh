#!/usr/bin/env bash
# J10: re-run the 2.9C beam x T sweep with the RTMW FP16 pose engine RESIDENT in the same process.
# Runs ON the Jetson. Launch detached:
#
#     nohup bash jetson/j10_lm_sweep_resident.sh > results/logs/j10.log 2>&1 < /dev/null &
#
# WHY. frontier.py composes LM_J from 2.9C, which measured the LM STANDALONE -- one process, no pose
# engine resident, no second TensorRT context in the shared 8 GB pool. Every end-to-end run has both
# models loaded, and the composition understates the decoder-width term by 2.08-2.27x (RESULTS.md
# 5.4). The error is signed consistently across all four cells: it overestimates greedy and
# underestimates beam 4, compressing the spread from both ends, which is what a different resident
# footprint would do. Four cells sharing a sign is a pattern, not a cause (OPEN-ISSUES-LM, J10).
#
# WHAT IT DECIDES. If residency reproduces the sign pattern, LM_J is simply the wrong table to compose
# from and the fix is to re-measure it in context. If it does not, the discrepancy is something else
# and the beam-width axis of the frontier stays unquotable -- it is the only axis REPORT.md 6 tells
# people not to quote.
#
# EVERYTHING ELSE IS HELD TO 2.9C. Same pruned checkpoint, same FULL mT5 (the pruned ckpt carries
# keep_ids indexing the original 250,112-token vocabulary and load_model does the slicing itself --
# handing it the pre-pruned directory raises "index 99537 is out of bounds"), same poses, same
# --n-clips 3, same beams, same lengths, same --repeat 3 in ONE process. The sweep loads the model
# once because that costs ~64 s on this board, and each configuration still gets its own power window
# with its own idle baseline.
#
# ONE PROCESS AT A TIME: two PyTorch/TensorRT processes on this board kill each other.
set -uo pipefail
cd "$(dirname "$0")/.."

ENGINE="${SLT_ENGINE:-models/rtmw/rtmw-l-m_256x192_fp16.engine}"
POSES="${SLT_POSES:-results/pkl_30clip_rtmw_fp16_sqnorm}"
CKPT="${SLT_CKPT_PRUNED:-weights/openasl_pose_only_slt_pruned.pth}"
MT5="${SLT_MT5:-weights/mt5-base}"
REPEAT="${SLT_REPEAT:-3}"
OUT="${SLT_OUT:-results/lm_sweep_pruned_resident.json}"
MIN_FREE_MB="${SLT_MIN_FREE_MB:-5200}"
R=(jetson/run.sh exec-batch)

if [ -f "$OUT" ]; then echo "[j10] have $OUT -- nothing to do"; exit 0; fi
for f in "$ENGINE" "$CKPT"; do
    [ -e "$f" ] || { echo "[j10] ABORT: missing $f" >&2; exit 1; }
done
[ -d "$MT5" ] || { echo "[j10] ABORT: missing $MT5" >&2; exit 1; }

echo "[j10] start $(date -u +%FT%TZ)"
echo "[j10] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"
who | sed 's/^/[j10]   /'

python3 jetson/drop_file_cache.py --target-free-mb="$MIN_FREE_MB" 2>&1 | tail -2 | sed 's/^/[j10]   /'
free_mb=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
echo "[j10]   MemFree ${free_mb} MB (want ${MIN_FREE_MB})"
# 4000 was not enough for the longest encoder sequence in the original sweep; this run additionally
# holds a TensorRT pose context, so the bar is higher, not lower.
if [ "$free_mb" -lt 4000 ]; then echo "[j10] ABORT: MemFree ${free_mb} MB too low" >&2; exit 1; fi

"${R[@]}" python3 -m unisign.lm_sweep --ckpt "$CKPT" --mt5 "$MT5" \
    --poses "$POSES" --n-clips 3 --beams 1 2 4 --lengths 256 205 137 103 68 \
    --repeat "$REPEAT" --pose-engine "$ENGINE" --out "$OUT" 2>&1 | sed 's/^/[j10]   /'
rc=${PIPESTATUS[0]}
echo "[j10] sweep exit=${rc}"
[ "$rc" -eq 0 ] && echo "[j10] wrote $OUT"
echo "[j10] ALL DONE $(date -u +%FT%TZ)"
