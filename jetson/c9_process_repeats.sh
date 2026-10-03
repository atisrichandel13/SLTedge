#!/usr/bin/env bash
# J8 / C9, the half that was actually missing: "every final row = 3 runs, mean ± std".
#
# WHAT WE HAD, and why it does not satisfy that. Audited 2026-10-03 across every power-bearing
# result file:
#   * DVFS is logged everywhere -- gpu_MHz, cpu0_MHz and temp_tj_C are in aux_avg of all 15 power
#     files. That half of C9 was already met.
#   * repeats, where they exist, are 3 PASSES INSIDE ONE PROCESS: the p8 rows cover 3 passes in one
#     power window (n_frames 612 = 204 x 3), and the e2e rows report mean ± std over 3 sentences in
#     one process. The five pose-engine rows report a per-frame distribution over 299 frames from a
#     SINGLE process and no repeat at all.
#   * 2.2b is direct evidence that the missing term is the bigger one: between two runs of the same
#     config, imread went 5.33 -> 6.83 ms and preprocess 4.76 -> 6.10, both x1.281. That is
#     process-to-process variance, and no within-process pass count can see it.
#
# So this runs each config as THREE SEPARATE PROCESSES -- fresh engine load, fresh CUDA context,
# fresh power window each time -- and the Mac-side aggregator reports mean ± std across them. Each
# process is short, so the whole stage is a few minutes.
#
# Resumable: a rep whose JSON exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

CLIP="${SLT_CLIP:-ixq65EiuJ_c-00:03:47.633-00:03:56.133}"
FRAMES="data/clips/$CLIP/frames"
REPS="${SLT_REPS:-3}"
OUT=results/c9_reps
mkdir -p "$OUT" results/logs
R=(jetson/run.sh exec-batch)

if [ ! -d "$FRAMES" ]; then
  echo "[c9r] ABORT: no frames at $FRAMES (they are the one clip kept on the board on purpose)" >&2
  exit 1
fi
nf=$(ls "$FRAMES" | wc -l | tr -d ' ')
echo "[c9r] clip=$CLIP frames=$nf reps=$REPS"
echo "[c9r] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"

engine_for() {
  case "$1" in
    rtmw_fp16) echo models/rtmw/rtmw-l-m_256x192_fp16.engine ;;
    rtmw_fp32) echo models/rtmw/rtmw-l-m_256x192_fp32.engine ;;
    *) echo "" ;;
  esac
}

for cfg in rtmw_fp16 rtmw_fp32; do
  eng=$(engine_for "$cfg")
  if [ -z "$eng" ] || [ ! -f "$eng" ]; then echo "[c9r] SKIP $cfg: no engine $eng" >&2; continue; fi
  for rep in $(seq 1 "$REPS"); do
    out="$OUT/${cfg}_rep${rep}.json"
    [ -s "$out" ] && { echo "[c9r] have $out"; continue; }
    users=$(who | wc -l)
    python3 jetson/drop_file_cache.py --target-free-mb=3000 >/dev/null 2>&1 || true
    free=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
    echo "[c9r] === $cfg rep$rep  (users=$users memfree=${free}MB)  $(date -u +%TZ)"
    "${R[@]}" python3 task1_rtmpose/04_infer_power.py \
        --engine "$eng" --frames "$FRAMES" --repeat 3 \
        --power-json "$out" > "results/logs/c9r_${cfg}_rep${rep}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "[c9r] FAILED $cfg rep$rep rc=$rc"; tail -4 "results/logs/c9r_${cfg}_rep${rep}.log"; continue; fi
    python3 -c "
import json;d=json.load(open('$out'))
l,p=d['latency'],d['power']
print('[c9r] %-16s rep%s  total %6.2f ms/frame  %5.3f W  %6.1f mJ/fr  gpu %sMHz  Tj %.1fC' % (
  '$cfg','$rep', l['wall_ms_per_frame'], p['avg_watts'], p['mJ_per_frame'],
  p['aux_avg'].get('gpu_MHz'), p['aux_avg'].get('temp_tj_C')))"
    sleep 8   # let clocks and temperature settle so reps are independent, not a ramp
  done
done
echo "[c9r] ALL DONE $(date -u +%FT%TZ)"
