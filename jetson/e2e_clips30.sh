#!/bin/bash
# End-to-end on the board over 30 test clips x 3 repeats, three pose front ends, one LM.
# Answers ASK-LATENCY-CLIP-COUNT-2026-11-04: every earlier e2e row was 3 repeats of ONE clip, which
# measures run-to-run jitter, not the across-sentence variance that governs RTF (the decoder scales with
# output tokens, the pose stage with frames). results/e2e_clips30.txt is the 931-clip test set sorted
# by duration and sampled at 30 evenly spaced quantiles -- a rule, not a choice.
#
#   shipped   RTMW-l-m 256x192 FP16                       (the deployed baseline, re-measured here)
#   xl        RTMW-x-l 256x192 FP16                       (§L33: +0.83 BLEU-4 established)
#   xl_yolo   RTMW-x-l + yolox-tiny box on the authors' 224 square (the board finds the signer)
#
# Same LM for all three: pruned checkpoint, pre-pruned mT5 dir, FP32, beam 4, cap 64, 24 fps.
#   ./jetson/e2e_clips30.sh            # all arms, skipping any already written
#   ARMS="xl" ./jetson/e2e_clips30.sh
set -u
cd "$(dirname "$0")/.."
CKPT=weights/openasl_pose_only_slt_pruned.pth
MT5=weights/mt5-base-openasl-pruned
LIST=results/e2e_clips30.txt
OUT=results/e2e_clips30
LM_ENG=models/rtmw/rtmw-l-m_256x192_fp16.engine
XL_ENG=models/rtmw_xl/rtmw-x-l_256x192_fp16.engine
DET_ENG=models/yolox_tiny/yolox_tiny_prenms_fp16.engine
mkdir -p "$OUT" results/logs
STATE="$OUT/board_state.jsonl"
GPU=/sys/devices/platform/bus@0/17000000.gpu/devfreq/17000000.gpu

# One JSON line of the conditions a latency number depends on. The board is shared: a number taken
# while another user's job holds the CPU or GPU is not a measurement of our pipeline.
board_state() {   # $1 = label
  local mode max foreign fcmds others ours tj free
  mode=$(nvpmodel -q 2>/dev/null | head -1 | sed 's/NV Power Mode: //')
  max=$(( $(cat $GPU/max_freq) / 1000000 ))
  foreign=$(ps -eo user,pcpu,comm --sort=-pcpu | awk 'NR>1 && $1!="tgoyal" && $1!="root" && $2+0>1.0' | wc -l)
  fcmds=$(ps -eo user,pcpu,comm --sort=-pcpu | awk 'NR>1 && $1!="tgoyal" && $1!="root" && $2+0>1.0 {printf "%s:%s:%s ", $1,$3,$2}')
  others=$(docker ps --format '{{.Names}}' | grep -vx slt-work | wc -l)
  ours=$(docker exec slt-work sh -c 'pgrep -fc [p]ython || true' 2>/dev/null | head -1); ours=${ours:-0}
  tj=$(timeout 2.5 tegrastats --interval 1000 2>/dev/null | tail -1 | grep -o 'tj@[0-9.]*' | cut -d@ -f2)
  free=$(awk '/MemFree/{printf "%d", $2/1024}' /proc/meminfo)
  echo "{\"label\": \"$1\", \"time\": \"$(date -Is)\", \"power_mode\": \"$mode\", \"gpu_max_MHz\": $max, \"users\": $(who | wc -l), \"foreign_procs_gt1pct\": $foreign, \"foreign\": \"$fcmds\", \"other_containers\": $others, \"our_python_procs\": $ours, \"tj_C\": \"${tj:-NA}\", \"memfree_MB\": $free}" | tee -a "$STATE"
  [ "$mode" = "15W" ] && [ "$max" -eq 612 ] && [ "$foreign" -eq 0 ] && [ "$others" -eq 0 ] && [ "$ours" -eq 0 ] \
    && [ "${tj%.*}" -lt 60 ] 2>/dev/null
}

for arm in ${ARMS:-shipped xl xl_yolo}; do
  case $arm in
    shipped) eng=$LM_ENG; extra="" ;;
    xl)      eng=$XL_ENG; extra="" ;;
    xl_yolo) eng=$XL_ENG; extra="--det-engine $DET_ENG --det-size 416" ;;
    *) echo "unknown arm $arm"; continue ;;
  esac
  sj="$OUT/e2e_$arm.json"
  if [ -s "$sj" ]; then echo "[e2e30] $arm already done, skipping"; continue; fi
  ok=0
  for i in $(seq 1 20); do
    if board_state "before_$arm"; then ok=1; break; fi
    echo "[e2e30] $arm: board not clean (try $i/20), waiting 30 s"; sleep 30
  done
  if [ $ok -ne 1 ]; then echo "[e2e30] $arm SKIPPED: board never clean for 10 min, no number recorded"; continue; fi
  users=$(who | wc -l)
  foreign=$(ps -eo user,pcpu,comm --sort=-pcpu | awk 'NR>1 && $1!="tgoyal" && $1!="root" && $2+0>1.0' | wc -l)
  python3 jetson/drop_file_cache.py --target-free-mb=4000 2>&1 | tail -1
  free=$(awk '/MemFree/{printf "%d", $2/1024}' /proc/meminfo)
  echo "[e2e30] === $arm  (users=$users foreign=$foreign memfree=${free}MB after reclaim) $(date +%T)"
  # sample other users' load DURING the arm; a line is written only when there is some
  fd="$OUT/foreign_during_$arm.txt"; : > "$fd"
  ( while true; do
      e=$(ps -eo user,pcpu,comm --sort=-pcpu | awk 'NR>1 && $1!="tgoyal" && $1!="root" && $2+0>1.0 {printf "%s:%s:%s ", $1,$3,$2}')
      [ -n "$e" ] && echo "$(date +%T) $e"
      sleep 10
    done ) >> "$fd" 2>&1 &
  mon=$!
  ./jetson/run.sh exec-batch python3 -m unisign.e2e_translate \
      --engine "$eng" $extra \
      --clip-list "$LIST" --clips-dir data/clips \
      --ckpt "$CKPT" --mt5 "$MT5" \
      --num-beams 4 --keep-fps 24 --repeat 3 \
      --power-interval-ms 50 \
      --out "$sj" --power-json "$OUT/power_$arm.json" \
      > "results/logs/e2e_clips30_$arm.log" 2>&1
  rc=$?
  kill $mon 2>/dev/null; wait $mon 2>/dev/null
  echo "[e2e30] $arm exit=$rc $(date +%T) (users_after=$(who | wc -l), samples with foreign load: $(wc -l < "$fd"))"
  board_state "after_$arm" > /dev/null || echo "[e2e30] $arm: board was NOT clean at the end -- check $STATE before trusting this arm"
  [ $rc -ne 0 ] && tail -8 "results/logs/e2e_clips30_$arm.log"
  sleep 5
done
