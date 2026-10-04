#!/usr/bin/env bash
# J6 / C9: pose latency and energy across the crop-area range.
#
# WHY. Every pose energy number in RESULTS.md comes from ONE clip whose 644x720 crop is at the 83.4th
# percentile of crop area across the 931-clip split. frontier.py flags this as a ~7% pessimistic bias
# on the absolute energy column and it is the last caveat on that axis. ~11.1 of the 25.1 ms/frame
# scales with crop area, so a median clip should be cheaper -- never measured.
#
# Five clips spanning 50,660 to 766,080 px (15.1x). The clips with the extreme areas were chosen
# subject to 120-320 frames so each run is a usable power window, and the two truncated clips 2.5i
# found are excluded. Three SEPARATE processes per clip, per the protocol 5.6 established: repeats
# inside one process miss the process-to-process term, which for FP16 is the larger one (imread and
# preprocess at ~9% CV while TRT is 0.3%).
#
# Resumable: a (clip, rep) whose JSON exists is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

ENG=models/rtmw/rtmw-l-m_256x192_fp16.engine
OUT=results/j6_crop
REPS="${SLT_REPS:-3}"
mkdir -p "$OUT" results/logs
R=(jetson/run.sh exec-batch)

[ -f jetson/j6_clips.txt ] || { echo "[j6] ABORT: no jetson/j6_clips.txt" >&2; exit 1; }
echo "[j6] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"
echo "[j6] reps=$REPS  engine=$ENG"

while IFS= read -r clip; do
  [ -z "$clip" ] && continue
  f="data/clips/$clip/frames"
  if [ ! -d "$f" ]; then echo "[j6] SKIP $clip: no frames" >&2; continue; fi
  nf=$(ls "$f" | wc -l | tr -d ' ')
  area=$(python3 -c "
import json;m=json.load(open('data/clips/$clip/meta.json'));print(m['crop_xywh'][2]*m['crop_xywh'][3])")
  for rep in $(seq 1 "$REPS"); do
    safe=$(echo "$clip" | tr ':.' '__')
    out="$OUT/${safe}_rep${rep}.json"
    [ -s "$out" ] && { echo "[j6] have $out"; continue; }
    python3 jetson/drop_file_cache.py --target-free-mb=3000 >/dev/null 2>&1 || true
    free=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
    echo "[j6] === area=${area}px frames=${nf} rep${rep}  (users=$(who|wc -l) memfree=${free}MB)"
    "${R[@]}" python3 task1_rtmpose/04_infer_power.py --engine "$ENG" --frames "$f" --repeat 3 \
        --power-json "$out" > "results/logs/j6_${safe}_rep${rep}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then echo "[j6] FAILED rc=$rc"; tail -4 "results/logs/j6_${safe}_rep${rep}.log"; continue; fi
    python3 -c "
import json;d=json.load(open('$out'));l,p=d['latency'],d['power']
print('[j6] %9s px  rep%s  %6.2f ms/fr (trt %5.2f)  %5.3f W  %6.1f mJ/fr' % (
  '$area','$rep', l['wall_ms_per_frame'], l['trt_ms']['mean'], p['avg_watts'], p['mJ_per_frame']))"
    sleep 8   # settle so reps are independent rather than a thermal ramp
  done
done < jetson/j6_clips.txt
echo "[j6] ALL DONE $(date -u +%FT%TZ)"
