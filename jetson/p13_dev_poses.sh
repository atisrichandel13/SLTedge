#!/usr/bin/env bash
# Extract DEV-split poses with the deployed pose front-end, for adaptation (RESULTS.md 2.5j).
#
# WHY DEV AND NOT TEST. The pose gap is a distribution shift: our keypoints agree with the authors'
# to a median 0.47% of the frame (2.5i) and no keypoint group carries the damage (2.5j), so the
# information is present and the model simply never trained on our extractor's noise. Fine-tuning on
# OUR keypoints should recover it. Adapting on test would be leakage, so adapt on dev (967 clips)
# and evaluate on test -- that keeps the n=400 test numbers in 2.5h honest.
#
# WHY ON THE BOARD. The adaptation target must be the DEPLOYED distribution, which is the RTMW-l-m
# FP16 TensorRT engine. An ONNX Runtime extraction off-board would be a near neighbour (P4: FP16 TRT
# agrees with FP32 on 99.81% of consumed keypoints within 5 px) but not the same thing, and the whole
# point is to match what ships.
#
# DISK. 967 clips of frames is ~6.9 GB and the board has ~17 GB free alongside weights/ (4.8 G) and
# models/ (2.4 G). So this works in BATCHES: it extracts every clip that currently has frames, then
# deletes the frames of clips whose pkls exist. Push a few hundred clips, run, push the next few
# hundred, run again. Both pkl dirs are checked before any delete.
#
# Resumable: a clip whose pkl exists is skipped, and pkls are written .part-then-renamed.
set -uo pipefail
cd "$(dirname "$0")/.."

CLIPS="${SLT_CLIPS:-data/clips_dev}"
ENG="${SLT_ENGINE:-models/rtmw/rtmw-l-m_256x192_fp16.engine}"
POSE="${SLT_POSE:-results/pkl_dev_rtmw_fp16}"
DROP="${SLT_DROP_FRAMES:-1}"      # 1 = reclaim frames once a clip has both pkls
R=(jetson/run.sh exec-batch)

[ -d "$CLIPS" ] || { echo "[dev] ABORT: no $CLIPS -- push dev clips (frames + meta.json) first" >&2; exit 1; }
[ -f "$ENG" ]  || { echo "[dev] ABORT: no engine at $ENG" >&2; exit 1; }

have_frames=$(find "$CLIPS" -mindepth 2 -maxdepth 2 -type d -name frames -not -empty | wc -l | tr -d ' ')
total=$(find "$CLIPS" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
echo "[dev] $CLIPS: $total clip dirs, $have_frames with frames"
echo "[dev] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"
who | sed 's/^/[dev]   /'; df -h ~ | tail -1 | sed 's/^/[dev]   /'
[ "$have_frames" -eq 0 ] && { echo "[dev] nothing to do"; exit 0; }

python3 jetson/drop_file_cache.py --target-free-mb=3500 2>&1 | tail -1
free=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
echo "[dev] MemFree ${free}MB"
[ "$free" -lt 2000 ] && { echo "[dev] ABORT: MemFree ${free}MB too low" >&2; exit 1; }

# --square-norm puts keypoints in the authors' square frame (2.5d), which is what the checkpoint and
# every eval in RESULTS.md expect. The _raw companion keeps the crop-frame form at no GPU cost.
"${R[@]}" python3 task1_rtmpose/09_batch_clips.py --engine "$ENG" --clips-dir "$CLIPS" \
    --config rtmw_fp16 --pkl-out "$POSE" --pkl-out-raw "${POSE}_raw" \
    --square-norm --progress-every 25 2>&1 | tail -6 | sed 's/^/[dev]   /'

n_pkl=$(ls "$POSE"/*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "[dev] $n_pkl pkls in $POSE"

if [ "$DROP" = "1" ]; then
  del=0
  for d in "$CLIPS"/*/; do
    c=$(basename "$d")
    if [ -s "$POSE/$c.pkl" ] && [ -s "${POSE}_raw/$c.pkl" ] && [ -d "$CLIPS/$c/frames" ]; then
      rm -rf "$CLIPS/$c/frames"; del=$((del+1))
    fi
  done
  echo "[dev] reclaimed frames for $del clips (both pkl normalisations verified first)"
  df -h ~ | tail -1 | sed 's/^/[dev]   /'
fi
echo "[dev] DONE $(date -u +%FT%TZ).  $n_pkl / $total clips have poses."
