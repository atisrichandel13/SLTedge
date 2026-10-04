#!/usr/bin/env bash
# Does the -2.24 BLEU-4 pose gap (2.5h) live in the HANDS?
#
# 2.5i showed our keypoints sit a median 0.47% of the frame from the authors' with no systematic
# offset, so a different pose model is unlikely to recover the gap -- but it also showed hands are
# the worst group (median 0.0060, p95 0.034 ~ 17 px on a 500 px square) and the face the best. Hand
# shape is the signal in sign language, so a small hand-only error is exactly the kind that could
# cost 2 BLEU-4 while looking negligible in a table of means.
#
# This swaps ONLY the hand keypoints (COCO-WholeBody 91..132, both hands) between the two sources
# and re-scores. Four rows on one matched clip set:
#   ours_m            our keypoints                          -- baseline
#   theirs_m          authors' keypoints                     -- ceiling
#   theirs_ourhands   authors' body+face, OUR hands          -- if this falls to ours_m, hands carry it
#   ours_theirhands   our body+face, AUTHORS' hands          -- if this rises to theirs_m, same conclusion
#
# Restricted to clips where both sources have identical frame counts, so a swap is frame-for-frame.
# 2.5i found 227 of 400 match; the rest differ by +-1 frame from decode boundaries. Using one matched
# subset for all four rows is the point -- 2.5c records quoting a ceiling at a different n than the
# rows it was compared against, and this must not repeat it.
set -uo pipefail
cd "$(dirname "$0")/.."

OURS=results/pkl_split_rtmw_fp16
THEIRS=data/openasl_pose_split
OUT=results/pkl_hybrid
CKPT=weights/openasl_pose_only_slt_pruned.pth
MT5=weights/mt5-base-openasl-pruned
LABELS=data/openasl_labels/labels.test
R=(jetson/run.sh exec-batch)

echo "[hyb] building hybrid pose sets  $(date -u +%FT%TZ)"
# A FILE, not a heredoc: run.sh exec-batch is `docker exec` without -i, so stdin never reaches the
# container and `python3 -` reads an empty program and exits 0. That silently produced no pose dirs
# and the first eval then died in rouge with ZeroDivisionError on an empty clip list.
"${R[@]}" python3 task1_rtmpose/11_make_hybrid.py --ours "$OURS" --theirs "$THEIRS" --out "$OUT" \
  || { echo "[hyb] ABORT: hybrid build failed" >&2; exit 1; }
for d in ours_m theirs_m theirs_ourhands ours_theirhands; do
  c=$(ls "$OUT/$d"/*.pkl 2>/dev/null | wc -l | tr -d " ")
  echo "[hyb]   $d: $c pkls"
  [ "$c" -lt 50 ] && { echo "[hyb] ABORT: $d has only $c pkls" >&2; exit 1; }
done

for cfg in ours_m theirs_m theirs_ourhands ours_theirhands; do
  out="results/eval_hybrid_${cfg}.json"
  [ -s "$out" ] && { echo "[hyb] have $out"; continue; }
  python3 jetson/drop_file_cache.py --target-free-mb=3000 >/dev/null 2>&1 || true
  free=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
  echo "[hyb] === $cfg  (users=$(who|wc -l) memfree=${free}MB)  $(date -u +%TZ)"
  "${R[@]}" python3 unisign/eval_openasl.py --ckpt "$CKPT" --mt5 "$MT5" \
      --poses "$OUT/$cfg" --labels "$LABELS" --num-beams 4 --max-new-tokens 64 --batch-size 1 \
      --fps 24 --out "$out" > "results/logs/eval_hybrid_${cfg}.log" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[hyb] FAILED $cfg rc=$rc"; tail -4 "results/logs/eval_hybrid_${cfg}.log"; continue; }
  python3 -c "
import json;d=json.load(open('$out'))
print('[hyb] %-18s n=%-4d BLEU-4 %6.2f  ROUGE-L %6.2f' % ('$cfg', d['n'], d['bleu']['bleu4'], d['rouge_l']))"
done
echo "[hyb] ALL DONE $(date -u +%FT%TZ)"
