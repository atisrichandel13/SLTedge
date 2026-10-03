#!/usr/bin/env bash
# The deployable config has never had its accuracy measured. Every board eval in RESULTS.md 2.5c
# uses the FULL released checkpoint (openasl_pose_only_slt.pth); the model we actually intend to
# ship is the PRUNED one, which 5.4 showed is -12% system energy, half the peak memory and the only
# one that runs source x beam 4 at all. So "our accuracy on the deployable model" is currently an
# estimate composed from two separate experiments, not a measurement.
#
# This closes that. Two evals on pose dirs ALREADY on the board, so no pose re-extraction and no
# data movement:
#   A  pruned ckpt + 24 fps poses   = the deployable config
#   B  pruned ckpt + source poses   = paired control, so the frame-rate cost is measurable on OUR
#                                     keypoints rather than inherited from the authors' poses
# Against the existing full-checkpoint rows (fps24 20.15, fps30 18.64) these give the pruning cost
# on our own pose extraction, which is the one term nothing has measured.
#
# batch size 1 to match the fps24/fps30 rows it is compared against: 5.1 showed batch size changes
# the output (2 of 30 clips differ between batch 8 and 1), so the comparison has to hold it fixed.
# cap 64 is the deployable setting (L3.2: identical to 100, and it bounds the worst case).
#
# One process at a time, no backgrounding: two PyTorch processes on this board kill each other
# with the CUDACachingAllocator assert (p5_all_evals.sh header).
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=weights/openasl_pose_only_slt_pruned.pth
MT5=weights/mt5-base-openasl-pruned     # pre-pruned dir: load_model takes the fast path (5.4)
LABELS=data/openasl_labels/labels.test
MIN_FREE_MB=${SLT_MIN_FREE_MB:-1800}

run_one() {  # $1 = pose dir, $2 = output name
  local poses="$1" name="$2"
  local out="results/eval_30clip_${name}.json"
  if [ -s "$out" ]; then echo "[p9] have $out"; return 0; fi
  if [ ! -d "$poses" ]; then echo "[p9] MISSING $poses, skipping $name" >&2; return 0; fi
  local users foreign
  users=$(who | wc -l)
  foreign=$(ps -eo user,pcpu,comm --sort=-pcpu | awk 'NR>1 && $1!="tgoyal" && $1!="root" && $2+0>1.0' | wc -l)
  python3 jetson/drop_file_cache.py --target-free-mb=3000 >/dev/null 2>&1 || true
  local free_mb; free_mb=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
  echo "[p9] === $name  (users=$users foreign=$foreign memfree=${free_mb}MB)  $(date -u +%H:%M:%SZ)"
  if [ "$free_mb" -lt "$MIN_FREE_MB" ]; then
    echo "[p9] ABORT $name: MemFree ${free_mb} MB < ${MIN_FREE_MB} MB" >&2; return 1
  fi
  local t0=$SECONDS
  ./jetson/run.sh exec-batch python3 unisign/eval_openasl.py \
      --ckpt "$CKPT" --mt5 "$MT5" --poses "$poses" --labels "$LABELS" \
      --num-beams 4 --max-new-tokens 64 --batch-size 1 \
      --out "$out"
  echo "[p9] done $name in $((SECONDS - t0)) s  (users_after=$(who | wc -l))"
  python3 -c "
import json;d=json.load(open('$out'))
print('[p9] %-26s n=%d  BLEU-4 %.2f  ROUGE-L %.2f' % ('$name', d['n'], d['bleu']['bleu4'], d['rouge_l']))"
}

run_one results/pkl_30clip_rtmw_fp16_sqnorm_fps24 pruned_fps24_bs1
run_one results/pkl_30clip_rtmw_fp16_sqnorm       pruned_src_bs1
echo "[p9] ALL DONE"
