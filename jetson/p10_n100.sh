#!/usr/bin/env bash
# n=100 board pass: pin down the accuracy of the DEPLOYABLE config.
#
# Why. Every accuracy number for the shipped system is currently COMPOSED from two experiments:
# the decode config on the authors' keypoints at n=976 (22.80 BLEU-4 / 43.13 ROUGE-L), and the
# substitution of our own board pose extraction for theirs at n=30 (-0.35 BLEU-4, CI [-4.23, +3.17]).
# The point estimate is reassuring and the interval is +-4 BLEU, entirely because of that n=30. This
# run takes the same comparison to n=100 from 100 DISTINCT videos, which should roughly halve it.
#
# Design: every row below is scored on exactly the same 100 clips, with the authors' reference poses
# restricted to those same 100 (data/openasl_pose_n100). That is deliberate -- RESULTS.md 2.5c
# records quoting a ceiling at n=40 against our rows at n=30, and this run must not repeat it.
#
# Decode settings are the deployable ones throughout: beam 4, cap 64, batch size 1. Batch size is
# held at 1 because 5.1 found 2 of 30 clips change output between batch 8 and batch 1, so it has to
# be fixed across any comparison. --fps 24 is per-clip correct: fps_ratio_for_clip derives each
# clip's own source rate from its duration, which is the hardcoded-29.97 bug this project already hit.
#
# Resumable: extraction skips clips whose pkl exists, each eval skips when its output exists.
# One process at a time -- two PyTorch processes on this board kill each other (p5_all_evals.sh).
set -uo pipefail
cd "$(dirname "$0")/.."

TAG=n100
ENG=models/rtmw/rtmw-l-m_256x192_fp16.engine
PRUNED_CKPT=weights/openasl_pose_only_slt_pruned.pth
PRUNED_MT5=weights/mt5-base-openasl-pruned
FULL_CKPT=weights/openasl_pose_only_slt.pth
FULL_MT5=weights/mt5-base
LABELS=data/openasl_labels/labels.test
REF=data/openasl_pose_n100
OURS=results/pkl_${TAG}_rtmw_fp16
R=(jetson/run.sh exec-batch)

prep_mem() {
  local want="${1:-3000}" f
  python3 jetson/drop_file_cache.py --target-free-mb="$want" >/dev/null 2>&1 || true
  f=$(awk '/MemFree/{print int($2/1024)}' /proc/meminfo)
  echo "[p10]   MemFree ${f} MB (want ${want})"
  if [ "$f" -lt 2000 ]; then echo "[p10] ABORT: MemFree ${f} MB too low" >&2; return 1; fi
  return 0
}

n_clips=$(find data/clips -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
n_ref=$(ls "$REF"/*.pkl 2>/dev/null | wc -l | tr -d ' ')
echo "[p10] start $(date -u +%FT%TZ)  clips=${n_clips}  ref_poses=${n_ref}"
echo "[p10] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"
who | sed 's/^/[p10]   /'; df -h ~ | tail -1 | sed 's/^/[p10]   /'

# ---- stage 1: our keypoints -> pkl (engine stays loaded across clips)
have=$(ls "$OURS"/*.pkl 2>/dev/null | wc -l | tr -d ' ')
if [ "$have" -ge "$n_clips" ]; then
  echo "[p10] have all ${have} pkls"
else
  prep_mem 3500 || exit 1
  echo "[p10] STAGE1 extract ($have/$n_clips done)  $(date -u +%TZ)"
  # one pass writes both normalisations; same keypoints, two exact normalisations, no extra GPU work
  "${R[@]}" python3 task1_rtmpose/09_batch_clips.py --engine "$ENG" --clips-dir data/clips \
      --config rtmw_fp16 --pkl-out "$OURS" --pkl-out-raw "results/pkl_${TAG}_rtmw_fp16_raw" \
      --square-norm --progress-every 25 2>&1 | tail -4 | sed 's/^/[p10]   /'
fi

# ---- stage 2: the evals.  name | poses | ckpt | fps
eval_one() {  # $1 name  $2 poses  $3 ckpt  $4 mt5  $5 fps('' = source)
  local name="$1" poses="$2" ckpt="$3" mt5="$4" fps="$5"
  local out="results/eval_${TAG}_${name}.json"
  [ -s "$out" ] && { echo "[p10] have $out"; return 0; }
  [ -d "$poses" ] || { echo "[p10] SKIP $name: no $poses" >&2; return 0; }
  local fpsarg=()
  [ -n "$fps" ] && fpsarg=(--fps "$fps")
  prep_mem 3000 || return 1
  echo "[p10] === $name  $(date -u +%TZ)"
  local t0=$SECONDS
  "${R[@]}" python3 unisign/eval_openasl.py --ckpt "$ckpt" --mt5 "$mt5" --poses "$poses" \
      --labels "$LABELS" --num-beams 4 --max-new-tokens 64 --batch-size 1 "${fpsarg[@]}" \
      --out "$out" > "results/logs/eval_${TAG}_${name}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then echo "[p10] FAILED $name rc=$rc"; tail -4 "results/logs/eval_${TAG}_${name}.log"; return 0; fi
  echo "[p10] done $name in $((SECONDS - t0)) s"
  python3 -c "
import json;d=json.load(open('$out'))
print('[p10] %-24s n=%-4d missing=%-4d BLEU-4 %6.2f  ROUGE-L %6.2f' % ('$name', d['n'], d['missing'], d['bleu']['bleu4'], d['rouge_l']))"
}

# the deployable config, and the three rows it has to be read against
eval_one pruned_ours_fps24   "$OURS" "$PRUNED_CKPT" "$PRUNED_MT5" 24
eval_one pruned_ceil_fps24   "$REF"  "$PRUNED_CKPT" "$PRUNED_MT5" 24
eval_one pruned_ours_src     "$OURS" "$PRUNED_CKPT" "$PRUNED_MT5" ""
eval_one pruned_ceil_src     "$REF"  "$PRUNED_CKPT" "$PRUNED_MT5" ""
# pruning cost on OUR keypoints at the deployable rate
eval_one full_ours_fps24     "$OURS" "$FULL_CKPT"   "$FULL_MT5"   24

echo "[p10] ALL DONE $(date -u +%FT%TZ)"
