#!/usr/bin/env bash
# The decisive experiment for the frontier's energy model (RESULTS.md 5.3).
#
# frontier.py composes LM_J from RESULTS.md 2.9C, which was measured on the PRUNED checkpoint
# (vocab 26,078). Every end-to-end run so far -- M1 and the whole 2x2 grid -- loads the FULL
# released checkpoint (vocab 250,112). The output projection scales with beams x vocab, which is
# why the composition understates the greedy->beam4 energy step by 3-5x. If that diagnosis is
# right, the same grid on the pruned checkpoint should match the composition.
#
# Also answers two side questions in the same runs:
#   * per-stage energy, the LM track's ASK-E2E-KNEE ask. --power-interval-ms 50 (not the 100
#     default) because the LM stage is only ~1.4 s and a stage window loses up to one sample
#     interval at each edge; stage_power reports n_samples_per_window so this stays auditable.
#   * whether the pruned model clears the source x beam 4 working-set limit that the full
#     checkpoint hits (3 failures, incl. 2 under expandable_segments).
#
# Vocabulary vintage: these weights are the Sep-17 26,078-id keep set, superseded for ACCURACY by
# the leak-free 26,025 set (L5.4). That does not matter here -- energy depends on the vocabulary
# SIZE, and the two sets differ by 53 rows, so this is apples-to-apples against 2.9C.
set -u
cd "$(dirname "$0")/.."
CLIP="${CLIP:-ixq65EiuJ_c-00:03:47.633-00:03:56.133}"
ENG=models/rtmw/rtmw-l-m_256x192_fp16.engine
CKPT=weights/openasl_pose_only_slt_pruned.pth
MT5=weights/mt5-base-openasl-pruned      # pre-pruned dir: load_model takes the fast path.
                                         # Pairing the pruned ckpt with the FULL mt5 dir instead
                                         # peaks ~4.2 GB, worse than the full checkpoint (model.py).
OUT=results/e2e_pruned
mkdir -p "$OUT" results/logs

echo "[pruned] clip=$CLIP ckpt=$CKPT mt5=$MT5"
for spec in "24 4" "24 1" "src 4" "src 1"; do
  set -- $spec; rate=$1; beams=$2
  tag="${rate}_beam${beams}"
  sj="$OUT/e2e_${tag}.json"
  if [ -s "$sj" ]; then echo "[pruned] $tag already done, skipping"; continue; fi
  users=$(who | wc -l)
  foreign=$(ps -eo user,pcpu,comm --sort=-pcpu | awk 'NR>1 && $1!="tgoyal" && $1!="root" && $2+0>1.0' | wc -l)
  echo "[pruned] === $tag  (users=$users foreign=$foreign memfree=$(awk '/MemFree/{printf "%d", $2/1024}' /proc/meminfo)MB before reclaim)"
  # The pruned path needs ~1.5 GB, not the >5.3 GB the full checkpoint needs (5.3), so the target
  # is far less critical here -- kept anyway so the condition is recorded rather than assumed.
  python3 jetson/drop_file_cache.py --target-free-mb=4000 2>&1 | tail -1
  free=$(awk '/MemFree/{printf "%d", $2/1024}' /proc/meminfo)
  echo "[pruned] $tag memfree=${free}MB after reclaim"
  if [ "$free" -lt 2000 ]; then
    echo "[pruned] $tag SKIPPED: only ${free}MB free, the pruned path needs ~1.5 GB"
    continue
  fi
  fpsarg=""
  [ "$rate" != "src" ] && fpsarg="--keep-fps $rate"
  ./jetson/run.sh exec-batch python3 -m unisign.e2e_translate \
      --engine "$ENG" \
      --frames "data/clips/$CLIP/frames" \
      --meta   "data/clips/$CLIP/meta.json" \
      --ckpt "$CKPT" --mt5 "$MT5" \
      --num-beams "$beams" $fpsarg --repeat 3 \
      --power-interval-ms 50 \
      --out "$sj" \
      --power-json "$OUT/power_${tag}.json" \
      > "results/logs/e2e_pruned_${tag}.log" 2>&1
  rc=$?
  echo "[pruned] $tag exit=$rc  (users_after=$(who | wc -l))"
  if [ $rc -eq 0 ]; then
    grep -E "^\[stage\]" "results/logs/e2e_pruned_${tag}.log" || true
  else
    tail -5 "results/logs/e2e_pruned_${tag}.log"
  fi
  sleep 5
done
echo "[pruned] ALL DONE"
