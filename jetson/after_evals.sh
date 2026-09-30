#!/usr/bin/env bash
# Runs AFTER jetson/p5_all_evals.sh and needs no Mac: waits for it, computes every paired bootstrap
# CI, and writes one readable summary. Launch detached (setsid nohup ... < /dev/null &) so it survives
# the ssh session closing.
#
#     setsid nohup ./jetson/after_evals.sh > results/logs/after_evals.log 2>&1 < /dev/null &
#
# Waits on the "ALL DONE" marker in p5_all.log, not on a pgrep of itself (a pgrep guard is what made
# two evals run concurrently and kill each other earlier). If the eval run dies without the marker,
# this proceeds with whatever evals exist and says so, rather than spinning forever.
set -uo pipefail
cd "$(dirname "$0")/.."

LOG=results/logs/p5_all.log
SUM=results/logs/final_30clip_summary.txt
R=(jetson/run.sh exec-batch)
NBOOT="${SLT_NBOOT:-4000}"

waited=0
while true; do
    grep -q "ALL DONE" "$LOG" 2>/dev/null && { echo "[after] evals finished"; break; }
    if ! pgrep -f p5_all_evals.sh > /dev/null; then
        echo "[after] WARNING: p5_all_evals.sh is gone without an ALL DONE marker."
        echo "[after] Proceeding with whatever evals exist -- the table below may be incomplete."
        break
    fi
    [ "$waited" -gt 5400 ] && { echo "[after] giving up waiting after 90 min"; break; }
    sleep 30; waited=$((waited + 30))
done

{
  echo "======================================================================"
  echo " 30-clip pose -> BLEU, released checkpoint, beam 4, jetson-lpcv-03 15 W"
  echo " generated $(date)"
  echo "======================================================================"
  echo
  "${R[@]}" python3 - <<'PY'
import json, glob, os
order = ['authors_ceiling','rtmw_fp32','rtmw_fp32_sqnorm','rtmw_fp16','rtmw_fp16_sqnorm',
         'rtmposex_fp32','rtmposex_fp16mixed']
rows = []
for f in sorted(glob.glob('results/eval_30clip_*.json')):
    d = json.load(open(f))
    rows.append((os.path.basename(f)[12:-5], d['n'], d['bleu']['bleu4'], d['rouge_l']))
rows.sort(key=lambda r: order.index(r[0]) if r[0] in order else 99)
print(f"{'config':26s} {'n':>3s} {'BLEU-4':>7s} {'ROUGE-L':>8s}")
for n, c, b, r in rows:
    print(f"{n:26s} {c:3d} {b:7.2f} {r:8.2f}")
print()
print("sqnorm = our own keypoints re-expressed in the authors' square normalisation frame")
print("         (common/renorm_to_openasl.py); extractor, precision and frames held fixed.")
PY
  echo
  echo "---------------- paired bootstrap, ${NBOOT} resamples, aligned by clip name -------------"
  for pair in "rtmw_fp32 authors_ceiling" "rtmw_fp32 rtmw_fp16" "rtmw_fp32 rtmposex_fp32" \
              "rtmw_fp32 rtmw_fp32_sqnorm" "rtmw_fp32_sqnorm authors_ceiling" \
              "rtmw_fp16 rtmw_fp16_sqnorm" "rtmposex_fp32 rtmposex_fp16mixed"; do
      set -- $pair
      a="results/eval_30clip_$1.json"; b="results/eval_30clip_$2.json"
      if [ ! -f "$a" ] || [ ! -f "$b" ]; then echo; echo "--- $1 -> $2 : MISSING, skipped"; continue; fi
      echo; echo "--- $1  ->  $2"
      "${R[@]}" python3 -m unisign.bootstrap_ci "$a" "$b" -n "$NBOOT" \
          --out "results/ci_30clip_$1__$2.json" 2>&1 | grep -E "bleu4|rouge_l|WARNING|clips"
  done
  echo
  echo "Reading it: 'sign established' means the 95 % CI excludes 0. 'NOT ESTABLISHED' means the"
  echo "direction of the difference is not supported at n=30 -- report it as no measured difference,"
  echo "not as a small one."
} | tee "$SUM"

echo "[after] wrote $SUM  $(date +%H:%M:%S)"
