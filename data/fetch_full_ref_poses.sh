#!/usr/bin/env bash
# Phase A stage 1b: the authors' released poses for the WHOLE test split, for the ceiling row.
#
#     ./data/fetch_full_ref_poses.sh          # fetch / resume
#     ./data/fetch_full_ref_poses.sh status
#
# Why it must be complete before the ceiling eval runs: eval_openasl.py scores every clip that has a
# pkl, so a partial pose directory produces a ceiling on a different clip set than our rows. That is
# exactly the mistake already made once (ceiling at n=40 against our n=30, RESULTS.md 2.5c), and the
# point estimates are not comparable when it happens even though the paired CIs still are.
#
# Resumable: openasl_pose_fetch.py skips any member whose destination file already exists, and it reads
# the 32 GB split archive over HTTP range requests rather than downloading it.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${SLT_REF_POSE:-data/openasl_pose}"
LOG="${SLT_LOG:-results/logs/fetch_ref_poses.log}"
PY="${SLT_PY:-$HOME/miniconda3/bin/python}"
WANT=976

if [ "${1:-}" = "status" ]; then
    have=$(ls "$OUT"/*.pkl 2>/dev/null | wc -l | tr -d ' ')
    echo "reference poses: ${have}/${WANT}   size: $(du -sh "$OUT" 2>/dev/null | cut -f1)"
    [ -f "$LOG" ] && { echo "--- last log lines:"; tail -3 "$LOG"; }
    exit 0
fi

mkdir -p "$(dirname "$LOG")" "$OUT"
echo "=== $(date) starting/resuming reference-pose fetch" >> "$LOG"
"$PY" data/openasl_pose_fetch.py --split test --labels-dir data/openasl_labels --out "$OUT" \
    >> "$LOG" 2>&1
rc=$?
echo "=== $(date) exited rc=$rc" >> "$LOG"
./data/fetch_full_ref_poses.sh status
exit $rc
