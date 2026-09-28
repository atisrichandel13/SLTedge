#!/usr/bin/env bash
# Waits for jetson/p8_l7_sweep.sh, then writes the three summary tables and the accuracy-vs-rate CIs.
# Needs no Mac. Launch detached:
#
#     setsid nohup ./jetson/p8_after.sh > results/logs/p8_after.log 2>&1 < /dev/null &
#
# Deliberately calls python by FILE PATH, never `python3 - <<HEREDOC`: `run.sh exec-batch` does not
# forward stdin, so a heredoc reaches the container as EOF and produces a silently empty report. That
# is exactly how results/logs/final_30clip_summary.txt came out blank on 2026-09-26.
set -uo pipefail
cd "$(dirname "$0")/.."

SUM=results/logs/p8_summary.txt
R=(jetson/run.sh exec-batch)
waited=0
while pgrep -f p8_l7_sweep.sh > /dev/null; do
    [ "$waited" -gt 5400 ] && { echo "[after] giving up waiting after 90 min"; break; }
    sleep 30; waited=$((waited + 30))
done
echo "[after] sweep no longer running; building tables $(date)"

{
    echo "generated $(date)  on $(hostname), nvpmodel $(cat /var/lib/nvpmodel/status 2>/dev/null)"
    echo
    python3 tools/p8_tables.py
    echo "=============================================================================="
    echo " Accuracy vs rate: paired bootstrap against 30 fps (2000 resamples, by clip name)"
    echo "=============================================================================="
    for fps in 24 16 12 8; do
        a=results/eval_30clip_fps30.json
        b=results/eval_30clip_fps${fps}.json
        if [ ! -f "$a" ] || [ ! -f "$b" ]; then echo "  30 -> ${fps} fps: MISSING"; continue; fi
        echo "--- 30 fps  ->  ${fps} fps"
        "${R[@]}" python3 -m unisign.bootstrap_ci "$a" "$b" -n 2000 \
            --out "results/ci_fps30_vs_fps${fps}.json" 2>&1 \
            | grep -E "bleu4|rouge_l|WARNING|paired"
    done
    echo
    echo "A CI that includes 0 means no measured accuracy cost at that rate -- report it that way,"
    echo "not as a small loss. Pair each row with the J/s-video column in table A."
} > "$SUM" 2>&1

cat "$SUM"
echo "[after] wrote $SUM  $(date +%H:%M:%S)"
