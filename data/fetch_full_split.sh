#!/usr/bin/env bash
# Phase A stage 1: fetch the whole OpenASL test split to the Mac. Safe to re-run at any time.
#
#     ./data/fetch_full_split.sh              # fetch / resume
#     ./data/fetch_full_split.sh status       # how far along
#
# RESUMABILITY. The state is the set of clip directories under data/clips: openasl_fetch.py
# --skip-existing reuses any clip whose frame count matches its meta.json and re-fetches anything
# truncated, and index.json is rebuilt by merging the metas on disk. So a dropped network, a closed
# laptop or a killed process costs at most the clip in flight. Just run it again.
#
# 974 clips over 455 videos, ~124 min of video, ~223k frames, ~7.6 GB. Expect roughly 2-3 h on a warm
# connection at the 8 s/clip we measured, plus retries: ~11 % of videos come back private.
set -uo pipefail
cd "$(dirname "$0")/.."

YT="${SLT_YTENV:-/private/tmp/claude-501/-Users-tushar-Documents-slt-SLTedge/b6ac1c6b-bb75-49c8-8921-249bf965866f/scratchpad/ytenv/bin}"
OUT="${SLT_OUT:-data/clips}"
LOG="${SLT_LOG:-results/logs/fetch_full_split.log}"
WANT=974

if [ "${1:-}" = "status" ]; then
    have=$(find "$OUT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' ')
    frames=$(find "$OUT" -name '*.jpg' | wc -l | tr -d ' ')
    echo "clips on disk: ${have}/${WANT}   frames: ${frames}   size: $(du -sh "$OUT" | cut -f1)"
    [ -f "$LOG" ] && { echo "--- last log lines:"; tail -4 "$LOG"; }
    exit 0
fi

if [ ! -x "$YT/python" ]; then
    echo "no yt-dlp env at $YT" >&2
    echo "set SLT_YTENV to a python env with a current yt-dlp (a 6-month-old one fails with HTTP 403)" >&2
    exit 1
fi

mkdir -p "$(dirname "$LOG")"

count_clips() { find "$OUT" -mindepth 1 -maxdepth 1 -type d | wc -l | tr -d ' '; }

# SUPERVISOR with a stall watchdog. Resumable is not the same as hang-proof: on 2026-09-28 a dropped
# network left yt-dlp blocked on a socket and the fetch sat at 397 clips for six hours without dying,
# so the resume logic never got a chance to run. The download path is fixed (process-group kill plus
# --socket-timeout), but a supervisor is the belt to that braces: if the clip count stops advancing for
# STALL_S, kill the whole thing and start again from disk state.
ATTEMPTS="${SLT_ATTEMPTS:-40}"
STALL_S="${SLT_STALL_S:-600}"

for attempt in $(seq 1 "$ATTEMPTS"); do
    have=$(count_clips)
    if [ "$have" -ge "$WANT" ]; then echo "[fetch] complete: ${have}/${WANT}"; break; fi
    echo "=== $(date) attempt ${attempt}, ${have}/${WANT} on disk" >> "$LOG"
    PATH="$YT:$PATH" "$YT/python" data/openasl_fetch.py \
        --split test --n-clips "$WANT" --max-per-video 0 \
        --min-dur 0 --max-dur 1e9 --seed 0 --max-attempts 1200 \
        --out "$OUT" >> "$LOG" 2>&1 &
    pid=$!

    last=$(count_clips); idle=0
    while kill -0 "$pid" 2>/dev/null; do
        sleep 30
        now=$(count_clips)
        if [ "$now" -gt "$last" ]; then last=$now; idle=0; else idle=$((idle + 30)); fi
        if [ "$idle" -ge "$STALL_S" ]; then
            echo "=== $(date) STALLED at ${now} clips for ${idle}s; killing attempt ${attempt}" >> "$LOG"
            # kill the process group so yt-dlp's ffmpeg children go too
            kill -- "-$(ps -o pgid= "$pid" | tr -d ' ')" 2>/dev/null || kill -9 "$pid" 2>/dev/null
            pkill -9 -f 'yt-dlp' 2>/dev/null; pkill -9 -f 'ffmpeg' 2>/dev/null
            break
        fi
    done
    wait "$pid" 2>/dev/null
    echo "=== $(date) attempt ${attempt} ended, $(count_clips)/${WANT} on disk" >> "$LOG"
    [ "$(count_clips)" -ge "$WANT" ] && break
    sleep 10
done

./data/fetch_full_split.sh status
