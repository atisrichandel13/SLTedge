#!/usr/bin/env bash
# J12: take results/pkl_split_rtmw_fp16{,_raw} from 400 clips to all 931 fetched test clips.
#
#     SLT_JSSH=<path>/jssh.exp SLT_JRSYNC=<path>/jrsync.exp ./jetson/j12_rest_of_split.sh
#     ./jetson/j12_rest_of_split.sh status
#
# AND IT DRIVES J9 STEP 1 TOO (the dev extraction), which is why it is parameterised:
#
#     SLT_CLIPS=data/clips_dev SLT_PKL=results/pkl_dev_rtmw_fp16 SLT_SKIP_REF=1 \
#       SLT_LOG=results/logs/j9_dev_poses.log ./jetson/j12_rest_of_split.sh
#
# The clip list is just the directories under SLT_CLIPS, and the same relative path is used on both
# sides of the rsync, so nothing else needs to change. Extraction is always at the clip's NATIVE
# rate -- pose pkls are rate-independent and the thinning happens at load (fps_ratio_for_clip) --
# so there is no rate flag here and the dev pkls are usable at any target fps.
#
# Runs on the POSE-TRACK MAC and drives the board over the expect wrappers, whose paths come from the
# environment so the Jetson password never enters the tracked tree.
#
# WHY 931 AND NOT 976. eval_n400_pruned_ours_fps24.json records missing=576 against all 976
# labels.test names, but only 931 are on disk. CORRECTED 2026-10-05 (RESULTS.md L20): 43 are dead
# links and 2 are OURS, both clips of Ads-4j06eJY, removed by a stale --exclude-yid default at
# data/openasl_fetch.py:245 that this script never passed and could not have cleared. The true ceiling
# is 933; the difference is ~0.001 on the half-width. And do NOT read `requested - n_ok` as proof the
# fetch exhausted its candidates: `requested` is --n-clips echoed back, so the identity holds by
# construction. 931 is the paired set we scored, so J12 was 931-400 = 531 clips.
# The interval the LM track wants narrows by sqrt(400/931) = 0.6555, to a half-width of +-0.790
# (REPLY-J12-ACCEPTED-2026-10-05.md), not the +-0.771 that sqrt(400/976) gives.
#
# WHY IT IS BATCHED. 531 clips of JPEGs is 3.66 GB and the board is shared scratch sitting at 86 %
# full. Each batch is pushed, extracted, verified and then has its frames deleted, so the board never
# holds more than one batch of frames. The pkls are the artifact; the frames are recoverable from the
# pose-track Mac at any time.
#
# RESUMABILITY. The board's pkl directory is the state, read fresh at the start of every run, so a
# dropped WireGuard link or a killed process costs at most the batch in flight. 09_batch_clips.py
# skips any clip that already has both pkls and writes .part-then-rename, so no pkl is ever half
# written. Re-running is always safe.
#
# ONE PROCESS AT A TIME: two PyTorch/TensorRT processes on this board kill each other. This script
# never runs two, and nothing else must be running on the board while it is.
set -uo pipefail
cd "$(dirname "$0")/.."

JSSH="${SLT_JSSH:?set SLT_JSSH to the expect ssh wrapper}"
JRSYNC="${SLT_JRSYNC:?set SLT_JRSYNC to the expect rsync wrapper}"
RHOST="${SLT_RHOST:-tgoyal@192.168.1.73}"
RPATH="${SLT_RPATH:-sign-lang-project}"
CLIPS="${SLT_CLIPS:-data/clips}"
REFSRC="${SLT_REFSRC:-data/openasl_pose}"
REFDST="${SLT_REFDST:-data/openasl_pose_split}"
# The reference push exists so the CEILING row is scored on our exact clip set. The dev split has no
# ceiling row -- it produces the adapted model, it is never scored against the authors' poses -- and
# data/openasl_pose holds test clips only, so for J9 step 1 this step is not merely unnecessary, it
# would look for pkls that cannot exist. SLT_SKIP_REF=1 turns it off.
SKIP_REF="${SLT_SKIP_REF:-0}"
PKL="${SLT_PKL:-results/pkl_split_rtmw_fp16}"
NBATCH="${SLT_BATCH_CLIPS:-90}"
ENG="${SLT_ENG:-models/rtmw/rtmw-l-m_256x192_fp16.engine}"
WORK="${SLT_WORK:-${TMPDIR:-/tmp}}/j12"
mkdir -p "$WORK"

# OWN THE LOG. Fixed 2026-10-05, after this script extracted 528 clips and recorded none of it.
# It was launched with its stdout through a pipeline, the reader exited two minutes in, and from then
# on every echo failed with EPIPE. `set -e` is deliberately off here (a batch shortfall must not kill
# the run), so those failures were silent: the log froze at 15:15 on batch 1 while the board went from
# 403 to 931 pkls over the next several hours. The end state was verifiable afterwards -- 931/931/931
# with no .part files and no frame dirs, and the reference push only runs after the batch loop's
# `done`, so the zero-gain guard proves every batch passed -- but the per-batch record was gone.
# A driver that runs unattended for hours must not depend on how it was invoked.
LOG="${SLT_LOG:-results/logs/j12.log}"
if [ "$LOG" != "-" ]; then
    mkdir -p "$(dirname "$LOG")"
    echo "[j12] logging to $LOG" >&2
    exec >> "$LOG" 2>&1
fi

# Process substitution through the expect wrapper is fragile quoting, so each directory is listed
# plainly and the intersection is taken here. A clip counts as done only when BOTH normalisations
# exist, matching 09_batch_clips.py's own complete() check.
board_ls() { "$JSSH" "cd $RPATH && ls $1/ 2>/dev/null" 2>/dev/null \
    | sed 's/\r$//' | grep -v '^spawn \|password:' | grep '\.pkl$' | sed 's/\.pkl$//'; }

# A plain `wc -l` through expect returns the spawn echo and the password prompt too, and digits from
# the host address then contaminate `tr -dc 0-9`: the first run read the board's 400 pkls as
# 0319216817316203400 and would have aborted on its own guard. Emit a sentinel and parse that.
board_count() { "$JSSH" "cd $RPATH && echo J12COUNT=\$(ls $1/*.pkl 2>/dev/null | wc -l)" 2>/dev/null \
    | sed 's/\r$//' | sed -n 's/.*J12COUNT=\([0-9][0-9]*\).*/\1/p' | tail -1; }

board_done() {
    comm -12 <(board_ls "$PKL" | sort) <(board_ls "${PKL}_raw" | sort)
}

local_clips() { find "$CLIPS" -mindepth 1 -maxdepth 1 -type d -exec basename {} \; | sort; }

if [ "${1:-}" = "status" ]; then
    board_done > "$WORK/done.txt"
    echo "clips on the pose-track Mac: $(local_clips | wc -l | tr -d ' ')"
    echo "pkl pairs on the board:      $(wc -l < "$WORK/done.txt" | tr -d ' ')"
    echo "still to extract:            $(comm -23 <(local_clips) <(sort "$WORK/done.txt") | wc -l | tr -d ' ')"
    exit 0
fi

echo "[j12] start $(date -u +%FT%TZ)"
board_done | sort > "$WORK/done.txt"
local_clips > "$WORK/all.txt"
comm -23 "$WORK/all.txt" "$WORK/done.txt" > "$WORK/todo.txt"
NTODO=$(wc -l < "$WORK/todo.txt" | tr -d ' ')
echo "[j12] ${NTODO} clips to extract; $(wc -l < "$WORK/done.txt" | tr -d ' ') pkl pairs already on the board"
[ "$NTODO" -eq 0 ] && { echo "[j12] nothing to do"; exit 0; }

split -l "$NBATCH" "$WORK/todo.txt" "$WORK/batch."
NB=$(ls "$WORK"/batch.* | wc -l | tr -d ' ')
echo "[j12] ${NB} batches of up to ${NBATCH} clips"

b=0
for f in "$WORK"/batch.*; do
    b=$((b + 1))
    n=$(wc -l < "$f" | tr -d ' ')
    before=$(board_count "$PKL")
    echo "[j12] === batch ${b}/${NB}: ${n} clips, board has ${before} pkls  $(date -u +%TZ)"

    # --files-from DISABLES recursion, so -r is explicit. Without it the first clip push moved 23 KB.
    # Do NOT filter rsync to a pattern: the first version grepped for "Number of|Total transferred",
    # which --info=stats1 never prints, so a batch that transferred nothing looked identical to one
    # that worked. Keep the last few lines, which carry both the byte counts and any error.
    sed 's#$#/#' "$f" > "$WORK/send.txt"
    "$JRSYNC" -- -rt --files-from="$WORK/send.txt" "$CLIPS/" "$RHOST:$RPATH/$CLIPS/" 2>&1 \
        | grep -v '^spawn \|password:' | tail -3 | sed 's/^/[j12]   rsync: /'

    # Confirm the frames actually landed and are complete before spending GPU time on them.
    staged=$("$JSSH" "cd $RPATH && echo J12COUNT=\$(python3 -c \"
import json,os
n=0
for v in os.listdir('$CLIPS'):
    d=os.path.join('$CLIPS',v,'frames')
    m=os.path.join('$CLIPS',v,'meta.json')
    if not os.path.isdir(d) or not os.path.exists(m): continue
    if len(os.listdir(d))==(json.load(open(m)).get('n_frames') or -1): n+=1
print(n)\")" 2>/dev/null | sed 's/\r$//' | sed -n 's/.*J12COUNT=\([0-9][0-9]*\).*/\1/p' | tail -1)
    echo "[j12]   ${staged:-?} clip(s) staged on the board with a complete frame set"
    if ! [ "${staged:-0}" -ge 1 ] 2>/dev/null; then
        echo "[j12] ABORT: nothing complete staged for batch ${b}; not extracting, not deleting." >&2
        exit 1
    fi

    "$JSSH" "cd $RPATH && python3 jetson/drop_file_cache.py --target-free-mb=3500 2>&1 | tail -1 && \
        awk '/MemFree/{printf \"[j12]   MemFree %d MB\n\", \$2/1024}' /proc/meminfo && \
        jetson/run.sh exec-batch python3 task1_rtmpose/09_batch_clips.py --engine $ENG \
            --clips-dir $CLIPS --config rtmw_fp16 --pkl-out $PKL --pkl-out-raw ${PKL}_raw \
            --square-norm --progress-every 50 2>&1 | tail -3" \
        2>&1 | grep -v --line-buffered '^spawn \|password:' | sed -u 's/^/[j12]   /'

    after=$(board_count "$PKL")
    if ! [ "$before" -ge 0 ] 2>/dev/null || ! [ "$after" -ge 0 ] 2>/dev/null; then
        echo "[j12] ABORT: could not read the board pkl count (before='${before}' after='${after}')." >&2
        exit 1
    fi
    gained=$((after - before))
    echo "[j12]   pkls ${before} -> ${after} (+${gained} of ${n} in this batch)"
    if [ "$gained" -eq 0 ]; then
        echo "[j12] ABORT: batch ${b} produced no pkls. Not deleting frames; fix before re-running." >&2
        exit 1
    fi
    [ "$gained" -lt "$n" ] && echo "[j12]   NOTE: ${gained} of ${n} captured; the shortfall keeps its frames and is retried on the next run"

    # Reclaim the board before the next batch, but ONLY for clips whose pkl pair exists -- a blanket
    # `find -name frames -delete` threw away the frames of clips that had just failed. Clip dirs and
    # their meta.json always stay: 09_batch_clips.py needs the meta, and p10_split.sh counts the dirs.
    "$JSSH" "cd $RPATH && python3 jetson/j12_reclaim.py --clips $CLIPS --pkl $PKL && df -h ~ | tail -1" \
        2>&1 | grep -v --line-buffered '^spawn \|password:' | sed -u 's/^/[j12]   /'
done

# ---- the authors' reference poses for the SAME clip set, or the ceiling row is scored on a different
# one. That is the n=40-vs-n=30 mistake in RESULTS.md 2.5c, and full_split_pass.sh refuses to run
# without them. data/openasl_pose on the pose-track Mac holds all 976, so this is a copy, not a fetch.
if [ "$SKIP_REF" = "1" ]; then
    echo "[j12] SLT_SKIP_REF=1: not pushing reference poses (no ceiling row for this split)"
else
    echo "[j12] pushing reference poses for the extracted set"
    awk '{print $1 ".pkl"}' "$WORK/all.txt" > "$WORK/ref.txt"
    "$JRSYNC" -- -rt --info=stats1 --files-from="$WORK/ref.txt" "$REFSRC/" "$RHOST:$RPATH/$REFDST/" \
        2>&1 | grep -v --line-buffered '^spawn \|password:' | tail -3 | sed -u 's/^/[j12]   ref rsync: /'
fi

REFTALLY=" / ref \$(ls ${REFDST}/*.pkl 2>/dev/null|wc -l)"
[ "$SKIP_REF" = "1" ] && REFTALLY=""
"$JSSH" "cd $RPATH && echo \"[j12] pkls: \$(ls ${PKL}/*.pkl|wc -l) / raw \$(ls ${PKL}_raw/*.pkl|wc -l)${REFTALLY}\" && df -h ~|tail -1" \
    2>&1 | grep -v --line-buffered '^spawn \|password:' | sed -u 's/^/[j12]   /'
echo "[j12] EXTRACTION DONE $(date -u +%FT%TZ)"
echo "[j12] next: the evals, which need a NEW tag or eval_one skips them --"
echo "[j12]   SLT_TAG=n931 SLT_EXPECT_N=931 jetson/p10_split.sh"
