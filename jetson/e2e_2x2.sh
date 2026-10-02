#!/usr/bin/env bash
# C9/M1b: end-to-end 2x2 over {source rate, 16 fps} x {beam 4, greedy} on ONE clip.
#
# Why a 2x2 and not just the one corner the LM track asked for. The composition
# pose_J_per_s * seconds + LM_J runs ~6 % low against M1 (source rate, beam 4). Running only
# 16 fps/greedy gives a second point but cannot say WHICH factor the composition mismodels.
# The 2x2 shares one clip, so frame count and decoder width are varied independently and the
# residual is attributable. M1's own cell is repeated as a control for run-to-run drift.
#
# Resumable: a config whose summary json already exists is skipped, so a dropped tunnel costs
# at most the config in flight.
set -u
cd "$(dirname "$0")/.."
CLIP="${CLIP:-ixq65EiuJ_c-00:03:47.633-00:03:56.133}"
ENG=models/rtmw/rtmw-l-m_256x192_fp16.engine
OUT=results/e2e_2x2
mkdir -p "$OUT" results/logs

echo "[2x2] clip=$CLIP"
for rate in src 16; do
  for beams in 4 1; do
    tag="${rate}_beam${beams}"
    sj="$OUT/e2e_${tag}.json"
    if [ -s "$sj" ]; then echo "[2x2] $tag already done, skipping"; continue; fi
    # occupancy is part of the record: an unrecorded co-tenant is what invalidated the
    # 09-29 memory probes (RESULTS.md 5.2)
    users=$(who | wc -l)
    foreign=$(ps -eo user,pcpu,comm --sort=-pcpu | awk 'NR>1 && $1!="tgoyal" && $1!="root" && $2+0>1.0' | wc -l)
    free=$(awk '/MemFree/{printf "%d", $2/1024}' /proc/meminfo)
    echo "[2x2] === $tag  (users=$users foreign=$foreign memfree=${free}MB)"
    fpsarg=""
    [ "$rate" != "src" ] && fpsarg="--keep-fps $rate"
    ./jetson/run.sh exec-batch python3 -m unisign.e2e_translate \
        --engine "$ENG" \
        --frames "data/clips/$CLIP/frames" \
        --meta   "data/clips/$CLIP/meta.json" \
        --ckpt weights/openasl_pose_only_slt.pth --mt5 weights/mt5-base \
        --num-beams "$beams" $fpsarg --repeat 3 \
        --out "$sj" \
        --power-json "$OUT/power_${tag}.json" \
        > "results/logs/e2e_2x2_${tag}.log" 2>&1
    rc=$?
    echo "[2x2] $tag exit=$rc  (users_after=$(who | wc -l))"
    [ $rc -ne 0 ] && tail -5 "results/logs/e2e_2x2_${tag}.log"
    sleep 5
  done
done
echo "[2x2] ALL DONE"
ls -la "$OUT"
