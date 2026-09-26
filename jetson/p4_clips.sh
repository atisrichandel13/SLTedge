#!/usr/bin/env bash
# Every OpenASL clip through every pose engine. Feeds P4 (keypoint agreement) and P5 (pose -> BLEU).
#
# Runs ON the Jetson, from ~/sign-lang-project:
#     ./jetson/p4_clips.sh                                  # the four deployable configs
#     ./jetson/p4_clips.sh rtmw_fp32 rtmw_fp16              # a subset
#
# Writes one dump per (config, clip) to results/kpts/<config>__<vid>.json, which is what
# task1_rtmpose/07_kpt_agreement.py pairs on. Already-finished dumps are skipped, so the script
# is safe to re-run after an interruption.
#
# Engines are rebuilt from the staged ONNX because the board keeps none between sessions (shared
# scratch). When the dumps are on the Mac (jetson/pull_results.sh), reclaim the space with
# jetson/run.sh clean-large.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=results/kpts
mkdir -p "$OUT"

# The persistent container, non-tty so this works over ssh.
R=(jetson/run.sh exec-batch)

# nvmap allocates only from MemFree and will not reclaim page cache, and we have no root: evict our
# own cached files before each build and each engine load or a 2 GB build can fail on an 8 GB board.
drop_cache() { python3 jetson/drop_file_cache.py >/dev/null 2>&1 || true; }

select_config() {
    case "$1" in
        rtmw_fp32)          ONNX=models/rtmw/rtmw-l-m_256x192.onnx; TAG=fp32;      FLAGS=() ;;
        rtmw_fp16)          ONNX=models/rtmw/rtmw-l-m_256x192.onnx; TAG=fp16;      FLAGS=(--fp16) ;;
        rtmposex_fp32)      ONNX=models/rtmpose-x.onnx;             TAG=fp32;      FLAGS=() ;;
        # plain FP16 overflows the head's ScaleNorm ReduceSum (peak 142292 > 65504); pinning the
        # seven mlp.0 layers to FP32 repairs it for under 3% latency. See RESULTS.md, P2.
        rtmposex_fp16mixed) ONNX=models/rtmpose-x.onnx;             TAG=fp16mixed; FLAGS=(--fp16 --fp32-layers 'mlp\.0') ;;
        rtmposex_fp16)      ONNX=models/rtmpose-x.onnx;             TAG=fp16;      FLAGS=(--fp16) ;;
        *) echo "unknown config: $1" >&2; return 1 ;;
    esac
    ENGINE="${ONNX%.onnx}_${TAG}.engine"
}

CONFIGS=("$@")
[ "${#CONFIGS[@]}" -eq 0 ] && CONFIGS=(rtmw_fp32 rtmw_fp16 rtmposex_fp32 rtmposex_fp16mixed)

CLIPS=(data/clips/*/frames)
[ -d "${CLIPS[0]}" ] || { echo "no frames under data/clips/*/frames -- push them from the Mac first" >&2; exit 1; }
# nvpmodel -q wants root; the mode is readable here (0 = 15 W, the mode the boards are pinned to).
echo "[p4] power mode: $(cat /var/lib/nvpmodel/status 2>/dev/null || echo unknown)"
echo "[p4] ${#CONFIGS[@]} config(s) x ${#CLIPS[@]} clip(s)"

for cfg in "${CONFIGS[@]}"; do
    select_config "$cfg"
    if [ ! -f "$ENGINE" ]; then
        drop_cache
        echo "[p4] building $ENGINE"
        t0=$SECONDS
        "${R[@]}" python3 task1_rtmpose/02_build_engine.py --onnx "$ONNX" --tag "$TAG" \
            --workspace-gb 1 "${FLAGS[@]}"
        echo "[p4] built in $((SECONDS - t0)) s"
    fi
    for frames in "${CLIPS[@]}"; do
        vid=$(basename "$(dirname "$frames")")
        out="$OUT/${cfg}__${vid}.json"
        if [ -f "$out" ]; then echo "[p4] skip $out"; continue; fi
        drop_cache
        t0=$SECONDS
        "${R[@]}" python3 task1_rtmpose/03_infer_frames.py --engine "$ENGINE" \
            --frames "$frames" --out "$out"
        echo "[p4] $cfg  $vid  $((SECONDS - t0)) s"
    done
done

echo "[p4] done. On the Mac: jetson/pull_results.sh, then"
echo "     python task1_rtmpose/07_kpt_agreement.py --ref results/kpts/rtmw_fp32__*.json \\"
echo "        --test results/kpts/rtmw_fp16__*.json --out results/kpt_agreement_rtmw_fp16_5signers.json"
