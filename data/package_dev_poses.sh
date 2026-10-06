#!/usr/bin/env bash
# Package the dev pose set for the LM track's J9 step 2, per REPLY-Q10-POSE-DELIVERY-2026-10-05.md.
# Runs on the POSE-TRACK MAC. Produces, next to the pkl directory's parent:
#
#     pkl_dev_rtmw_fp16.tar        ~335 MB, FLAT (entries are <clip>.pkl, not results/...)
#     pkl_dev_rtmw_fp16.sha256     checksum of the tar
#     pkl_dev_rtmw_fp16.manifest   one clip name per line, 918 lines
#
#     ./data/package_dev_poses.sh
#     SLT_PKL=results/pkl_dev_rtmw_fp16 SLT_EXPECT_N=918 ./data/package_dev_poses.sh
#
# WHY A TAR AND NOT A DIRECTORY. Colab mounts Drive over FUSE with high per-file latency, so reading
# 918 pkls per epoch through the mount is pathological -- the same reason colab_probe_scale.py stages
# to /content and writes only results back. One cp, extract once per session, train off local disk.
# Uncompressed on purpose: these are float arrays, gzip buys little and costs CPU every session.
#
# WHY THE MANIFEST AND CHECKSUM ARE NOT CEREMONY. A partially transferred archive that extracts
# without error is the failure class that put a 4-frame clip for a 7.8 s utterance into the n=400 set
# and got it all the way to an evaluation. The LM track is adding --expect-n 918 to the training entry
# point, the same shape as unisign/eval_openasl.py's guard, and that guard needs the expected count to
# come from HERE rather than from whatever happened to extract.
#
# SQUARE-NORM ONLY. The n=931 test arm was scored on results/pkl_split_rtmw_fp16 -- verified in
# results/eval_n931_pruned_ours_fps24.json's own config.poses -- so training must use the matching
# normalisation. The _raw companion stays on the board and this Mac; shipping it would invite step 2
# to adapt to a distribution the test set does not have.
#
# NATIVE RATE, NOT PRE-THINNED. fps_ratio_for_clip returns min(1.0, target/src) and thins at load, so
# one set serves every target rate -- including the still-unmeasured L16 adapted-@-16 fps row.
# Pre-thinning at 24 would foreclose that, and would not even be uniformly 24: clips already at or
# below the target are not thinned at all.
set -uo pipefail
cd "$(dirname "$0")/.."

PKL="${SLT_PKL:-results/pkl_dev_rtmw_fp16}"
EXPECT_N="${SLT_EXPECT_N:-918}"
OUT_DIR="${SLT_OUT_DIR:-results}"
BASE="$(basename "$PKL")"
TAR="$OUT_DIR/$BASE.tar"

[ -d "$PKL" ] || { echo "[pkg] ABORT: no directory $PKL" >&2; exit 1; }
n=$(find "$PKL" -name '*.pkl' | wc -l | tr -d ' ')
echo "[pkg] $PKL holds $n pkl(s)"
if [ "$n" -ne "$EXPECT_N" ]; then
    echo "[pkg] ABORT: expected $EXPECT_N, found $n. Not packaging a partial set -- that is the whole" >&2
    echo "[pkg]        point of the manifest. Re-run the extraction or set SLT_EXPECT_N deliberately." >&2
    exit 1
fi

# -C so the entries are flat: extracting gives <clip>.pkl in the current directory, with no
# results/ prefix for the consumer to strip.
echo "[pkg] writing $TAR"
tar -cf "$TAR" -C "$PKL" .
find "$PKL" -name '*.pkl' -exec basename {} .pkl \; | sort > "$OUT_DIR/$BASE.manifest"
shasum -a 256 "$TAR" > "$OUT_DIR/$BASE.sha256"

# Verify the archive lists exactly what the manifest claims, before anyone trusts either.
tar -tf "$TAR" | sed 's#^\./##' | grep '\.pkl$' | sed 's/\.pkl$//' | sort > /tmp/.pkg_intar.$$
if ! diff -q /tmp/.pkg_intar.$$ "$OUT_DIR/$BASE.manifest" >/dev/null; then
    echo "[pkg] ABORT: tar contents do not match the manifest" >&2
    rm -f /tmp/.pkg_intar.$$; exit 1
fi
rm -f /tmp/.pkg_intar.$$

echo "[pkg] OK  $(du -h "$TAR" | cut -f1)  $(wc -l < "$OUT_DIR/$BASE.manifest" | tr -d ' ') clips"
echo "[pkg] sha256: $(cut -d' ' -f1 "$OUT_DIR/$BASE.sha256")"
echo "[pkg]"
echo "[pkg] Upload these three to Drive under sltedge/poses/ :"
echo "[pkg]   $TAR"
echo "[pkg]   $OUT_DIR/$BASE.sha256"
echo "[pkg]   $OUT_DIR/$BASE.manifest"
echo "[pkg] (No Drive CLI on this Mac -- no CloudStorage mount, no rclone, no gdrive -- so the upload"
echo "[pkg]  is a manual step. See REPLY-Q10-ANSWERED for the headroom question.)"
