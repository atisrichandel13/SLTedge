#!/usr/bin/env bash
# Package the dev pose set for the LM track's J9 step 2, per REPLY-Q10-POSE-DELIVERY-2026-10-05.md.
#
# RUNS ON EITHER THE BOARD OR THE POSE-TRACK MAC -- the relative paths are the same in both repo
# copies. Prefer the BOARD (REPLY-Q10-DRIVE-ROUTE): the pkls are already there, so it saves a 335 MB
# hop, and jetson/pull_results.sh already rsyncs results/ off the board excluding only *.md and
# *.npz (:13-14) -- so a tar in the board's results/ arrives through the existing tested channel,
# while .gitignore's results/*.tar keeps it out of the history. Those two compose by luck, not design.
#
# Produces, next to the pkl directory's parent:
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
# COPYFILE_DISABLE stops macOS tar writing AppleDouble "._" members for extended attributes. Without
# it, building on the pose-track Mac produces an archive whose first entry is "._." and the integrity
# check below aborts -- which is the check working, but the archive should not be wrong in the first
# place. Harmless and ignored on the Jetson. Found 2026-10-07 building the test tar on macOS; the dev
# tar was clean only because it was built on the board.
COPYFILE_DISABLE=1 tar -cf "$TAR" -C "$PKL" .
# LC_ALL=C on every sort here. Clip names carry colons and mixed case, and BSD vs GNU collation
# order them differently, so a manifest sorted in the board's locale and a listing sorted in the
# consumer's produce a diff that looks like missing files and is not. Seen for real on 2026-10-06:
# the pulled directories compared "MISMATCH" against this manifest while holding the identical 918
# names. Byte-order collation is the same everywhere.
find "$PKL" -name '*.pkl' -exec basename {} .pkl \; | LC_ALL=C sort > "$OUT_DIR/$BASE.manifest"
# sha256: coreutils on the Jetson, perl script on macOS. Both exist on this board (checked:
# /usr/bin/sha256sum and /usr/bin/shasum), but pick whichever is present so the script is portable.
if command -v sha256sum >/dev/null 2>&1; then
    ( cd "$OUT_DIR" && sha256sum "$BASE.tar" ) > "$OUT_DIR/$BASE.sha256"
elif command -v shasum >/dev/null 2>&1; then
    ( cd "$OUT_DIR" && shasum -a 256 "$BASE.tar" ) > "$OUT_DIR/$BASE.sha256"
else
    echo "[pkg] ABORT: no sha256sum or shasum" >&2; exit 1
fi

# Verify the archive lists exactly what the manifest claims -- and nothing else.
# COMPARED AS SETS, not as two sorted sequences. The sorted-sequence form was the first version and
# it is the weaker one: on 2026-10-06 a manifest sorted in the board's locale, diffed against a
# listing sorted on macOS, reported a mismatch with nothing missing, because these clip names carry
# colons and mixed case and BSD and GNU collate them differently. Pinning LC_ALL=C fixes one site and
# has to be remembered at the next; a set difference is immune by construction, and its failure mode
# is not a FALSE mismatch -- which is the expensive direction, because a manifest that cries wolf
# gets ignored the day it is right. Shape taken from the LM track's train_adapt.py guard (L-reply
# 2026-10-06 1). It also reports the two directions separately, which is what a reader needs.
python3 - "$TAR" "$OUT_DIR/$BASE.manifest" <<'PYCHK'
import sys, tarfile
tar_path, man_path = sys.argv[1], sys.argv[2]
with tarfile.open(tar_path) as tf:
    entries = [n[2:] if n.startswith("./") else n for n in tf.getnames()]
entries = [e for e in entries if e not in ("", ".")]
nonpkl = sorted(e for e in entries if not e.endswith(".pkl"))
if nonpkl:
    sys.exit("[pkg] ABORT: archive holds %d non-pkl entries: %s" % (len(nonpkl), nonpkl[:5]))
have = {e[:-4] for e in entries}
want = {ln.strip() for ln in open(man_path) if ln.strip()}
missing, extra = sorted(want - have), sorted(have - want)
if missing or extra:
    sys.exit("[pkg] ABORT: archive does not match the manifest.\n"
             "  manifest %d names, archive %d\n"
             "  %d in the manifest, absent from the archive: %s\n"
             "  %d in the archive, absent from the manifest: %s"
             % (len(want), len(have), len(missing), missing[:5], len(extra), extra[:5]))
if len(entries) != len(have):
    sys.exit("[pkg] ABORT: %d entries collapse to %d names -- duplicates in the archive"
             % (len(entries), len(have)))
PYCHK
rc=$?
[ "$rc" -ne 0 ] && exit 1

echo "[pkg] OK  $(du -h "$TAR" | cut -f1)  $(wc -l < "$OUT_DIR/$BASE.manifest" | tr -d ' ') clips"
echo "[pkg] sha256: $(cut -d' ' -f1 "$OUT_DIR/$BASE.sha256")"
echo "[pkg]"
echo "[pkg] Upload these three to Drive under sltedge/poses/ :"
echo "[pkg]   $TAR"
echo "[pkg]   $OUT_DIR/$BASE.sha256"
echo "[pkg]   $OUT_DIR/$BASE.manifest"
echo "[pkg] Drive is reachable ONLY from inside Colab in this project (drive.mount, authenticated by"
echo "[pkg] the runtime) -- neither Mac has a mount, rclone or gdrive. So the last leg is a browser"
echo "[pkg] action from whichever machine holds the file. If this ran on the board, jetson/pull_results.sh"
echo "[pkg] brings it to the pose-track Mac through the existing channel."
