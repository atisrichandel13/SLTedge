#!/usr/bin/env bash
# Split a handoff archive into verifiable parts, for a transfer channel that truncates silently.
#
# WHY. On 2026-10-07 Microsoft Teams delivered pkl_split_rtmw_fp16.tar to the LM-track Mac at
# 208,823,294 of 340,500,480 bytes -- a clean prefix, 591 of 931 members, exit status fine
# (REPLY-TEST-TAR-TRUNCATED-2026-10-07.md 1). Re-sending the whole archive retries the whole risk and
# tells you nothing new if it fails again. Parts localise the failure: only the bad part is re-sent,
# and a per-part hash says which one it is without reading the payload.
#
#     ./data/split_handoff.sh results/pkl_split_rtmw_fp16.tar
#     SLT_PART_MB=64 ./data/split_handoff.sh results/pkl_split_rtmw_fp16.tar
#
# Writes results/parts/<name>.tar.part-aa ... and prints the per-part sha256 table to paste into the
# reply doc -- deliberately NOT written as a tracked file, so the authority for the hashes is git on
# both sides and nothing regenerable enters the history (.gitignore ignores results/parts/).
#
# The receiver rejoins with `cat` in shell glob order -- which is the split order, because `split`
# names parts aa, ab, ac in sequence -- then runs data/verify_handoff.sh on the result. That check is
# the one that matters: the part hashes find the bad part, the whole-archive hash proves the rejoin.
set -uo pipefail
cd "$(dirname "$0")/.."

TAR="${1:-}"
[ -n "$TAR" ] || { echo "usage: $0 <tar>" >&2; exit 2; }
[ -f "$TAR" ] || { echo "[spl] ABORT: no such file $TAR" >&2; exit 1; }
PART_MB="${SLT_PART_MB:-80}"
BASE="$(basename "$TAR")"
OUT="results/parts"

if command -v sha256sum >/dev/null 2>&1; then
    hash_of() { sha256sum "$1" | cut -d' ' -f1; }
else
    hash_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

# Verify the source before splitting. Splitting a bad archive produces parts that each verify against
# their own hash while the rejoined whole is still wrong -- the failure this script must not invent.
SIDE="${TAR%.tar}.sha256"
if [ -f "$SIDE" ]; then
    want=$(cut -d' ' -f1 "$SIDE"); got=$(hash_of "$TAR")
    [ "$want" = "$got" ] || { echo "[spl] ABORT: $TAR does not match $SIDE -- not splitting a bad source" >&2; exit 1; }
    echo "[spl] source sha256 OK"
else
    echo "[spl] WARNING: no $SIDE next to the archive; splitting an unverified source" >&2
fi

mkdir -p "$OUT"
rm -f "$OUT/$BASE.part-"*
split -b "${PART_MB}m" "$TAR" "$OUT/$BASE.part-"

echo "[spl] $(ls "$OUT/$BASE.part-"* | wc -l | tr -d ' ') parts of ${PART_MB} MB in $OUT/"
echo "[spl]"
echo "[spl] | part | bytes | sha256 |"
echo "[spl] |---|---:|---|"
for f in "$OUT/$BASE.part-"*; do
    printf '[spl] | `%s` | %s | `%s` |\n' "$(basename "$f")" "$(wc -c < "$f" | tr -d ' ')" "$(hash_of "$f")"
done
echo "[spl]"
echo "[spl] Receiver, in the directory holding the parts:"
echo "[spl]   shasum -a 256 *.part-*            # compare against the table above"
echo "[spl]   cat $BASE.part-* > $BASE"
echo "[spl]   ./data/verify_handoff.sh $BASE    # proves the rejoin, not just the parts"
