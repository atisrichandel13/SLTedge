#!/usr/bin/env bash
# Verify a received pose handoff archive -- RUN THIS ON THE RECEIVING MACHINE, BEFORE USING THE FILE.
#
# WHY THIS EXISTS. data/package_dev_poses.sh checks the archive where it is BUILT, and
# jetson/pull_results.sh brought it board -> pose-track Mac where it was checked again. Nobody checked
# the leg that actually failed. On 2026-10-07 pkl_split_rtmw_fp16.tar reached the LM-track Mac at
# 208,823,294 of 340,500,480 bytes -- a clean 61% prefix, 591 of 931 members, stable mtime, no
# transfer in flight (REPLY-TEST-TAR-TRUNCATED-2026-10-07.md 1). Their own shasum caught it, which is
# the point; this script is so that catching it does not depend on remembering to.
#
#     ./data/verify_handoff.sh ~/Downloads/pkl_split_rtmw_fp16.tar
#     ./data/verify_handoff.sh results/pkl_dev_rtmw_fp16.tar results/pkl_split_rtmw_fp16.tar
#
# The .sha256 and .manifest sidecars must sit next to the tar. Both are tiny and both arrived intact
# in the failure above -- a short transfer takes the big file, so the sidecars are usable authorities
# even when the payload is not.
#
# EXIT 0 only if, for every archive: the checksum matches its sidecar AND the member set equals the
# manifest exactly. Anything else exits 1 and names the file, because the failure mode that cost us
# the most this project is a bad artefact sitting unmarked beside good ones.
set -uo pipefail

[ "$#" -ge 1 ] || { echo "usage: $0 <tar> [<tar> ...]" >&2; exit 2; }

if command -v sha256sum >/dev/null 2>&1; then
    hash_of() { sha256sum "$1" | cut -d' ' -f1; }
elif command -v shasum >/dev/null 2>&1; then
    hash_of() { shasum -a 256 "$1" | cut -d' ' -f1; }
else
    echo "[vfy] ABORT: no sha256sum or shasum" >&2; exit 2
fi

bytes_of() { wc -c < "$1" | tr -d ' '; }

bad=0
for TAR in "$@"; do
    BASE="${TAR%.tar}"
    echo "[vfy] $TAR"
    for s in "$BASE.sha256" "$BASE.manifest"; do
        [ -f "$s" ] || { echo "[vfy]   ABORT: missing sidecar $s"; bad=1; continue 2; }
    done

    want_hash=$(cut -d' ' -f1 "$BASE.sha256")
    got_hash=$(hash_of "$TAR")
    got_bytes=$(bytes_of "$TAR")
    man_n=$(grep -c '[^[:space:]]' "$BASE.manifest")

    if [ "$got_hash" = "$want_hash" ]; then
        echo "[vfy]   sha256 OK  ($got_bytes bytes)"
    else
        echo "[vfy]   sha256 FAILED"
        echo "[vfy]     expected $want_hash"
        echo "[vfy]     got      $got_hash"
        echo "[vfy]     $got_bytes bytes on disk"
        # Distinguish a short transfer from corruption in place: a prefix truncation lists a readable
        # head and then stops, so the readable count is strictly short of the manifest. Corruption at
        # full length lists everything (or fails somewhere in the middle at full size).
        readable=$(tar -tf "$TAR" 2>/dev/null | grep -c '\.pkl$' || true)
        echo "[vfy]     $readable of $man_n members readable"
        if [ "$readable" -lt "$man_n" ]; then
            echo "[vfy]     => short/truncated transfer, not corruption in place. Re-send it."
        else
            echo "[vfy]     => full length but wrong bytes. Re-send it."
        fi
        echo "[vfy]   DO NOT USE. Mark it so it cannot be picked up by mistake:"
        echo "[vfy]     mv '$TAR' '$TAR.BAD'"
        bad=1
        continue
    fi

    # Member set against the manifest. Set comparison, not sorted-sequence: these clip names carry
    # colons and mixed case, and BSD vs GNU collation orders them differently, which produced a FALSE
    # mismatch on 2026-10-06. Same shape as package_dev_poses.sh's build-side check.
    python3 - "$TAR" "$BASE.manifest" <<'PYCHK'
import sys, tarfile
tar_path, man_path = sys.argv[1], sys.argv[2]
with tarfile.open(tar_path) as tf:
    entries = [n[2:] if n.startswith("./") else n for n in tf.getnames()]
entries = [e for e in entries if e not in ("", ".")]
nonpkl = sorted(e for e in entries if not e.endswith(".pkl"))
if nonpkl:
    sys.exit("[vfy]   ABORT: archive holds %d non-pkl entries: %s" % (len(nonpkl), nonpkl[:5]))
have = {e[:-4] for e in entries}
want = {ln.strip() for ln in open(man_path) if ln.strip()}
missing, extra = sorted(want - have), sorted(have - want)
if missing or extra:
    sys.exit("[vfy]   ABORT: %d manifest names, %d in archive; %d missing %s; %d extra %s"
             % (len(want), len(have), len(missing), missing[:5], len(extra), extra[:5]))
if len(entries) != len(have):
    sys.exit("[vfy]   ABORT: %d entries collapse to %d names -- duplicates" % (len(entries), len(have)))
print("[vfy]   manifest OK  %d clips" % len(have))
PYCHK
    [ $? -ne 0 ] && bad=1
done

if [ "$bad" -ne 0 ]; then
    echo "[vfy] FAIL -- at least one archive is not usable"
    exit 1
fi
echo "[vfy] ALL OK"
