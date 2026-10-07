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

if stat -f %m . >/dev/null 2>&1; then mtime_of() { stat -f %m "$1"; }; else mtime_of() { stat -c %Y "$1"; }; fi

bad=0
for TAR in "$@"; do
    BASE="${TAR%.tar}"
    echo "[vfy] $TAR"
    for s in "$BASE.sha256" "$BASE.manifest"; do
        [ -f "$s" ] || { echo "[vfy]   ABORT: missing sidecar $s"; bad=1; continue 2; }
    done

    # CROSS-CHECK THE SIDECARS AGAINST THE REPO'S COMMITTED COPIES. The .sha256 and .manifest that
    # arrive beside the tar rode the SAME leg as the payload, so they are the weaker authority. A short
    # transfer leaves them intact -- that is why they were usable on 2026-10-07 -- but that is luck
    # about which file the truncation landed in, not a property of the route. The case they cannot
    # catch is a rebuilt-or-restaged pair that agrees with itself: sidecar and tar match, and neither
    # is the artefact the repo names. git is the only copy that did not travel the failing route, so it
    # is the authority of record. All four sidecars are committed under results/.
    #
    # On disagreement this ABORTS rather than preferring either one. Which build is current is a
    # question for the pose track; guessing past it is how a clean-looking number ends up measuring the
    # wrong set of clips.
    REPO_RESULTS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../results" 2>/dev/null && pwd)" || REPO_RESULTS=""
    SIDE_DIR="$(cd "$(dirname "$BASE")" 2>/dev/null && pwd)" || SIDE_DIR=""
    NAME="$(basename "$BASE")"
    if [ -n "$REPO_RESULTS" ] && [ "$SIDE_DIR" != "$REPO_RESULTS" ]; then
        if [ -f "$REPO_RESULTS/$NAME.sha256" ]; then
            ship_h=$(cut -d' ' -f1 "$BASE.sha256")
            repo_h=$(cut -d' ' -f1 "$REPO_RESULTS/$NAME.sha256")
            if [ "$ship_h" != "$repo_h" ]; then
                echo "[vfy]   ABORT: shipped .sha256 disagrees with results/$NAME.sha256"
                echo "[vfy]     shipped $ship_h"
                echo "[vfy]     repo    $repo_h"
                echo "[vfy]     Do not proceed on either. Ask which build is current."
                bad=1
                continue
            fi
            echo "[vfy]   sidecar sha256 agrees with results/$NAME.sha256"
        fi
        if [ -f "$REPO_RESULTS/$NAME.manifest" ]; then
            # Set comparison again, for the collation reason documented below.
            if ! python3 - "$BASE.manifest" "$REPO_RESULTS/$NAME.manifest" <<'PYMAN'
import sys
a = {ln.strip() for ln in open(sys.argv[1]) if ln.strip()}
b = {ln.strip() for ln in open(sys.argv[2]) if ln.strip()}
if a != b:
    sys.exit("[vfy]   ABORT: shipped .manifest disagrees with the committed one: "
             "%d shipped, %d committed, %d only-shipped %s, %d only-committed %s"
             % (len(a), len(b), len(a - b), sorted(a - b)[:3], len(b - a), sorted(b - a)[:3]))
print("[vfy]   sidecar manifest agrees with results/%s.manifest  (%d clips)"
      % (sys.argv[2].rsplit("/", 1)[-1][:-9], len(a)))
PYMAN
            then
                echo "[vfy]     Do not proceed. Ask which clip set is current."
                bad=1
                continue
            fi
        fi
    fi

    want_hash=$(cut -d' ' -f1 "$BASE.sha256")
    # Size on BOTH sides of the hash read. Reading 325 MB takes about a second, which costs nothing and
    # is a free settle interval: if the file grew across it, the transfer is still running and the
    # mismatch below is not a verdict. See the IN FLIGHT branch for why this is not a nicety.
    bytes_before=$(bytes_of "$TAR")
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
        # IS IT STILL ARRIVING? Decided BEFORE calling it a bad transfer, because a short file being
        # written and a short file abandoned are indistinguishable from ONE sample, and a premature
        # "re-send it" asks for a 325 MB resend of a file that was going to be fine. Seen for real on
        # 2026-10-07: all five Teams parts measured short at 10:54:14 and four were complete and
        # hash-correct twenty seconds later, looking no different from the 61% truncation half an hour
        # before.
        #
        # This costs SLT_SETTLE_S x 3 seconds and ONLY on the failure path -- a good archive never
        # reaches here. That is the right place to spend it: the happy path stays instant and the
        # expensive-to-get-wrong path gets a real measurement instead of one sample.
        #
        # Two signals, because the first one alone is not enough. The before/after-hash pair catches a
        # fast writer for free, but it was measured on 2026-10-07 FAILING to catch a 20 MB/s append to
        # a 250 MB file -- the hash read of cached zeros was quicker than the gap between writes, and
        # both samples landed between the same pair of appends. So the explicit settle loop below is
        # the mechanism of record and the free pair is only an early out.
        settle_b="$got_bytes"
        grew=0
        for _ in 1 2 3; do
            sleep "${SLT_SETTLE_S:-2}"
            nb=$(bytes_of "$TAR")
            [ "$nb" = "$settle_b" ] || { grew=1; settle_b="$nb"; }
        done
        if [ "$bytes_before" != "$got_bytes" ] || [ "$grew" -ne 0 ]; then
            echo "[vfy]     IN FLIGHT: grew $bytes_before -> $settle_b while being checked."
            echo "[vfy]     => NOT a verdict. Wait for the size to settle, then re-run."
            bad=1
            continue
        fi
        echo "[vfy]     size settled over $(( ${SLT_SETTLE_S:-2} * 3 ))s, so this is a finished transfer"
        age=$(( $(date +%s) - $(mtime_of "$TAR") ))
        if [ "$age" -lt 30 ]; then
            echo "[vfy]     CAUTION: last written ${age}s ago -- settled, but only just."
        fi
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
    if [ $? -ne 0 ]; then bad=1; fi
done

if [ "$bad" -ne 0 ]; then
    echo "[vfy] FAIL -- at least one archive is not usable"
    exit 1
fi
echo "[vfy] ALL OK"
