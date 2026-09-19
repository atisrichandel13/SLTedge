#!/usr/bin/env bash
# Pull run artefacts off the board WITHOUT clobbering the repo's own documents.
#
#   jetson/pull_results.sh [user@host]
#
# The board carries a full copy of the repo, including an older results/RESULTS.md. A plain
# `rsync board:results/ results/` therefore silently reverts the Mac's write-up (this happened on
# 2026-09-18 and had to be recovered from the session transcript). Markdown is authored on the Mac
# only, so it is excluded here; *.npz references stay on the board (large, gitignored).
set -euo pipefail
REMOTE="${1:-tgoyal@192.168.1.73}"
cd "$(dirname "$0")/.."
rsync -az --info=stats1 \
  --exclude='*.md' --exclude='*.npz' \
  -e "ssh -o StrictHostKeyChecking=accept-new -o HostKeyAlias=jetson-lpcv-03" \
  "$REMOTE:sign-lang-project/results/" results/
echo "[pull] results/ updated (documents and *.npz left alone)"
