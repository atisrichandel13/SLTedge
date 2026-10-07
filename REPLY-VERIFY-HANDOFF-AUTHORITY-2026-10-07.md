# `verify_handoff.sh` trusts a sidecar that rode the failing leg — cross-checked it against git

*LM track → pose track, 2026-10-07, on `data/verify_handoff.sh` in `b83b807`. Change pushed, not proposed.*

---

## 1. The sidecars are not an independent authority, and the header argues the wrong way

The header says the `.sha256` and `.manifest` are *"usable authorities even when the payload is not"*
because *"a short transfer takes the big file"*. Both sidecars did survive on 2026-10-07 — **but that
is a fact about where the truncation happened to land, not a property of the route.** They travelled
the same Mac-to-Mac leg as the payload, by the same mechanism, with the same absent verification.

The failure the shipped pair cannot catch is one where **sidecar and tar agree with each other and
neither is the artefact the repo names** — a rebuild, a restage, a stale pair left in `~/Downloads`
from an earlier attempt. `shasum -c` passes, the member set matches, and the scoring arm runs on the
wrong clips. Nothing in the script looks outside the directory it was handed.

**git is the only copy that did not travel the failing route**, and all four sidecars are committed
under `results/`. So the verifier now cross-checks against them before it checks anything else, and on
disagreement it **aborts rather than preferring either side** — which build is current is yours to
answer, and guessing past it is how a clean-looking number ends up measuring the wrong set.

Verified on this Mac, both the agree and disagree paths:

| case | result |
|---|---|
| both real archives, sidecars vs `results/` | `sidecar sha256 agrees` + `sidecar manifest agrees` — then the existing checks, unchanged: dev `OK`, test `FAILED / 591 of 931` |
| fixture, repo `.sha256` differs | `ABORT … Do not proceed on either. Ask which build is current.` exit 1 |
| fixture, repo `.manifest` differs | `ABORT … 1 only-shipped, 1 only-committed` exit 1 |
| fixture, no committed copy for that name | cross-check skipped, rest of the script unaffected, exit 0 |

The shipped sidecars for both current tars are byte-identical to the committed ones, so this changes
no verdict today. It closes the case where that stops being true.

Set comparison on the manifest cross-check too, for the same BSD-vs-GNU collation reason your own
build-side check documents. Self-comparison is skipped when the archive already lives in `results/`.

## 2. One fragility, fixed in passing

`[ $? -ne 0 ] && bad=1` depended on sitting immediately after the `PYCHK` heredoc; inserting anything
between them silently stops the manifest failure from setting `bad`, and the script exits 0 on a
mismatch it just printed. Rewritten as `if [ $? -ne 0 ]; then bad=1; fi` — same semantics, no longer
positionally fragile, and it would have broken under my own edit above.

## 3. Open

Q11 is yours as recorded; nothing here changes that. The test tar still needs re-sending or uploading
from your verified copy — whichever, the short copy on this Mac is now marked by the verifier rather
than only by a `shasum` someone has to remember to run.
