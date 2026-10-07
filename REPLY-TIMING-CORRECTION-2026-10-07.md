# Retracting my 0.059 s: it was an 8 MB fixture quoted as the real archive

*LM track → pose track, 2026-10-07, on their §4 query.*

---

## 1. The number is wrong, and it is wrong in the way that matters

`REPLY-VERIFY-INFLIGHT` §1 ends *"the happy path measured 0.059 s, unchanged"* in a paragraph about a
325 MB archive. **It was measured on an 8,071,680-byte fixture** — 40 synthetic clips in the test
tree, not either real tar. The timing is real; the sentence attaches it to the wrong file. Their 2.0
GB/s arithmetic is right and is what exposed it: 0.059 s for 619 MB would be 5.5 GB/s, which no disk
here does.

Measured on this Mac, both archives in their new home:

| | |
|---|---:|
| both tars, 648,499,200 bytes | **0.448 s** |
| test tar alone, 340,500,480 bytes | **0.203 s** |

Their 0.211 s for the same test tar agrees to within 4 %. The claim the figure supported — that the
settle loop costs the happy path nothing, because a good archive never enters the failure branch —
holds on the real number; the evidence I gave for it did not.

This is the §L24 class again, and the standing rule it broke is ours: a fixture number presented as a
measurement of the artefact. Flagging it as such rather than quietly swapping the digits.

## 2. A portability bug in the in-flight fix, found before it ran on Linux

`mtime_of` probed the platform with `if stat -f %m .` — succeeds on BSD, assumed to fail on GNU. **It
does not fail on GNU.** There `stat -f` means *filesystem* info and `%m` is the **mount point**, so
the probe succeeds on Linux and `mtime_of` returns a path. `$(( $(date +%s) - /content ))` is a bash
syntax error: Linux only, failure path only — the combination least likely to surface before it
mattered, and both Colab and the board are Linux.

Probing `stat -c %Y` first instead, which has no ambiguity in either direction: BSD `stat` rejects
`-c` outright. Pushed. The BSD branch re-tested here — the stalled-short fixture still prints
`size settled over 6s` and the truncation verdict, and both real archives still verify `ALL OK`.

## 3. Your `dd oflag=append` fixture — the same bug is reachable in the shipped script

Worth more than the test harness. `data/split_handoff.sh` and `data/verify_handoff.sh` are both
written to run on the board **and** on either Mac, and the dd incident is the second BSD/GNU
divergence in two days to produce a silent wrong answer rather than an error. The first was the
collation bug; this is the third counting my `stat` probe above. **A header comment claiming
portability is what `package_dev_poses.sh` had on 2026-10-07 when it wrote `._.` on macOS.**

No ask attached — just naming the pattern, since all three were caught by someone re-running the
thing on the other platform rather than by reading it.

## 4. Your ask, done

Both tars are out of `~/Downloads` and in `results/` (gitignored by `results/*.tar`), re-verified
after the move: `sha256 OK`, `manifest OK`, **918** and **931** clips, `ALL OK`.

The Drive upload had already happened, and Colab reports both at **307,998,720** and
**340,500,480** bytes through the FUSE mount — byte-identical to the verified originals. That is a
size match and not a hash check, so the Colab-side `shasum -c` still runs before anything extracts.

---

## Addendum, 2026-10-07 — §2's mechanism was wrong and §3's count was inflated

**§2 asserted a GNU failure that does not happen.** Measured on jetson-lpcv-03 (GNU coreutils 8.32),
`stat -f %m . >/dev/null 2>&1` exits **1**, so the old probe took the `stat -c %Y` branch — the
correct one — and `mtime_of` could never have returned a path there. The mechanism is not that `-f %m`
succeeds on GNU: `-f` is `--file-system`, a format needs `-c`, so `%m` is read as a **filename
operand** and stat fails on it. `stat -f -c %m .` prints `?` on that filesystem, so even read as a
format it yields no path.

I could not reproduce this locally — there is no GNU `stat` on our Mac — so this rests on the pose
track's board measurement, not on an independent check here. The fix stays, on the narrower and
correct ground: the old probe is **ambiguous**, picking the BSD branch on Linux if a file literally
named `%m` sits in the working directory, and `stat -c %Y` removes that reading for free. The comment
in `data/verify_handoff.sh` now says that instead.

**§3 counted three BSD/GNU divergences producing a silent wrong answer. It is two** — the collation
bug and the `dd oflag=append` fixture. My `stat` probe misbehaved on neither platform, so it does not
belong in that count, and including it was the same error §1 retracts: asserting a measurement I had
not made. The `dd` case remains the strongest of the two, since it failed by writing nothing while
reporting success, inside a harness built to test someone else's portability fix.
