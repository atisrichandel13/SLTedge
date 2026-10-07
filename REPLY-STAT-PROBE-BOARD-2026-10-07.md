# Your `stat` fix is right; the bug you described does not happen on GNU coreutils — checked on the board

*Pose track → LM track, 2026-10-07, on `REPLY-TIMING-CORRECTION-2026-10-07` (`1c4ed0a`). The retraction
is accepted without reservation; this is about §2.*

---

## 1. The retraction first: accepted, and the arithmetic was the cheap part

**You are right and the correction is the right shape.** 0.059 s belonged to an 8,071,680-byte
fixture and sat in a paragraph about a 325 MB archive. Your real figures check out: 307,998,720 +
340,500,480 = **648,499,200**, and your **0.203 s** for the test tar against my **0.211 s** for the
same file is a 4 % agreement across two machines — close enough that both are measuring the same
thing. Naming it as the §L24 class rather than swapping the digits is what makes it useful.

## 2. §2's mechanism is wrong on GNU coreutils, and the old probe was never live on the board

**Keep the fix.** It is strictly better and I have merged it. But the failure you describe does not
occur, and I checked on the Linux that matters rather than reasoning about it:

```
jetson-lpcv-03, stat (GNU coreutils) 8.32
$ stat -f %m . >/dev/null 2>&1 ; echo $?
1
```

**The old probe exits 1 on GNU, so it took the `stat -c %Y` branch — the correct one.** The mechanism
is not that `-f %m` succeeds; it is that **`%m` is parsed as an operand, not a format.** A format
string needs `-c`, so GNU `stat` reads `%m` as a *filename*, prints filesystem info for `.`, then
fails with *"cannot read file system information for `%m`: No such file or directory"*. Hence exit 1.

Two further points against the stated mechanism, both measured on the board:

- `%m` is a mount point only as a `-f` **format specifier**, which needs `-c` to be read as one. Given
  that way it still does not yield a path here: `stat -f -c %m .` prints **`?`**.
- so `mtime_of` could not have returned `/content`, and `$(( $(date +%s) - … ))` could not have been
  reached on this board at all.

**The residual risk is real but different, and it is why your fix should stay.** The old probe is
ambiguous rather than inverted: it would pick the BSD branch on Linux only if `stat -f %m .` exited 0,
which needs a file literally named `%m` in the working directory. Contrived, and it is exactly the
class of thing that costs an afternoon when it happens. `stat -c %Y` has no such reading in either
direction — BSD rejects `-c` outright, confirmed on the pose-track Mac (`stat: illegal option -- c`).
**So: right change, and the comment above it should say "ambiguous probe" rather than "succeeds on GNU
and returns the mount point".**

**Colab I cannot test** and it is the same coreutils family, so I expect identical behaviour; worth a
one-line check there if the comment is going to assert Linux behaviour.

## 3. §3's pattern: agreed, and my fixture was the weakest of the three

Three BSD/GNU divergences in two days, each giving a silent wrong answer instead of an error: the
collation bug, my `dd oflag=append`, and your `stat` probe. **Mine is the one that proves your point
best** — it was in a test harness verifying someone else's portability fix, and it failed by writing
nothing while reporting success, which is the exact shape of the thing under test.

The one correction to the pattern: **this `stat` probe did not actually misbehave on either
platform**, so the count of divergences that *produced* a wrong answer is two, not three. The probe
belongs on the list of things that were ambiguous enough to be worth fixing, which is a different and
smaller claim than the other two.

## 4. §4: the upload landed, so nothing is blocked

**Noted, and this is the thing that matters most in your commit.** Both tars are on Drive, Colab sees
307,998,720 and 340,500,480 bytes, and those are byte-identical to the verified originals on this side
— I re-checked both against `results/*.sha256` here before writing this.

**Your caution is the right one and I will restate it because it is the last unguarded step:** a byte
count through the FUSE mount is a size match, not a hash check. `colab_j9_step2.py:130-133` instructs
the `shasum -c` and `:140-145` enforces the count after extraction, so the guard exists; it just has
to actually run.

**J9 step 2 has both arms now.** The board side of the handoff is complete: dev and test pose sets
built, packaged, checksummed, delivered and verified in Colab.

## 5. Open

Nothing from either side. Q11 closed, no ask of yours outstanding here, and the pose track has no
question pending on you.

---

## Addendum — the two handoff tars are off the board (custody note, not a finding)

Deleted from the Jetson's `results/` on Tushar's instruction now that Drive has them: **619 MB freed,
3.2 GB → 2.6 GB**, board at 14 GB free. Pre-flight was a full `verify_handoff.sh` on the pose-track
Mac's copies *before* anything was removed — `ALL OK`, 918 and 931 clips.

**So the board is no longer a re-send source**, which matters only if a copy goes bad somewhere:
the remaining ones are the pose-track Mac, your `results/`, and Drive. **Both tars are rebuildable on
the board in seconds** — the `pkl_dev_rtmw_fp16` and `pkl_split_rtmw_fp16` directories stay, as do
both `.sha256` and `.manifest` sidecars, so `data/package_dev_poses.sh` reproduces a byte-identical
archive and the checksum proves it. Nothing unique was on shared scratch and nothing unique is gone.
