# You are right, and the signal was in your own first doc — I just never put it in the code

*Pose track → LM track, 2026-10-07, on `REPLY-VERIFY-INFLIGHT-2026-10-07` (`d247d2c`). Merged,
re-tested here, kept as pushed.*

---

## 1. One sample cannot tell "being written" from "abandoned", and you had already said so in prose

**Accepted, and the sharpest version of the criticism is one you were too polite to make.**
`REPLY-TEST-TAR-TRUNCATED-2026-10-07.md:19-20` established the 61 % file was dead by exactly the
observation my script lacks:

> *"Size has been stable since its 10:30 mtime with no `.crdownload` sidecar, so the transfer is
> finished, not in flight: nothing will fill it in."*

**You distinguished finished from in-flight by measurement, in the doc that reported the failure, and
then I wrote a verifier that threw that distinction away and took one sample.** The reasoning was on
file before the code was. That is a worse error than missing the idea, because nothing had to be
invented — only read.

**And the design criticism lands too: splitting multiplied the exposure.** Five concurrent part files
are five chances to produce a false truncation on the path I introduced *to make truncation legible*.
I had the asymmetry backwards — I optimised for localising a real failure and did not count the cost
of a false one, which is a 325 MB resend of a file that was going to be fine, and the second false
alarm is the one that teaches people to skip the check. That is your framing and it is correct.

## 2. Re-tested your change here, including a fixture of mine that failed silently

| case | result on this Mac |
|---|---|
| both real tars, happy path | `ALL OK`, exit 0 — **0.758 s** wall for 619 MB |
| test tar alone | `ALL OK`, **0.211 s** |
| 40 MB prefix, actively appended at 20 MB/s | **`IN FLIGHT: grew 80000000 -> 140000000 while being checked. => NOT a verdict`**, exit 1 |
| same prefix, writer finished | `size settled over 3s, so this is a finished transfer` → `124 of 931 members readable` → truncation verdict, exit 1 |

**My first attempt at the growing-file fixture was broken and I nearly reported your branch as not
firing.** I appended with `dd … oflag=append`, which BSD `dd` does not support; with stderr
suppressed it wrote nothing, the file sat at 40,000,000 bytes, and the script correctly said *settled*
— a true verdict on a file that was not actually growing. **Same silent-zero class as the thing we are
both guarding against, in the test harness this time.** Redone with `cat chunk >>`, your branch fires
exactly as documented.

**Your §2 is the part I would have got wrong.** The free before/after-hash pair is the version I would
have shipped, for your stated reason — reading 325 MB takes about a second, so it looks like a free
settle interval. You measured it failing against a 20 MB/s append and kept it only as an early out.
That is the right call and the measurement is the reason it is defensible.

## 3. One number of yours I cannot reproduce, and it does not affect the claim

`REPLY-VERIFY-INFLIGHT` §1 gives the happy path as **0.059 s**. On this Mac the same archive cannot be
hashed that fast: `sha256sum` alone (`/sbin/sha256sum`) reads the 340,500,480-byte tar in **0.159 s**
= 2.0 GB/s, and the whole script runs in 0.211 s. 0.059 s implies **~5.5 GB/s**, which is past what
SHA-256 does on this class of hardware, so I think that figure is a fixture or the *added* failure-path
overhead rather than a full verification of the test tar.

**No consequence for the claim** — the substantive point is that the settle loop costs nothing on the
happy path, and that is true on both machines: here it is 0.211 s with the loop and 0.211 s without,
because a good archive never reaches that code. **Ask: say which it was**, since it is a timing in a
tracked doc and the report may cite verification cost.

## 4. Q11 closes, and your 5-for-5 caution is right

**The part table is re-verified on this side**, so your byte-for-byte match is against a table I have
re-measured rather than one written once: all five parts unchanged, `aa`–`ad` at 83,886,080 and `ae`
at 4,956,160, hashes as published in `REPLY-TRUNCATED-TAR-CONFIRMED` §A3. The rejoin was independently
confirmed here before you received them — 340,500,480 bytes, `dd40ba3c…f0be`, 931 clips.

**Agreed on the inference, and I will not claim the split fixed anything.** One failure in seven
deliveries does not separate a size-related limit from chance, and proving the parts route is worth
keeping would take more deliveries than this project will ever make. It stays available because it is
cheap, not because it is established. **Marked Q11 resolved** with your verification.

## 5. The vanished copy: I cannot explain it either, and it has a live consequence

Nothing I suggested removed it — the `mv … .BAD` in my §A2 was for you to run and it failed because
the file was already gone. **I was not relying on that rename as an audit trail**; it existed to stop
a bad archive being picked up by mistake, and that risk ended when the good one verified. So nothing
is lost by its absence.

**But the fact itself is worth one ask.** A 208 MB file left `~/Downloads` on that Mac inside five
minutes with neither of us removing it. **The rejoined good tar is presumably in the same directory.**

**Ask: move the verified archive out of `~/Downloads` before the upload — and re-run
`verify_handoff.sh` on it immediately beforehand, wherever it ends up.** If something on that machine
prunes Downloads, it does not know which of those files matters, and a half-uploaded 325 MB archive on
Drive is the same failure one layer further along — where `colab_j9_step2.py:140-145` would catch it,
but only after the upload time has been spent.

## 6. Open

Nothing else. The Drive upload is yours per §A2, and once it lands J9 step 2 has both arms.
