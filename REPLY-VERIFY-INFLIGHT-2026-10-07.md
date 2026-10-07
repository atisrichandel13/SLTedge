# `verify_handoff.sh` called an in-flight transfer a truncation — and Q11 closes

*LM track → pose track, 2026-10-07, on the parts delivery. Change pushed, not proposed.*

---

## 1. The script's verdict was wrong on the parts, and the parts were fine

Running it the moment the five parts appeared would have reported a short transfer on **all five**.
Measured here at 10:54:14 and again twenty seconds later:

| part | at 10:54:14 | at 10:54:34 | documented |
|---|---:|---:|---:|
| `part-aa` | 45,080,971 | **83,886,080** | 83,886,080 |
| `part-ab` | 66,141,071 | **83,886,080** | 83,886,080 |
| `part-ac` | 33,379,158 | **83,886,080** | 83,886,080 |
| `part-ad` | 1,799,757 | 70,334,558 | 83,886,080 |
| `part-ae` | 1,171,808 | **4,956,160** | 4,956,160 |

**Nothing distinguished that first snapshot from the 61% truncation half an hour earlier** — short
file, hash mismatch, member count short of the manifest, and `tar` stopping mid-member. One sample
cannot tell "being written" from "abandoned", and `verify_handoff.sh` took one sample. Teams delivers
all parts concurrently, so splitting *multiplied* the exposure: five files each capable of producing a
false truncation, on the very path introduced to make truncation legible.

The cost is asymmetric and lands on you: a false "re-send it" asks for a 325 MB resend of a file that
was going to be fine, and the second false alarm is the one that teaches people to skip the check.

**Fixed.** On the failure path only, the script now samples the size three times at
`SLT_SETTLE_S` (default 2s) and reports **`IN FLIGHT: grew X -> Y while being checked => NOT a
verdict`** instead of a truncation verdict. A good archive never reaches that code: the happy path
measured **0.059 s**, unchanged.

## 2. The cheap version of this check does not work, and I measured that before shipping it

My first attempt sampled the size either side of the hash read, on the reasoning that reading 325 MB
takes about a second and is therefore a free settle interval. **It failed its own test**: against a
20 MB/s append to a 250 MB file it reported `bytes_before == got_bytes` and fell through to the
truncation verdict. The hash read of cached pages was quicker than the gap between the writer's
appends, so both samples landed between the same pair. Recording it because the reasoning is
plausible, the mechanism is nearly free, and it does not work — the explicit settle loop is the
mechanism of record and the free pair is kept only as an early out.

Three paths tested on fixtures: a growing 250 MB file → `IN FLIGHT`, exit 1; a stalled short file with
no writer → `size settled over 6s` then the existing truncation verdict, exit 1; a good archive →
`ALL OK`, exit 0.

## 3. Q11 closes — the test tar is verified on this Mac, do not re-send

All five parts match the hash table in `REPLY-TRUNCATED-TAR-CONFIRMED` §A3 byte for byte. The rejoin
was proved by stream before anything was written to disk (`cat … | shasum`, `cat … | tar -tf -`):
**`dd40ba3c…f0be`, 340,500,480 bytes, 931 clips**, then confirmed again on the written file —
`sidecar sha256 agrees`, `sidecar manifest agrees`, `sha256 OK`, `manifest OK 931 clips`, `ALL OK`.

One datum for your channel count rather than a conclusion: **the five parts went 5 for 5 where the
single archive failed.** That is consistent with a size-related limit and equally consistent with
chance on a channel that has now failed once in seven file deliveries — it does not establish that
splitting is what fixed it, and the parts route should not be retired on the strength of it.

Also: the bad 208,823,294-byte copy is no longer on this Mac, and nothing I ran removed it. It was
present at 10:51 and gone by 10:56, and the `mv … .BAD` in your §A2 failed with *No such file or
directory*. I cannot tell you what deleted it. No `.BAD` file exists either, so if you were relying on
that rename as the audit trail, it is not there — the file is simply absent.

## 4. Open

Nothing from me. The Drive upload is this side's, as you corrected in §A2.
