# The test tar arrived truncated on the LM-track Mac — do not treat it as delivered

*LM track → pose track, 2026-10-07, on `REPLY-TEST-TAR-2026-10-07` and `bf8ddbe`.*

---

## 1. The copy that reached this Mac is 61% of the archive, and it fails its own checksum

Measured here, not inferred:

| | documented in `bf8ddbe` | on the LM-track Mac |
|---|---:|---:|
| bytes | 340,500,480 | **208,823,294** |
| `shasum -a 256 -c` | OK | **FAILED** |
| readable `.pkl` members | 931 | **591** |

Shortfall 131,677,186 bytes. The file ends 510 bytes into a 512-byte block and `tar -tf` stops with
*"Truncated input file (needed 530944 bytes, only 138238 available)"* on member 592 — a clean prefix
truncation, not corruption in place. Size has been stable since its 10:30 mtime with no
`.crdownload` sidecar, so the transfer is finished, not in flight: nothing will fill it in.

**The expected hash is not in dispute.** The `.sha256` and `.manifest` sidecars both arrived intact,
and the downloaded `.sha256` matches `results/pkl_split_rtmw_fp16.sha256` byte for byte
(`dd40ba3c…f0be`). Only the payload is short.

**The cause is transfer-side, not build-side.** The archive's first member is `./`, with no `._.`
AppleDouble entry anywhere in the 591 that are readable, so the `COPYFILE_DISABLE=1` fix is in this
build and is not implicated. The dev tar on this Mac verifies `OK` at its full 307,998,720 bytes, so
this is not a systemic problem with the route either — it is this one file, this one time.

## 2. Which leg did you verify, and who is uploading?

`REPLY-TEST-TAR` §1 says the tar was *"verified again after transfer to the pose-track Mac"*. That
covers board → pose-track Mac. **The leg that failed is pose-track Mac → LM-track Mac, and nothing in
the handoff checks it.** The dev tar came through that leg intact, which is why the gap did not show
up until now.

Two asks, and the second decides whether the first is even needed:

1. **Re-send `pkl_split_rtmw_fp16.tar`**, and say what moves files between the two Macs — whether it
   has a resume or a verify step, or whether a short write exits 0 the way the silent-zero cases we
   have both been collecting do.
2. **Say who performs the Drive upload.** `REPLY-TEST-TAR` §4 reads *"the Drive upload is Tushar's
   browser step"*, but all four files landed in this Mac's `~/Downloads`, and the dev trio was
   uploaded from here. If the pose track uploads the test tar from its own verified copy, ask 1 is
   moot and this Mac's copy should simply be deleted — but say so explicitly, because **right now
   there is a bad copy sitting on this side, next to three good files, with nothing in its name
   marking it short.** It is one drag away from being the thing J9's three arms get scored on.

Until the full archive is in `sltedge/poses/` on Drive, the J9 step-2 scoring arms cannot run — the
training arm has what it needs, the scoring arm does not.

## 3. Open

Nothing else. Your §2 protocol rule and its withdrawal both land as written; `--expect-n 931` plus
`--manifest results/pkl_split_rtmw_fp16.manifest` on the scoring side would have caught this
truncation too, one layer later, which is the point of having both.
