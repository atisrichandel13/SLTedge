# Two of the 45 missing test clips are ours, not YouTube's

*LM track → pose track, 2026-10-05, on `REPLY-N931-RESAMPLES-2026-10-05.md` §3. Your conclusion is
right and nothing in §L19 or §L17 moves. The reason given for it is wrong for 2 clips, and the wrong
reason is load-bearing elsewhere.*

---

## 1. `index.json`'s arithmetic is not the check it has been used as

Both tracks have cited this identity as proof the fetch ran to exhaustion:

> *"931 + 43 = 974, so the fetch exhausted its candidate list rather than stopping early"*
> — `REPLY-J12-ACCEPTED-2026-10-05.md`, repeated in §2.5k, §L19 and `WORKSPLIT.md` J12.

**It is true by construction and cannot distinguish the two cases.** `index.json`'s `requested` field
is `want` echoed back — the `--n-clips` argument, which `data/fetch_full_split.sh:32` hardcodes to
`SLT_WANT="${SLT_WANT:-974}"`. It is not a count of what was attempted. So `n_ok + n_failed ==
requested` holds whenever the walk completes, whatever the candidate list contained, and **974 was
the number we asked for, not the number available.**

This is the §2.5i family again: a quantity that agrees with itself by construction, read as a check.

## 2. All 976 have a usable bbox, and 2 were never attempted

Ran `data/openasl_fetch.py`'s own `candidates()` with the exact filters the fetch script uses
(`--split test --min-dur 0 --max-dur 1e9 --max-per-video 0`), differing only in passing an **empty
exclude set**:

| | count |
|---|---:|
| `labels.test` names | 976 |
| test candidates **with a bbox**, no exclusions | **976** |
| `n_ok` | 931 |
| `n_failed` — every one carrying a recorded reason | 43 |
| accounted for | 974 |
| **unaccounted for, no failure record** | **2** |

So *"974 of the 976 had a usable bbox"* is wrong: **976 do.** The two unaccounted names are
`Ads-4j06eJY-00:00:14.000-00:00:15.733` and `Ads-4j06eJY-00:07:37.233-00:07:47.200` — in the test
split, with bboxes, in neither `clips` nor `failures`, with no reason recorded, because nothing ever
tried to fetch them.

## 3. We excluded them ourselves, and the flag cannot be turned off

`data/openasl_fetch.py:245`:

```python
ap.add_argument("--exclude-yid", action="append", default=["Ads-4j06eJY"],
                help="already have this signer (the baseline clip); repeatable")
```

`Ads-4j06eJY` is **our own single-clip development fixture** — the Bitcoin clip behind C2, C5, L4,
L8, `data/test_frames/` (299 JPEGs) and
`data/openasl_ref_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl`. The default was right for what the
flag was written for: a 5-clip signer-diverse sample at `--max-per-video 1`, where re-fetching a clip
already on disk wasted one of 40 attempts. It was never revisited when the same script was pointed at
a full-split run whose entire purpose is coverage.

**The reusable bug: an `action="append"` flag with a non-empty default cannot be cleared from the
command line.** Verified on that exact signature — no flag gives `['Ads-4j06eJY']`, and
`--exclude-yid X` gives `['Ads-4j06eJY', 'X']`. `fetch_full_split.sh` never passes `--exclude-yid`,
and could not have overridden it if it had. Worth knowing before the same pattern is used for a
deliberate exclusion that someone later needs to lift.

## 4. What survives and what does not

**Survives — your conclusion, unchanged.** The ceiling is **933 rather than 931**, and `sqrt(931/933)`
is 0.9989: about 0.001 on the BLEU-4 half-width. **The detectability ratio is 0.71 either way, and
nothing in §L19, §L17 or the J9 framing changes.** Two clips are not worth a re-fetch, and the fetch
environment is on your Mac, so this is explicitly **not** a request to run anything.

**Does not survive — the wording, in three places.** *"45 test names have no clip and never will"*,
*"974 of the 976 had a usable bbox"*, and the `931 + 43 = 974` exhaustion argument. 43 are dead links
and genuinely exhausted. 2 are self-inflicted and fetchable; we already hold the frames and the
authors' pose for one of them.

**Ask:** when §2.5k / `WORKSPLIT.md` J12 next get touched, say *"43 dead links and 2 excluded by a
stale `--exclude-yid` default"* rather than 45 unfetchable, and stop citing `requested − n_ok` as an
exhaustion check. Recorded in full as **§L20**.

## 5. One thing this buys, worth keeping

Because of that exclusion, the clip every early single-clip result was developed against is **not in
the 931-clip paired test set.** That is the right side of the line to be on — no single-clip tuning on
the Bitcoin clip can have leaked into the headline evaluation. It also explains why the pose gap was
invisible at n=1: the clip it was invisible on is not in the set where it is established.
