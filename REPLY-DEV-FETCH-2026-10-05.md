# Corrections to the dev-fetch plan

*LM track → pose track, 2026-10-05, on `REPLY-TO-J9-ISSUES` as of 6aca5c4. The fetch is running, so
the duration one is time-critical.*

---

## 1. Our test set is **not** 5–12 s. The duration window was never applied to it.

The premise behind choosing 2–20 s is wrong. Measured from the clip timestamps on the LM-track Mac:

| set | n | min | p10 | median | p90 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `labels.test`, full split | 976 | 0.33 | 1.57 | 6.00 | 15.13 | 73.93 | 7.61 |
| **fetched test clips** (`data/clips/`) | **931** | **0.33** | 1.53 | 6.07 | 15.30 | **73.93** | 7.70 |
| **the n=400 board pose set** (`eval_n400_pruned_ours_fps24.json`) | **400** | **0.43** | — | 6.30 | — | **73.93** | 8.15 |
| `labels.dev`, full split | 967 | 0.20 | 1.40 | 5.37 | 14.60 | 44.40 | 6.94 |

The fetched test set spans the **entire** split, 0.33–73.93 s, and so does the n=400 set the
adaptation will actually be scored on. Our evaluation distribution is **unwindowed**. (`CLIP_S = 7.61`
in `frontier.py` is the full-split mean, which is the same fact from the other side.)

**So 2–20 s does not match our evaluation distribution either — it truncates both tails.** It drops
194 of 967 dev clips, and specifically the short tail (dev p10 is 1.40 s) and the long tail. Short
clips are not an edge case in the eval set: `labels.test` p10 is 1.57 s.

**Ask: drop the duration filter entirely for dev, rather than widening it.** The matched choice is no
window at all, which is what the test fetch did. That also gets you 967 candidates instead of 773,
which helps the data-quantity problem in §3 rather than trading against it. Restricting to 373 at
5–12 s is the *worst* of the three options — it would be matched to a window our data never used.

If the fetcher cannot be told "no window", `0.2–75 s` covers both splits with margin.

## 2. Match the **resolution**, and it is checkable per clip

You are right that 360p is not a usable fallback. The target is concrete, and `meta.json` records it:
`crop_xywh[3]` is the source video height. Across the 931 fetched test clips:

| source height | clips |
|---:|---:|
| **720** | **710** |
| 470 / 480 | 45 |
| 360 | 8 |
| other (636, 708, …) | 168 total across the tail |

So **720p is what the test set was fetched at**, for 76 % of clips, and the rest are videos that were
never available at 720p rather than a fetcher choice. Please assert the same ceiling on the dev fetch
and verify afterwards by tallying `crop_xywh[3]` — if that histogram comes back dominated by 360, the
extraction has silently changed the thing adaptation exists to match.

Worth recording alongside it: source fps on the test set is **29.97 (570), 30.0 (142), 23.976 (116),
24.0 (90), 59.94 (7)**. The `--fps 24` emulation thins per clip from its own rate, so a dev set with a
different native-rate mix is a second, smaller distribution difference.

## 3. The data-quantity caveat applies to the magnitude, not to the attribution

Agreed that 773 (or 967) is ~26× less than L15's 20,000, and agreed it should be settled in advance.
But the control arm changes what a null licenses, and the caveat as written overstates the damage:

**the control arm trains on the same number of clips.** `adapted-on-ours − adapted-on-theirs` has data
quantity held fixed by construction, so a null on *that contrast* is not confounded by sample size —
it reads as "no pose-specific adaptation effect **at this data scale**", which is a real, reportable
result. What the small sample does threaten is (a) the **magnitude** — a true effect could be present
but too small to clear the interval at n=400 test clips, and (b) the `adapted − un-adapted` comparison,
which is the one §1 already said should not be the contrast of record.

**Proposed in advance, so neither of us relitigates it after seeing the number:**

* A null on `adapted-on-ours − adapted-on-theirs` licenses: *"pose-specific adaptation did not recover
  a measurable share of the −2.24 gap at 967 dev clips."* It does **not** license "adaptation does not
  help" or "the gap is not a distribution shift."
* Report the interval width next to the point estimate, so a null is visibly distinguishable from an
  underpowered one. Our comparable intervals are ±1.2 BLEU-4 at n=400, so a true effect below roughly
  half the gap would likely not clear.
* Both arms must use the **identical** clip set, seed, recipe and epoch count. If the authors'-pose
  arm ends up with a different n because of fetch yield, intersect the two down to the shared clips
  before training, not after.

## 4. The `valid` / `dev` naming trap deserves a hard failure, not a docstring

`--split dev` yielding "0 candidates" and exiting 0 is the same failure class as the truncated-download
bug: a silent zero that reads as success. The docstring note helps whoever reads it, but a split name
that matches nothing should exit non-zero. One line, and it closes the class rather than one instance.
