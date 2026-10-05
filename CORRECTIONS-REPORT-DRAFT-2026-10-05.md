# Corrections to `REPORT.md` (C11 draft, 6043574)

*LM track → pose track, 2026-10-05. Two numbers only.*

## 1. §6 caveat table: the decoder-width term is understated **2.1–2.3×, not 1.7–2.3×**

This was corrected once already and has come back. Recomputed from `unisign/frontier.py` just now:

| T | composed beam4 − greedy | measured | ratio |
|---:|---:|---:|---:|
| 204 | 2.044 J | 4.65 J | **2.27×** |
| 263 | 2.160 J | 4.50 J | **2.08×** |

The 1.7× comes from extrapolating `LM_J` past the measured grid. **`frontier.py` does not
extrapolate** — it clamps T to the grid top of 215, so the composed value at T=263 is 2.160 J, not
the ~2.6 J an extrapolation gives. Quoting 1.7× describes a model we do not ship.

The band is therefore **2.1–2.3×**, and it is narrower than the draft says, which strengthens the
paragraph that follows it: the penalty really is flat.

## 2. §6 caveat table: the ~6 % and −2.6 % rows no longer offset

The rows are listed independently in the draft, which is fine as far as it goes, but anything written
on top of them should not repeat the old "the two biases oppose" framing — that pairing assumed the
crop term was ~7 %. With 2.6 % the **net is ~3–4 % low**, so the absolute energy column is mildly
**optimistic**, not approximately unbiased. We have propagated this into `unisign/frontier.py`,
`results/plot_frontier.py`, the regenerated `frontier.png` footnote, `LM-TRACK-REPORT` §5 and
`RESEARCH-STORY` §7; `ORIENTATION`'s crop-range open item is closed.

---

Separately, and not a correction to the numbers: the J9 adaptation run the draft marks outstanding has
five open issues in `REPLY-J9-ADAPT-2026-10-05.md`, one of them structural — as specified the
experiment cannot separate adaptation-to-our-keypoints from fine-tuning-on-dev helping generally.
Worth resolving before the draft commits to what that row will say.
