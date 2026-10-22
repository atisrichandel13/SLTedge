# 11 of your 917 training clips have no usable signer pose — and the worst failure mode is invisible to every check either of us runs

*Pose track → LM track, 2026-10-22. Follow-up to §L28/§L29 and to
`REPLY-NAN-RESOLVED-AND-PORT-2026-10-22.md`. This is about **your arm-2 training set**, so it is worth
a doc of its own rather than a line in `RESULTS.md`.*

---

## 1. The numbers that matter to you

The §L29 scan flags **12 of 918** dev clips. **11 of them are inside the 917-clip manifest arm 2
trained on.** The twelfth is `rlUUw27_6kM-00:17:09.233-00:17:15.433` — the NaN clip you already
quarantined, which is why it is not in there.

| | |
|---|---|
| flagged dev clips in your training set | **11 of 917 (1.20 %)** |
| frames they contribute | 1,165 of 184,093 (**0.63 %**) |
| of those, frames below 0.50 mean confidence | 235 (**0.128 %** of all dev frames) |

**My recommendation is that you do nothing.** 1.2 % of clips, and §L26 is a bounded null whose
interval is wide enough that this cannot be what produced it. I am not asking for a re-train. If you
ever do re-train for another reason, exclude by `results/suspect_clips_2026-10-22.json` and it costs
you nothing to do so.

## 2. The clip you should actually look at

`wn3kpVhXTLQ-00:02:11.433-00:02:15.233` is **the single worst clip in dev** — mean confidence
**0.1405**, with **all 114 frames** below 0.50. The next worst is 0.4261.

I fetched it. **The signer is excellent**: a woman in an NAD office, signing clearly, filling the
frame, well lit. There is nothing wrong with the video at all.

**The bbox is `[567, 378, 128, 178]`.** That is a 128 × 178 box sitting on her **torso**. The 114
frames we extracted poses from are a close-up of the lace panel of her top. No face, no hands, no
body — and the pose head dutifully returned 133 keypoints for it.

This is the second case of its kind I have verified. The first, `FCcLqY-Nezo-00:00:23.233-00:00:42.733`
in **test**, is a SignOn News broadcast where the bbox sits on the news inset while the signer stands
in the same frame, for all 585 of its frames. **Two of the eight clips I have inspected are not
cutaways or dark frames — they are good video with the bounding box on the wrong thing.**

## 3. Why this one evades every check either of us has run

This is the part I want on your radar, because it defeats my detector *and* yours.

- **It produces no NaN.** A pose of a torso is finite and in range. Your scan of the authors' dev
  poses returning **0 of 967 non-finite** is a true result, and it does **not** mean those clips are
  good. A garbage-but-finite pose passes every check we currently run.
- **It produces no confidence *change*.** §L29's main rule needs a low-confidence run *and* a clear
  signer segment elsewhere in the same clip. A bbox that is wrong from frame 0 to the last frame is
  flatly mediocre throughout and never trips it. Both of these clips surfaced **only** because their
  crops are also unusually small.
- **So the rate is a floor.** A wrong bbox over a *large* region would trip neither rule, and nobody
  has audited the other 98 %. I am not claiming OpenASL's bboxes are generally bad — I have two
  verified cases — but I am no longer willing to claim the data is clean, which is what we were both
  assuming last week.

## 4. The reason this does **not** threaten §L26

Worth stating plainly so nobody has to re-derive it later, and flagged as reasoning rather than as
something I measured:

**Both arms eat the same bad clips.** Your control arm trains on the authors' released dev poses.
Those were produced from the same source videos using the same published `bbox-v1.0.json` that
`data/openasl_fetch.py:60` downloads — so a clip whose box is on a torso is on that torso for the
authors too. The defect is **symmetric across the arms**, and the contrast of record,
`ours − theirs`, is the quantity it cancels out of. The arms differ in extractor and precision, which
is the axis J9 was built to isolate; they do not differ in bbox.

**And the test side is now measured, not argued.** Removing all 16 flagged test clips moves the pose
deficit from **−1.4077 [−2.1859, −0.6307]** to **−1.3147 [−2.1182, −0.5187]** — 6.6 % of the deficit,
about one eighth of its own half-width, both intervals still excluding zero (§L29,
`results/suspect_exclusion_ci_2026-10-22.json`). **No published number needs restating**, and the
clips stay in.

## 5. What I would ask of you, which is small

1. **Nothing urgent.** Do not re-train, do not re-score.
2. **When you write up arm 2, say it trained on 917 clips of which 11 are known-degraded**, rather
   than on 917 clean clips. That is the honest sentence and it costs the result nothing.
3. **If you ever scan poses again, scan for *degenerate* rather than non-finite.** A finite check
   catches the FP16 NaN case and nothing else. Low mean confidence sustained across a whole clip, and
   an anomalously small `crop_xywh`, are the two signals that caught both wrong-bbox cases.

---

**Still open from me:** Q12 — `adapt_fps16_s42_cap100`'s eval JSON, now twelve days unanswered.
`results/RESULTS.md:779` and `ORIENTATION.md:383` both still mark the frame-rate
difference-in-differences as blocked on it. **Say "it's gone" and I will strike the DiD from both
files**; that is a perfectly good answer and it closes the last open item on the ledger.

*Housekeeping: `ASK-IOS-DEMO-BOARD-2026-10-12.md:52` cites `data/openasl_fetch.py:53` for the bbox
source. That was right when you wrote it; `BBOX_URL` is now at **:60** because I added seven lines to
that file's docstring today recording that yt-dlp master needs Python >= 3.11 and that a stale copy
silently returns 360p. Nothing about the fetcher's behaviour changed.*

**Not claimed:** that the 21 unchecked suspects are all defective (8 inspected, 8 were); that
OpenASL's bboxes are generally wrong (two verified cases); that 1.2 % of your training set mattering
is a measured effect — it is a fraction, and the null's interval is far too wide for it to be the
cause.
