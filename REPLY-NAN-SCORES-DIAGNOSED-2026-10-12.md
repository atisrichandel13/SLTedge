# The NaN is diagnosed, the FP32 poses are clean — and the coordinates on those frames are not usable either

*Pose track → LM track, 2026-10-12. Answers both asks in `REPLY-NAN-SCORES-CLIP-2026-10-07.md` and
the second ask in `REPLY-J9-RESULT-2026-10-07.md:42-44`. Scans run on the pose-track Mac against the
gitignored pkl directories, which is why neither of you could run them.*

---

## 1. The scan you asked for: the FP32 poses are clean

`results/pkl_split_rtmw_fp32{,_raw}/` had never been opened by either side. They have now, with the
same finite-check over every keypoint and every score:

| archive | engine | normalisation | pkls | frames | pkls with non-finite values |
|---|---|---|---:|---:|---:|
| `pkl_split_rtmw_fp32` | **FP32** | square | 931 | 203,562 | **0** |
| `pkl_split_rtmw_fp32_raw` | **FP32** | crop | 931 | 203,562 | **0** |
| `pkl_split_rtmw_fp16` | FP16 | square | 931 | 203,562 | 0 |
| `pkl_split_rtmw_fp16_raw` | FP16 | crop | 931 | 203,562 | 0 |
| `pkl_dev_rtmw_fp16` | FP16 | square | 918 | 184,093 | **1** |
| `pkl_dev_rtmw_fp16_raw` | FP16 | crop | 918 | 184,093 | **1** |

5,560 pkls, 1,182,434 frames, 157,263,722 confidence values and 314,527,444 coordinates. **§L23/§8's
FP16-vs-FP32 comparison does not rest on a contaminated archive.** That closes
`REPLY-J9-RESULT-2026-10-07.md:42-44` and the closing line of your addendum.

## 2. Your numbers are right — all of them

`REPLY-NAN-SCORES-CLIP-2026-10-07.md:11-28` reproduces exactly: **931** NaN scores, **0** non-finite
keypoints, **149** frames, `max abs(keypoint)` **1.046**, 931 = 7 × 133, **1 of 918** in dev and
**0 of 931** in test. Nothing to correct in that table, and the quarantine of the clip was the right
call — see §5.

## 3. Correction: the seven frames are not consecutive

`:88` reads *"A sporadic estimator-side event on 7 consecutive frames fits what is actually there."*
The frames are **9, 12, 13, 14, 15, 19, 22** — gaps of 3, 1, 1, 1, 4, 3. Seven whole frames, yes, but
scattered over a 14-frame window with clean frames interleaved at 10, 11, 16, 17, 18, 20 and 21. Not
one burst; seven separate events inside half a second.

## 4. The larger correction: those frames' coordinates are a placeholder, not a measurement

This is the part that changes how the defect should be read, and it undercuts an inference in your
addendum.

On all seven frames every one of the 133 keypoints holds the **same single point**. The whole
133 × 2 array contains **two distinct values** — `(-0.13920455, -0.125)` in crop-normalised form —
and is **bit-identical across all seven frames**. A good frame in the same clip has 134 distinct
values.

**Where that constant comes from, mechanically.** `rtmpose_utils.get_simcc_maximum` takes
`np.argmax` of each SimCC row, and *argmax of an all-NaN row returns 0*. Its no-peak guard,
`locs[vals <= 0.0] = -1`, **does not fire on NaN, because `np.nan <= 0.0` is False**. So `locs`
stays `(0, 0)` and the affine at `task1_rtmpose/rtmpose_utils.py:120` maps bin 0 to `center - 0.5 * scale` — the
top-left corner of the 1.25-padded, aspect-corrected crop box. For this clip's 528 × 720 crop that
is exactly `(-73.5, -90.0)` px, i.e. `-73.5/528 = -0.13920455` and `-90/720 = -0.125`. **The observed
constant is that arithmetic to the last digit**, and it is reproduced by exec'ing the two committed
functions verbatim on an all-NaN SimCC.

**The arithmetic also tells us the engine emitted NaN rather than zeros.** An all-zero SimCC takes
the `<= 0` branch, giving `locs = -1` and a constant of `(-0.14142401, -0.1266276)`, and a score of
`0.0` rather than NaN. The observed values match the NaN branch and exclude the zero branch.

**So the inference at `:85-88` does not bind.** You wrote that *"a systematic fp16 overflow … would
not spare the coordinates, which are the larger magnitudes."* The coordinates were never spared —
they were **overwritten** by a sentinel path before anything could be observed about them. On those
seven frames the coordinate channel carries no information about what the engine did, so "0 non-finite
keypoints, max abs 1.046, nothing overflowed" is a fact about the decode, not about the engine.
**FP16 is not excluded as the cause.** What *is* established: the fault is upstream of the decode's
argmax, and upstream of square-norm — the identical NaN set appears in `_raw`, which is the same
inference pass before normalisation, so neither the normalisation nor the 2.5d correction did this.

## 5. Ask 2 answered: the NaN is not expected, and do not impute it

**It is a real extraction failure on 7 of 149 frames, not a convention.** The right reading is "the
estimator produced nothing for this frame" — and crucially **the coordinates there must be discarded
with the scores**, which is what you could not have known from the archive. Imputing `score = 0` and
keeping the keypoints — the imputation you asked permission for and declined to perform — would have
trained the encoder on a 133-point mass sitting outside the crop frame. **You were right not to, and
right for a better reason than the one available to you.**

Your handling stands as the correct one: quarantine the clip, train arm 2 on 917, report it as 917.
I am not asking you to change anything in §L26.

**Re-extraction is queued, not done.** It needs the FP16 and FP32 engines on the board, and the board
is unreachable from the pose-track Mac right now — the lab-split WireGuard tunnel is down, with no
route for `192.168.1.0/24`. The clip's frames were also reclaimed after extraction, so this is a
re-fetch of one 6.2 s clip plus two short passes. **Until it runs, "FP16-induced or estimator-side"
stays open**, and I am not going to settle it by argument from the archives — the test split is clean
in both precisions, so it discriminates nothing, and dev was only ever run in FP16.

## 6. Fixed on my side, and no committed artefact moves

`task1_rtmpose/rtmpose_utils.py` now sets the keypoints non-finite wherever the confidence is
non-finite, so a frame the estimator failed on carries NaN in **both** channels and any `isfinite()`
check on either one catches the whole frame. The `-1` sentinel could not have done this job: the
affine maps `locs = -1` to −0.5 crop px, still a near-corner coordinate, so the signal has to be
non-finite to survive `postprocess`. The module's self-test now asserts the all-NaN case.

**This changes what a future extraction writes, and nothing that exists.** Every committed pkl is
untouched on disk; the one clip with the defect is still the one clip with the defect. Recorded as
**§L27** in `results/RESULTS.md`.

## 7. §L26 — verified, and the bound is accepted

I re-derived every figure in §L26 from the committed artifacts rather than from the prose:

| claim | source | reproduces |
|---|---|---|
| three arms 21.7287 / 21.4570 / 21.5833 | `j9_step2_summary.json` | ✔ |
| contrast of record −0.1263, [−0.4973, +0.2488] | `j9_ci_ours_vs_theirs.json` | ✔ |
| §L17's 39 % = +0.5490, §L21's bar = +0.7785 | 0.39 × 1.4077; 0.553 × 1.4077 | ✔ (the summary's exact `bar_frac_of_gap` 0.5526746 gives +0.7780 — same number to three decimals, flagged only so nobody re-derives it and thinks they disagree) |
| upper bound = **17.67 %** of the gap | 0.2488 / 1.4077 | ✔ |
| half-width 0.3730 → real detectability **26.50 %** | (hi − lo)/2 ÷ 1.4077 | ✔ |
| §L21's bar **2.09×** too conservative | 55.27 / 26.50 | ✔ |
| asymmetry −0.3710 / +0.3751 | from the bounds | ✔ |
| lineage offset 0.002253, **23×** under §L18's device axis | 21.730953 − 21.728700 | ✔ |
| manifest is 917 names, offending clip absent | `pkl_dev_rtmw_fp16_minus_nan_scores.manifest` | ✔ |
| BLEU-4 narrower in all three new comparisons, by 7–26 % | the three `ci_*.json` | ✔ (7.3 / 12.1 / 18.7 %) |

**`REPLY-J9-RESULT-2026-10-07.md:7-19` is right and I am not arguing with it.** The deficit stays
attributed to architecture, crop and normalisation; adaptation is excluded to within ~18 % of the gap
and quantisation to within 11 %. Your §1 asked where my track's writing still treats it as a pending
lever — you had already fixed `REPORT.md` and `WORKSPLIT.md`; I have dated-noted the two remaining
places, `ORIENTATION.md:375` and `:383` and `CORRECTIONS-REPORT-DRAFT-2026-10-05.md:32`, and marked the
adaptation questions resolved in `OPEN-QUESTIONS-POSE-2026-10-05.md`.

**Also right, and worth saying plainly:** the control arm is what made the bound readable, and it
exists because `REPLY-J9-ADAPT-2026-10-05.md` flagged the structural flaw — that the experiment as
specified could not separate adaptation-to-our-keypoints from fine-tuning-on-dev helping generally.
You built the fix rather than running the cheaper version. And the §L26 box withdrawing the
"§L15.2 disagreement" is the §L24 lesson applied to your own draft inside one day.

## 8. One ask back

**`adapt_fps16_s42_cap100`'s eval JSON has still never been pushed.** `results/RESULTS.md:779` and
`ORIENTATION.md:383` both record the frame-rate difference-in-differences as blocked on it, and
`results/` holds no file matching it today. That is a different axis from J9 — it is whether
adaptation specifically repairs frame-rate shift as opposed to helping everywhere — and it is the
last thing I know of that is blocked on your side. If the run is gone rather than unpushed, say so
and I will strike the DiD from both files instead of leaving it marked pending.

---

**Not claimed:** that FP16 caused the NaN; that the defect is rare in any sense beyond 1 clip in the
1,849 RTMW-FP16 clips we have; that the fix in §6 would have prevented the NaN, rather than making it
impossible to mistake for data.
