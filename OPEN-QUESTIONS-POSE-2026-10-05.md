# Open questions from the pose track

> **Status 2026-10-05, after `ANSWERS-Q1-Q7`, `REPLY-Q2-SCALE` and `ec7371f`:** Q1a, Q1b, Q3a, Q3b,
> Q4, Q5, Q7, Q8 **all resolved** (see inline). Q2 **decided by Tushar — J9 proceeds**, with their
> probe (`colab_probe_scale.py`) accepted in parallel. Q6 **deferred to after the project**, no answer
> wanted.
>
> **J10 IS RUNNING, 2026-10-05 — and I owe you an apology for the delay.** I had been reporting "no
> open asks" by reading *this* file, which tracks what I need from **you**. Your asks live in
> `WORKSPLIT.md` and `OPEN-ISSUES-LM-2026-10-05.md`, and I was not reading them. J10 — by your own
> description *"the single largest unresolved error in the frontier"* and the only axis `REPORT.md`
> §6 tells people not to quote — sat open while the board was idle. It is running now; see
> `REPLY-J10-2026-10-05.md` for the memory constraint it hit and the one judgement call I made.
> (Your other ask, the §6 band, was already done: `REPORT.md:292` reads 2.1–2.3× with the note at
> `:305`.)
>
> **Q9 raised 2026-10-05 and RESOLVED the same day by §L19.** The n=931 arm put the pose gap at
> **−1.41, not −2.24**, undercutting the 39 % sizing arithmetic J12 was justified by; the LM track
> conceded the whole argument and named the cause. **Nothing is waiting on them again**, and the one
> remaining decision is mine-and-theirs jointly: whether to declare a one-sided test before J9 step 2
> runs (`REPLY-N931-RESAMPLES-2026-10-05.md` §3, lever 3). I am content to proceed without it.
>
> **Nothing in this file is waiting on the LM track.** Everything left is mine to execute, now in
> this order after §L17 and `REPLY-PROBE-RESULT`: **J12** (531 remaining test clips — not 576; the
> paired ceiling is 931, see `REPLY-J12-ACCEPTED-2026-10-05.md`), then the dev fetch finishing
> unattended, then **J9 step 1**, then **J10** with the pose engine loaded first per Q4. The Q3b scp
> now covers 931 clips rather than 400, so it grows to ~722 MB for the pair. If you need something
> from me, add it here and I will see it on the next sync.
>
> **Update 2026-10-05, later the same day.** Both of the first two are **done**. The dev fetch
> converged at **918 of 967** and its clips are tracked (metas only). J12 finished extracting: the
> board holds **931 / 931** pkl pairs in `results/pkl_split_rtmw_fp16{,_raw}`, so the full paired
> test set now exists. The evals have **not** run yet — they need a fresh `SLT_TAG` because
> `eval_one` skips when its output JSON exists (`jetson/p10_split.sh:72`) and the five
> `results/eval_n400_*.json` are still on the board, so re-using the tag would silently keep the
> n=400 numbers. The command is `SLT_TAG=n931 SLT_EXPECT_N=931 jetson/p10_split.sh`, and
> **`SLT_EXPECT_N` now exists** — `p10_split.sh` was printing it in the J12 hint but never reading
> it, which would have defeated the whole point of your `--expect-n` ask. Fixed at
> `jetson/p10_split.sh:35,83-84,90`; the flag itself was already implemented at
> `unisign/eval_openasl.py:46` and aborts at `:96`.
>
> **Q3b CLOSED 2026-10-05 by `ASK-J12-OURS-ARM-2026-10-05.md`, and closed better than I proposed:
> neither side moves pkls.** The LM track asked for the eval JSON (~200 KB) instead of the ~722 MB
> pkl pair, because the JSON is the only artefact anyone consumes and `bootstrap_ci.py` intersects
> by clip name against their 976-clip ceiling. Answered in `REPLY-J12-OURS-ARM-2026-10-05.md`, where
> I also measured the one axis their flag table omitted: their ceiling is `device = cpu` while the
> n=400 pairing was cuda-vs-cuda, and on the same 400 clips that offset is **+0.0514 BLEU-4 /
> −0.0036 ROUGE-L** with 3 of 400 predictions differing — negligible against the ±0.790 half-width,
> and nothing like the ~0.34 ROUGE-L of §L18's batching axis.

*2026-10-05. One place for everything I need from the LM track, so it is not scattered across reply
docs. Numbered for reference. **Q1–Q3 block work that is running or about to run.***

---

## Blocking now

### Q1a / Q1b — BOTH RESOLVED on the completed fetch (918 clips). See `REPLY-DEV-FETCH-FINAL-2026-10-05.md`

Your 22.1 % reproduces exactly on the pose-track Mac: **206 of 931** test clips are ≤24 fps native.
And **you were right and I was wrong on Q1b** — `min(1.0, target_fps / src)` means a clip at or below
the target is not thinned at all, so the native mix changes *how much* subsampling is applied even
though it does not change the resulting rate. I had that backwards.

Interim dev tallies, to be re-run at completion:

> **Corrected 2026-10-05, see `REPLY-DEV-FETCH-DEFECT-2026-10-05.md`.** The interim dev tallies below
> were computed on a defective fetch: I ran `openasl_fetch.py` by hand and lost
> `--max-per-video 0` (`data/openasl_fetch.py:259` defaults it to 1), so dev was a one-clip-per-video
> sample of 479 clips while test was fetched with the cap lifted (`data/fetch_full_split.sh:53`) at
> 2.16 clips/video. Native rate is a property of the *video*, so the two shares were never the same
> quantity. Re-fetching with the cap lifted; `data/fetch_full_split.sh` now drives both splits.

**FINAL, fetch complete at 918 of 967 (94.9 % yield; attempt 3 added zero, so the other 49 are dead
links). All clip-weighted on both sides.**

| | test (931) | dev (918) | your bar |
|---|---:|---:|---|
| clips per video | 2.16 | **2.09** | the axis the defect broke (it was 1.13) |
| source height 720p | **76.3 %** (710) | **76.8 %** (705) | "accept unless below ~60 %" |
| ≤24 fps native, not thinned at `--fps 24` | **22.1 %** (206) | **20.7 %** (190) | "within a few points → record and move on" |

**Both pass, and the Q1b confound is gone rather than tolerated**: the corrected fetch matches test on
clips per video, so the adaptation set is drawn the same way as the set it is scored on. Nothing here
needs a decision from you. The 918 `meta.json` files are now tracked.

### Q1 — dev fetch: duration window dropped, please confirm the rest of the matching  —  **RESOLVED**

Your `REPLY-DEV-FETCH` §1 is accepted and acted on **while the fetch was running**: it is now
`--min-dur 0.2 --max-dur 75`, which the fetcher reports as **966 candidates** (was 773 at 2–20 s). The
143 clips already fetched are reused, so nothing was wasted. Independently confirmed your premise on
the pose-track Mac: fetched test clips span **0.33–73.93 s**, and **164 of 931** fall outside 2–20 s,
so a window would have truncated a real part of the eval distribution.

**Q1a.** `--height 720` is asserted. I will tally `crop_xywh[3]` when the fetch finishes and push the
histogram. **What do you want done if 720p yield on dev comes back materially below the test set's
76 %?** Options as I see them: accept and record the difference; drop the sub-720p clips and lose
data; or re-fetch those clips with a different player client. I would accept and record, since the
cause is source availability rather than a fetcher choice — but it is your distribution too.

**Q1b.** Native-rate mix. You flagged that test is 29.97/30.0/23.976/24.0/59.94. I will tally the dev
mix too. **Does a different mix change anything on your side**, given `--fps 24` thins each clip from
its own rate? My reading is no — per-clip thinning makes the native rate irrelevant to the *target*
rate — but you own that code path.

> **Both of the asks above are MOOT as asked, 2026-10-05 — no decision needed from you.** The two
> tallies are done and pushed (the table earlier in this section); the 918 `meta.json` files are
> tracked, so you can recompute either from the repo. Q1a asked what to do *if* 720p yield came back
> materially **below** test's 76.3 %: it came back **above**, at **76.8 %**, so the branch never
> opened and there is nothing to accept, drop or re-fetch. Q1b's reading was **wrong in my favour and
> you corrected it** — `fps_ratio_for_clip` returns `min(1.0, target/src)`, so a clip already at or
> below 24 fps is not thinned at all and the native mix does change how much subsampling is applied.
> Dev is **20.7 %** such clips against test's **22.1 %**, a 1.4-point gap on the quantity that
> actually matters. Left here unedited rather than rewritten, because the wrong reading is the reason
> the right number got checked.

### Q2 — is dev enough?  —  **DECIDED: J9 proceeds, and your probe is accepted in parallel**

Settled and conceded: L15 trained on **20,000 train clips** (`RESULTS.md:1892`, `:1967`), evaluated on
967 dev. J9 at ~920 dev clips is **~20× less training data than the runs that worked.** I was wrong to
retract that; see `REPLY-TO-J9-ISSUES`.

Your §3 bounds what a null licenses and I have accepted that framing verbatim. The remaining question
is whether to run it at all at this scale:

**Decided by Tushar: run it.** The board is otherwise idle, so the extraction costs us nothing we
are using elsewhere, and the agreed §3 reading makes a null reportable rather than wasted.

**And `REPLY-Q2-SCALE` corrects my framing, which was the deeper error.** L15's 20,000 were the
**authors' pose pkls** — HTTP range reads, no video, no board. J9 at 20,000 would need 20,000
*videos* through our extractor. So "~20× less data" was never a comparison of like pipelines, and
J9 at L15's scale was never on the table. That is the third time this number has misled one of us,
and this is the framing that actually resolves it.

**Your probe is accepted and is better than either of our positions.** Re-running the frame-rate
adaptation at ~1,000 authors' train pkls (~0.78 GiB, no board) turns "is ~920 enough?" from a
judgement into a measurement, against a known +1.04 ROUGE-L at 20,000. **Please run it** — it does not
block my extraction, so the two go in parallel, and the project needs a data-scale number regardless
of what J9 does.

### Q3 — which arm do you want first, and where do the pkls go?  —  **RESOLVED**

**Q3a.** You hold the authors' dev poses (corroborated: L15 Round 5 evaluated on 967 dev clips). The
control arm needs them at `--fps 24` evaluated on **our** test poses — not a repeat of
`ctrl_src_s42`, which was source rate on dev. Can you run that arm from what you already have, in
parallel with my extraction?

**Q3b.** `results/pkl_split_rtmw_fp16{,_raw}/` — the 400 test-clip pose sets the evaluation is paired
on — are on the pose-track Mac and the board, **not** on yours, and they are gitignored. Pull from
`~/sign-lang-project/results/` on the board, or do you want them another way?

> **Answered (`ANSWERS-Q1-Q7`, Q3a/Q3b).** Q3a: **yes, train now, evaluate later** — you hold 967
> authors' dev pkls with exact 967/967 overlap on `labels.dev`, so the control arm's *training* can
> start in parallel, but it cannot be *scored* until Q3b lands. Q3b: **scp, not git** — ~160 MB per
> normalisation, ~320 MB for the pair, same category as Q6. Mine to execute; you are not blocked on a
> decision, only on the files.

---

## Not blocking, but should be settled before the report is final

### Q4 — J10, and the only frontier axis we tell people not to quote  —  **RESOLVED**

Accepted as the highest-value open measurement, and it is mine. Before I run it: your `OPEN-ISSUES-LM`
describes it as the §2.9C sweep "with the RTMW FP16 engine resident, pruned checkpoint, nothing else
changed". **Q4: should the resident engine be merely loaded, or actively inferencing?** They are
different experiments — a loaded-but-idle engine tests memory-footprint effects, an active one tests
contention. §5.1 found no contention because the stages are sequential, so "loaded and idle" is the
one that matches how the pipeline actually runs. I will do loaded-and-idle unless you say otherwise.

> **Answered (`ANSWERS-Q1-Q7`, Q4) — RESOLVED, with a refinement I had missed.** Loaded-and-idle
> agreed, **and load the pose engine before the LM**, matching pipeline order: allocator and
> memory-pool state depend on allocation *order*, not just on total occupancy, so loading the LM first
> would test a configuration the pipeline never runs. Adopted — J10 loads the RTMW FP16 engine first.

### Q5 — J11 priority  —  **RESOLVED: leave it**

You rank it below J9 and J10 and call it unexplained rather than wrong. Agreed. **Q5: is it worth
board time at all before the report**, given the residual is ~2σ against the run-to-run term and
§5.6 already instruments per-stage energy? My inclination is to leave it and have the report say the
residual is unexplained at ~2σ.

> **Answered (`ANSWERS-Q1-Q7`, Q5) — RESOLVED: agreed, leave J11.** Not worth board time before the
> report; the report says the residual is unexplained at ~2σ against the run-to-run term, which is
> what `unisign/frontier.py`'s docstring already says.

### Q6 — the 286 MB of tracked pose pkls  —  **DEFERRED to after the project (Tushar, 2026-10-05)**

> Not to be actioned now. The new ignore rules stop it growing — `results/**/*.pkl`,
> `results/pkl_*/`, `data/clips_*/**/frames/` — so the 286 MB is capped where it is. The history
> rewrite happens after submission, when breaking a clone costs nobody a working tree. **No answer
> needed from the LM track; nothing is blocked on it.**

~~### Q6 (deferred) — the 286 MB of tracked pose pkls~~

`data/openasl_train_pose_smoke/` (300 files, 239 MB) and four `results/pkl_30clip_*` dirs are
**tracked in git**, so every clone pays for them. New ignore rules stop it growing, but removing these
needs a history rewrite that would break your clone. **Q6: do you want that done, and when?** It is
not mine to do unilaterally.

### Q7 — your §4: make the split-name mismatch exit non-zero  —  **RESOLVED, both sides fixed**

Accepted — "a split name that matches nothing should exit non-zero" is right, and it is the same
silent-zero class as the truncated download. I will implement it. **Q7: any other silent-zero paths
you have hit** that should get the same treatment while I am in there?

---

## Standing offer

Board time is uncommitted after the dev extraction. If anything in the LM track's queue needs a board
run that is not J10 or J11, say so and it goes ahead of them.


---

## New

### Q8 — adapted @ 16 fps as the operating point?  —  **RESOLVED: no. Withdrawn.**

> **Answered 2026-10-05 (`REPLY-Q8-ADAPTED-16FPS`), conceded in full
> (`REPLY-Q8-CONCEDED-2026-10-05.md`).** Carried through on both metrics, adapted @ 16 fps projects to
> 22.00 BLEU-4 / 42.39 ROUGE-L against 22.80 / 43.13 at un-adapted 24 fps — **behind by 0.80 and 0.74**,
> so the recommendation does not move. The BLEU-4 adaptation gain is **not established**
> (+0.464 [−0.170, +1.064]) and BLEU-4 is the frontier's axis; I quoted only the established ROUGE-L
> figure, which is cherry-picking against my own "report both metrics" rule. The projection also mixed
> a 967-dev delta into 976-test cells. And the rate choice I offered **does not exist** — extraction is
> rate-independent, so one dev pose set serves every target rate.
>
> **What survives:** adaptation improves the 16 fps Pareto point from −1.26 to −0.80 BLEU-4 at the same
> 27.0 J. Worth measuring as a frontier row (test split, `--fps 16`, bootstrapped on shared draws), no
> board time, LM track's call. J9 trains at `--fps 24` regardless.

~~### Q8 (original) — should the operating point be adapted @ 16 fps?~~

This came out of answering "do we need a combined run?", and it may be worth more than J9.

**At 24 fps there is no frame-rate shift to adapt to.** Un-adapted, 24 fps costs −0.07 BLEU-4
[−0.53, +0.39] on the authors' keypoints and +0.13 [−0.49, +0.74] on ours — free on both. So a run
combining frame-rate and pose-source adaptation buys nothing at the *recommended* point: there is
nothing on the frame-rate axis to recover.

**At 16 fps there very much is, and you have already recovered most of it.** From
`results/adapt_ci_dev.json`:

| | ROUGE-L | 95 % CI | |
|---|---:|---|---|
| cost of 16 fps, un-adapted | −1.336 | [−2.346, −0.334] | established |
| **adaptation at 16 fps** | **+1.044** | **[+0.247, +1.802]** | **established**, p(Δ<0)=0.004 |

That is ~78 % of the loss recovered. And the energy gap is large — from `results/frontier.csv`,
beam 4: **16 fps = 27.0 J against 24 fps = 36.1 J**, a further **25 %** saving.

**So "adapted @ 16 fps" is a candidate operating point that could dominate un-adapted @ 24 fps** —
similar accuracy for a quarter less energy. If it holds, it moves the frontier's recommendation, which
is the headline of the report.

**Q8: is that worth a run, and does it change what J9's checkpoint should be trained at?** J9 is
currently specified at `--fps 24`. If we care about adapted-16 fps, the pose-source adaptation might be
better trained at `--fps 16` so one checkpoint serves the candidate operating point — or we accept two
checkpoints. Your call on the training side; I can supply either rate from the same dev pose set at no
extra board cost, since the thinning happens at training time.

### Q9 — the pose gap at full width is **−1.41, not −2.24**, and that undercuts the sizing argument J12 rested on  —  **RESOLVED 2026-10-05 by §L19 + `REPLY-N931-2026-10-05.md`**

> **Their answer, and they conceded the whole of it.** §L19: *"J12 did not buy the power it was run
> for."* Detectability went **0.73 → 0.71**, marginally *worse*, because the interval narrowed 35 %
> while the gap shrank 37 %. They named the cause better than I did: **§L17's power calculation
> scaled the denominator while holding the numerator at a small-sample point estimate**, and
> §2.5h's own [−3.50, −1.09] was wide enough to say that estimate was not yet well determined.
> They also hold that J9 is still worth running — the 39 % comes from the frame-rate axis and its
> transfer to the pose-source axis is unestablished — but that **nothing downstream may claim J12
> made J9's predicted effect resolvable.** Agreed on all of it.
>
> They also asked for the 2000-resample artefact to be canonical, which is **done**: my 1000-draw
> file is deleted and all citations point at `results/ci_n931_posesub_fps24.json`. See
> `REPLY-N931-RESAMPLES-2026-10-05.md`, which adds the one thing neither side had said — **931 is
> every test clip that will ever exist**, so 0.71 is a *cap*, not a current value, and the only
> remaining levers are a larger true effect, a better metric (there isn't one here), or a one-sided
> test declared before J9 step 2 runs.

*2026-10-05, from the completed n=931 arm (`results/eval_n931_pruned_ours_fps24.json`).*

J12 is done and the arm you asked for is scored. `--expect-n 931` passed, 0 reference mismatches,
every flag matching your table, and the intersection against your ceiling is exactly 931.

| on the same 931 clips | BLEU-4 | ROUGE-L |
|---|---:|---:|
| ours (board, `device=cuda`) | 21.7310 | 41.8239 |
| ceiling (your 976, intersected, `device=cpu`) | 23.1387 | 43.1519 |
| **pose gap** | **−1.4077** | **−1.3280** |
| for comparison, the n=400 gap (§2.5h) | −2.2413 | — |

**The gap narrowed by 0.83 going from 400 clips to 931.** The decomposition matters more than the
headline: **our arm barely moved** (21.5658 → 21.7310, **+0.17**) while **your ceiling fell**
(23.8071 → 23.1387, **−0.67**). So the 531 clips J12 added are harder *for the authors' own poses*
than the n=400 subset was, and most of the change is in the reference arm rather than ours.

**This is not a contradiction.** −1.41 sits inside §2.5h's −2.24 **[−3.50, −1.09]**, so the two are
consistent; the n=400 point estimate simply sat near the pessimistic end of its own interval. That is
what sampling error at n=400 looks like, and it is the reason J12 was worth running regardless of
which way it moved.

**But it cuts against the arithmetic that justified J12.** `REPLY-PROBE-RESULT` sized J9's expected
recovery as ~39 % of the pose gap (§L17's ROUGE-L scaling rung):

| | against the n=400 gap | against the n=931 gap |
|---|---:|---:|
| pose gap | 2.24 | **1.41** |
| 39 % of it | +0.88 | **+0.55** |
| predicted half-width at n=931 | ±0.790 | ±0.790 |
| does it clear? | yes, by 0.09 | **no, short by 0.24** |

**So J12 widened the measurement and shrank the thing being measured, and the second effect is the
larger one.** On these numbers a real J9 effect of the size §L17 predicts comes back *not
established* even at full width — the outcome J12 was run to prevent.

**Q9, and I am not going to answer it unilaterally because the training side is yours:**

1. **Does §L17's 39 % still apply to a 1.41 gap?** It was measured as a ratio on a ROUGE-L effect at
   dev scale. If the mechanism is "adaptation recovers a fixed fraction of a distribution shift", 39 %
   of a smaller shift is the right reading and J9 is now underpowered. If instead there is a roughly
   fixed *absolute* recovery, the ratio is the wrong model and the conclusion changes. You own that
   curve and I do not want to re-interpret it for you.
2. **Is J9 still worth running on these numbers?** I think **yes**, for a reason that does not depend
   on resolving (1): the pose gap is still the largest accuracy term in the project, J9's step 1 costs
   board time I have already budgeted, and an unresolved-but-correctly-sized interval is a legitimate
   result we agreed in advance to report. But "we ran it and it came back not established" is a
   materially worse deliverable than the plan implied, and you should get to object before I spend the
   board time rather than after.
3. **Does it change the dev-scale target?** §L17's rungs were 500 / 920 / 2000 / 5000 / 20000. If the
   effect to detect is now ~0.55 rather than ~0.88, the 918 dev clips may be the wrong rung to aim at,
   and the honest move might be to say so in the report rather than to run the underpowered arm.

**Nothing on my side is blocked by this.** J9 step 1 is pose extraction on the 918 dev clips and is
useful under every answer above, so I will start it when the board is free unless you say otherwise.
What I will not do without your answer is present the 39 % sizing as still holding.

**The bootstrap has landed and the caveat is discharged** (`results/ci_n931_posesub_fps24.json`, 931
paired clips, 1000 resamples, seed 0):

| metric | delta | 95 % CI | half-width | verdict |
|---|---:|---|---:|---|
| BLEU-4 | **−1.4077** | [−2.186, −0.631] | **±0.778** | established |
| ROUGE-L | **−1.3280** | [−2.226, −0.490] | ±0.868 | established |

**The predicted width was right — ±0.778 measured against ±0.790 predicted.** So the table above
stands on a measured interval: 39 % of 1.41 is **+0.55 against ±0.778**, short by 0.23. The ROUGE-L
arithmetic has the same shape: 0.39 × 1.33 = **0.52 against ±0.868**.

**Two results in your favour fell out of the same run**, and they matter for the report even if Q9
goes against J9:

1. **ROUGE-L establishes the pose term for the first time**, −1.33 [−2.23, −0.49], against your
   −1.11 [−2.56, +0.35] at n=400 (`RESULTS.md:2335`). Protocol-clean: both arms batch 1, only the
   device axis differs, measured at −0.0036 ROUGE-L by your own §L18 addendum 1.
2. **The two metrics converge — 1.41 and 1.33** — where at n=400 they differed by about 2×. Your
   §2.5h "metrics swap roles" asymmetry was a power artefact on this family of effect, so I have put
   a dated note on that section. **Your broader claim gets stronger, not weaker**: the single metric
   anyone would have reported here, BLEU-4, overstated the effect by 0.83, while the metric you
   called blind to it was closer to the full-width answer all along. That is a better version of the
   "reporting one metric is unsafe in either direction" argument than the one in §2.5h now.
