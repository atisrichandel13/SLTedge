# Open questions from the pose track

> **Status 2026-10-05, after `ANSWERS-Q1-Q7`, `REPLY-Q2-SCALE` and `ec7371f`:** Q1a, Q1b, Q3a, Q3b,
> Q4, Q5, Q7, Q8 **all resolved** (see inline). Q2 **decided by Tushar — J9 proceeds**, with their
> probe (`colab_probe_scale.py`) accepted in parallel. Q6 **deferred to after the project**, no answer
> wanted.
>
> **Nothing in this file is waiting on the LM track.** Everything left is mine to execute: the dev
> fetch, then board extraction, then scp'ing the two 400-clip pose dirs across per Q3b. If you need
> something from me, add it here and I will see it on the next sync.

*2026-10-05. One place for everything I need from the LM track, so it is not scattered across reply
docs. Numbered for reference. **Q1–Q3 block work that is running or about to run.***

---

## Blocking now

### Q1a / Q1b — RESOLVED, both inside your bars (interim, 449 of ~966 clips fetched)

Your 22.1 % reproduces exactly on the pose-track Mac: **206 of 931** test clips are ≤24 fps native.
And **you were right and I was wrong on Q1b** — `min(1.0, target_fps / src)` means a clip at or below
the target is not thinned at all, so the native mix changes *how much* subsampling is applied even
though it does not change the resulting rate. I had that backwards.

Interim dev tallies, to be re-run at completion:

| | test (931) | dev (449 so far) | your bar |
|---|---:|---:|---|
| ≤24 fps native — not thinned at `--fps 24` | **22.1 %** (206) | **18.9 %** (85) | "within a few points → record and move on" |
| source height 720p | **76.3 %** (710) | **76.8 %** (345) | "accept unless below ~60 %" |

Both pass. I will push the final histograms when the fetch ends; nothing here needs a decision.

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
