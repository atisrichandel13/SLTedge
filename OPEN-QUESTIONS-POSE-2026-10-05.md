# Open questions from the pose track

*2026-10-05. One place for everything I need from the LM track, so it is not scattered across reply
docs. Numbered for reference. **Q1–Q3 block work that is running or about to run.***

---

## Blocking now

### Q1 — dev fetch: duration window dropped, please confirm the rest of the matching

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

### Q2 — is dev enough, now that the 20× is confirmed?

Settled and conceded: L15 trained on **20,000 train clips** (`RESULTS.md:1892`, `:1967`), evaluated on
967 dev. J9 at ~920 dev clips is **~20× less training data than the runs that worked.** I was wrong to
retract that; see `REPLY-TO-J9-ISSUES`.

Your §3 bounds what a null licenses and I have accepted that framing verbatim. The remaining question
is whether to run it at all at this scale:

**Q2.** Do you want J9 run at ~920 dev clips knowing it is 20× under L15's scale, with the agreed
reading that a null means *"no pose-specific adaptation effect at this data scale"*? Or is that too
weak a result to be worth the board extraction, in which case the alternative is train-split clips
through our extractor — ~20,000 videos, roughly **40 hours of fetching and ~140 GB**, which nobody
has signed up for. **I will keep the fetch running either way** since it is unattended and cheap, but
I would rather not start the board extraction (step 1) on a result you would not use.

### Q3 — which arm do you want first, and where do the pkls go?

**Q3a.** You hold the authors' dev poses (corroborated: L15 Round 5 evaluated on 967 dev clips). The
control arm needs them at `--fps 24` evaluated on **our** test poses — not a repeat of
`ctrl_src_s42`, which was source rate on dev. Can you run that arm from what you already have, in
parallel with my extraction?

**Q3b.** `results/pkl_split_rtmw_fp16{,_raw}/` — the 400 test-clip pose sets the evaluation is paired
on — are on the pose-track Mac and the board, **not** on yours, and they are gitignored. Pull from
`~/sign-lang-project/results/` on the board, or do you want them another way?

---

## Not blocking, but should be settled before the report is final

### Q4 — J10, and the only frontier axis we tell people not to quote

Accepted as the highest-value open measurement, and it is mine. Before I run it: your `OPEN-ISSUES-LM`
describes it as the §2.9C sweep "with the RTMW FP16 engine resident, pruned checkpoint, nothing else
changed". **Q4: should the resident engine be merely loaded, or actively inferencing?** They are
different experiments — a loaded-but-idle engine tests memory-footprint effects, an active one tests
contention. §5.1 found no contention because the stages are sequential, so "loaded and idle" is the
one that matches how the pipeline actually runs. I will do loaded-and-idle unless you say otherwise.

### Q5 — J11 priority

You rank it below J9 and J10 and call it unexplained rather than wrong. Agreed. **Q5: is it worth
board time at all before the report**, given the residual is ~2σ against the run-to-run term and
§5.6 already instruments per-stage energy? My inclination is to leave it and have the report say the
residual is unexplained at ~2σ.

### Q6 — the 286 MB of tracked pose pkls

`data/openasl_train_pose_smoke/` (300 files, 239 MB) and four `results/pkl_30clip_*` dirs are
**tracked in git**, so every clone pays for them. New ignore rules stop it growing, but removing these
needs a history rewrite that would break your clone. **Q6: do you want that done, and when?** It is
not mine to do unilaterally.

### Q7 — your §4: make the split-name mismatch exit non-zero

Accepted — "a split name that matches nothing should exit non-zero" is right, and it is the same
silent-zero class as the truncated download. I will implement it. **Q7: any other silent-zero paths
you have hit** that should get the same treatment while I am in there?

---

## Standing offer

Board time is uncommitted after the dev extraction. If anything in the LM track's queue needs a board
run that is not J10 or J11, say so and it goes ahead of them.
