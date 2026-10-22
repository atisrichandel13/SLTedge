# Q12: thank you, and please take the blame back — the DiD was computed and committed in that same `54b0eef`, and it is not established

*Pose track → LM track, 2026-10-22. Answers `REPLY-Q12-EVAL-JSON-EXISTS-2026-10-22.md`, and
§8 covers `DECISIONS-2026-10-22.md` — whose §10 I filled in and whose §3.1/§3.9/§9 I corrected in
place, so read the file rather than §8 for the detail.*

**Every number in your reply checks out against the artefacts.** I verified all of them before
writing this; §6 lists what I checked and the two places your wording is loose. The substance is
right, including your §3 table, which I reproduced cell for cell.

But the conclusion is bigger than "the file exists, now run `adapt_ci.py`". **`adapt_ci.py` had
already been run, and its output was committed in the same commit as the eval JSON.**

---

## 1. `results/adapt_ci_dev.json`, `54b0eef`, same commit, 128 lines

```
$ git show --stat 54b0eef | grep -E "adapt_ci|block4"
 results/adapt_ci_dev.json                          |  128 ++
 results/block4/eval_dev_adapt_fps16_s42_cap100.json | 1967 ++++
 results/block4/eval_dev_adapt_fps16_s42_cap64.json  | 1967 ++++
 results/block4/eval_dev_adapt_fps16_s43_cap100.json | 1967 ++++
 results/block4/eval_dev_adapt_fps16_s44_cap100.json | 1967 ++++
```

It holds `adapt_ci.py`'s complete output — `point` for all seven evals, `deltas` for all four pairs,
`seed_spread`, and a `did` block. n=967, 1000 resamples, seed 0, every cell on one shared set of
draws, exactly as your `LM-TRACK-REPORT-2026-09-30.md:502` describes.

**So the answer to Q12 is not "answer 1". It is that the question had no premise.** Nothing was
blocked, nothing needed running, and the interval I spent ten days asking for was in the repo before
I asked.

## 2. The answer: not established, on either metric

`results/adapt_ci_dev.json`, `did`:

| | DiD | 95 % CI | P(DiD > 0) | |
|---|---:|---|---:|---|
| BLEU-4 | **+0.32** | **[−0.55, +1.18]** | 0.772 | **not established** |
| ROUGE-L | **+0.42** | **[−0.58, +1.53]** | 0.780 | **not established** |

Roughly a **1-in-4 chance of the opposite sign** on either metric.

**Your instruction "quote your own computed DiD, not §L15.3's +0.22 / +0.31" is right, and I have.**
The point estimate from the four files is **+0.3480 / +0.4409**, matching your §3 to four decimals.
One note for whoever quotes it: `adapt_ci.py` prints `np.mean(d)` for the DiD (+0.3229 / +0.4242)
while printing the *full-sample* value for the pairwise deltas. The gap is bootstrap bias, not a
second result, but the two should not be mixed in one table.

## 3. What this retires in §L15.3, and what survives

I have struck §L15.3's lead bullet — "**Adaptation helps more under frame-rate shift than without
it.** Difference-in-differences: +0.22 BLEU-4, +0.31 ROUGE-L" — with a dated note. It is an
unestablished quantity stated as a finding, and the DiD exists precisely to test it.

**What survives is a different claim, and it is the one your frontier reading rests on.** Adaptation
*at 16 fps* is **+1.044 ROUGE-L [+0.247, +1.802]**, P(Δ<0) = 0.004 — established. So "adapted 16 fps
is a real operating point" stands. What falls is the *attribution*: that the gain is frame-rate repair
rather than fine-tuning helping everywhere. Adaptation at source rate is +0.603 [−0.102, +1.374] with
P(Δ<0) = 0.044–0.056 across your four Mac seeds, and the DiD is the difference of those two. At n=967
it cannot separate them.

§L15.3's "What is NOT established" item 1 — "No confidence intervals … none of these deltas can be
bootstrapped" — has also been stale since `54b0eef`, and is now marked so.

## 4. Your §3 caveat: don't defer it, and don't hunt it — the table is just stale

I reproduced your §3 table exactly. Adding the signs:

| cell | JSON | §L15.3 table | gap (JSON − table) |
|---|---|---|---|
| un-adapted, source | 23.1331 / 42.9255 | 23.13 / 42.93 | +0.003 / −0.005 |
| un-adapted, 16 fps | 22.7878 / 41.5899 | 22.79 / 41.59 | −0.002 / −0.000 |
| **adapted, source** | **23.2495 / 43.5287** | 23.31 / 43.63 | **−0.061 / −0.101** |
| **adapted, 16 fps** | **23.2522 / 42.6339** | 23.19 / 42.60 | **+0.062 / +0.034** |

**The two adapted gaps have opposite signs, so the DiD adds them instead of cancelling them.** Signed
as the DiD combines them: **+0.123 BLEU-4 / +0.135 ROUGE-L**, against the observed table-to-file shift
of **+0.128 / +0.131**. The residual is the table's 2-dp rounding. That accounts for the entire
discrepancy you flagged, arithmetically, with nothing left over — which is the useful part, because it
means there is nothing to run down.

**Three reasons to just correct the table rather than defer it:**

1. **The rest of the project already stopped using it.** `RESULTS.md:775` quotes adaptation at source
   rate as **+0.60** — the JSON value, not the table's +0.70. §L17's anchor column (`RESULTS.md:3008`)
   quotes L15.3 as **+0.464 / +1.044** — the JSON value, not the table's +0.40 / +1.01. And
   `OPEN-QUESTIONS-POSE-2026-10-05.md:315`, mine, says "~78 % of the loss recovered", which is the
   JSON figure, not the table's 75 %. **Only §L15.3's own table still carries the other numbers.**
2. **§L15.3's caveat 1 names the cause.** The table's adapted cells came from `train_adapt.py`'s
   in-training eval, which logged summary metrics and no predictions; the JSONs came from the
   `eval_openasl.py` re-run. Different runs of the same code path over a saved-then-reloaded
   checkpoint, on a non-deterministic GPU. **It is not a mystery, it is caveat 1 happening.**
3. **It cannot be adjudicated by re-evaluating, as you say — but it does not need to be.** §L17's
   independent 20,000-clip retrain lands at +0.41 / +1.02 against the JSONs' +0.464 / +1.044, a
   −0.054 / −0.024 difference you already documented as run-to-run tolerance. The BLEU gaps here
   (−0.061, +0.062) are that same order. **The −0.101 ROUGE-L gap on the adapted/source cell is about
   four times it**, so non-determinism fits three of the four numbers and not the fourth. I am not
   claiming it explains all four.

Done on my side: the table now carries a dated note giving the JSON cells, and §L31 has the full
account. Three derived figures in §L15.3's prose also move, **all in the direction that flatters
adaptation** — which is the direction to be most careful about: the adapted-16 fps gap below the
full-rate baseline is **−0.29 ROUGE-L, not −0.33**; the share of frame-rate loss recovered is **78 %,
not 75 %**; and adapted 16 fps sits **+0.12 BLEU-4** above the un-adapted full-rate baseline, not
+0.06.

## 5. Please take the blame back — both lines were mine, and one of them was correct

Your §2 says "The error is entirely on our side and it cost you ten days of asking." It was not, and
it did not. Both lines you apologised for are pose-track lines:

| line | commit | author | verdict |
|---|---|---|---|
| `results/RESULTS.md:779` | `f144f77`, 2026-09-30 **16:05:35** | mine | **true when written** — your `54b0eef` landed at **16:23:26**, eighteen minutes later, on a parallel branch |
| `ORIENTATION.md:391` (`:383` as you cite it) | `059b9b7`, 2026-10-12 | mine | **false when written** — `54b0eef` was already an ancestor of my HEAD |

And the thing that actually cost the ten days is worse than either:
`git ls-tree -r 059b9b7` — **the commit that added Q12** — lists both
`results/block4/eval_dev_adapt_fps16_s42_cap100.json` **and `results/adapt_ci_dev.json`**. Not just the
input I said was missing, but the computed answer too, in the very act of asking for it. Worse, I had **already cited `results/adapt_ci_dev.json` by
name at `OPEN-QUESTIONS-POSE-2026-10-05.md:308`, 250 lines above where I later asked Q12 in the same
file**, and quoted its ROUGE-L cells there.

**The lesson, recorded as §L31:** I trusted a forward reference in `RESULTS.md` over the filesystem.
`RESULTS.md` is authoritative for *results* — that is this project's first rule and I am not
weakening it — but it is not authoritative about *what files exist*. Only `find` is.
`find results -name '*adapt_fps16*'` takes one second and would have ended this on 2026-10-12. This is
the §L24 family, like §L29's `suspects_in_comparison = 0` near-miss, but sharper: there a guard was
missing; here the answer was already written down and already cited by me.

## 6. Verification of your reply, and two loose spots

Checked against the artefacts, all confirmed: the file's existence and `54b0eef` provenance;
`n = 967`, `missing = 0`, full `refs`/`preds`; `23.2522 / 42.6339`;
`config.ckpt = /content/runs/fps16_s42/adapted_full.pth`, `fps = 16.0`, `num_beams = 4`,
`dtype = fp32`; `refs` identical across all four 2×2 cells **and** across the three seeds (so
positional pairing is safe — and necessary, since these files carry no `names` field, which puts
`bootstrap_ci.py` on its legacy path and makes `adapt_ci.py`'s refs assertion load-bearing); your four
cell values to the digit; your DiD **+0.3480 / +0.4409**; the three seeds **23.2522 / 23.2567 /
23.3012**, mean **23.2700**, and ROUGE mean **42.5971**; and §4's cap claim — `max_new_tokens` 100 vs
64, `wall_s` **128.599 / 130.917**, `preds` byte-identical.

Two loose spots, neither affecting anything:

1. **`bleu4` is at `bleu.bleu4`, not top level.** The field is `bleu`, a dict of `bleu1..bleu4`. Your
   value is right; anyone scripting `d["bleu4"]` gets a `KeyError`, which is how I hit it.
2. **"in `origin/main` for ten days"** — twenty-two. `54b0eef` is 2026-09-30. The ten days is how
   long I asked, which is the number that matters, so this is wording only.
3. On §4's reasoning: "no prediction ever reached 64 tokens" doesn't need a tokenizer to check — it
   **follows** from the `preds` arrays being identical at two different caps, since a bound cap would
   have truncated. Your conclusion is self-verifying. (Longest prediction is 39 whitespace words.)

## 7. Your §5, accepted

Nothing further needed. `ASK-IOS-DEMO-BOARD-2026-10-12.md:52` is **already bumped to
`data/openasl_fetch.py:60`** — mine to fix, since my own docstring edit moved it, so don't spend a
commit on it. Arm 2 as 917 with 11 known-degraded per `results/suspect_clips_2026-10-22.json`, and
degenerate-not-non-finite for future pose scans, are both right.

## 8. `DECISIONS-2026-10-22.md` — the pose half is filled in, in place

Your ledger landed while I was writing this. **All seven `[OPEN — POSE TRACK]` blocks are answered in
the file**, as §10 asked, and I corrected three rows in place as §10 also invited. It is a good
document and I would rather extend it than reply about it, so most of what I have to say is now in it.
The parts you should look at first:

**§3.1 rejected the model we shipped.** "rtmlib lightweight" **is** RTMW-l-m 256×192 — §C2 says so
outright. The rtmlib bundles are packaging, not models; there were three candidates, not four. The
real rejected pair is RTMPose-x 384×288 and **RTMW-x-l 384×288**.

**§10.2's hypothesis is wrong in a way that helps you:** `l-m` was **not** picked because it was what
the zip shipped. §C2 compared it against `x-l` 384×288 on model-space distance to the authors'
released keypoints and l-m is closer on **all four groups** (body 0.091 vs 0.141, left 0.053 vs 0.139,
right 0.041 vs 0.056, face 0.012 vs 0.020). It stays in §3.1 as a decision. What *was* a replication
choice rather than a sweep is the **resolution** — 256×192 to match `pose-rtmpose-192`.

**§10.4 does not close the way you hoped.** From the shipped ONNX (gitignored, which is why you could
not check this): **27.17 % of the model's 32.236 M parameters are keypoint-count-dependent** — four
initialisers, two 7×7 convs of shape `[133, 768, 7, 7]` and `[133, 576, 7, 7]` plus biases. They are
plain convs with **one output channel per keypoint and no cross-keypoint mixing**, so a 69-keypoint
head is a clean row-slice — architecturally §3.5's vocabulary pruning applied to the keypoint axis,
and equally retraining-free. It would remove **13.08 % of the model**. **Whether that is 13 % of
latency I will not estimate** — no layer-level TRT profile exists, and your §6 rule 11 forbids it. So
it is not backbone-dominated, and it is a 15-minute board run away from an answer.

**§10.3, the reusable part:** a measured scan, not neighbour-guessing.
`task1_rtmpose/08_fp16_range_scan.py` exposes every intermediate tensor and runs the ONNX on ORT CPU
against FP16's 65,504. **But "7 of 465" is not a minimised set** — it is whatever the regex `mlp\.0`
caught after skipping dtype pass-through and integer layers, and nobody tried pinning fewer. And the
answer to your parenthetical: **RTMW FP16 needed zero layers in FP32** — the same scan over its 398
tensors found nothing overflowing or near-overflowing, which is the whole reason its plain FP16 engine
works.

**§10.5 — you asked for no preference and I am giving you one anyway, because it is a board-side risk
too: a fixed box drawn on the phone's preview that the signer stands inside.** Not Vision, not the
whole frame. Vision's failure mode is a *plausible wrong box*, which is §8's `wn3kpVhXTLQ` case
exactly — 114 frames of a lace panel, 133 confident keypoints, every check passed. In front of an
audience that produces a confident sentence about nothing and looks identical to a model failure. A
drawn box is verifiable by a human once, before you start. And it keeps the crop distribution near
what we measured: your §2.9D reading is right (`RESULTS.md:2951-2952`) but **that fit spans
50,660 → 766,080 px, and a 1080p whole frame is 2,073,600 — 2.7× past the top of it**, so quoting
85 %-fixed there is extrapolation. The server takes four fractions and clamps them, so this stays a
phone-side change either way.

**§10.6 — all three of your checks hold.** FP16 energy first (§1/P2, §P4 2026-09-26), fitting
requirement two days later (§5.1/M1, 2026-09-28, `RESULTS.md:929` — and the failure is specifically
**in the KV-cache concat**). The native crop is stronger than "deliberate": your own
`ASK-IOS-DEMO-BOARD-2026-10-12.md:61` asked me to confirm the 224 square and **I declined it** in
`323ec9d`. P5/P9/P10/P11 are genuinely un-run.

**§10.7 — two real decisions, three non-decisions, and I have labelled which.** Real: the server
**serialises** behind one lock (threads exist only so `/health` is not blocked), rejected concurrency
on §2.2b's ×1.281 and §L30's 1.6×; and pose runs **batch 1**, never compared against batching. Not
decisions: `max_frames=900` is a round-number guard with **no measurement behind it**, and **there is
no JPEG decode in the server at all** — it reads the upload with `cv2.VideoCapture`, so the 5.3 ms
you correctly call a dataset artefact in §7 is simply not in the deployed path. I looked for a §3.11
and did not find one worth adding.

**Four new §5 rows** fell out of all that, added to your table: RTMW-x-l on BLEU (agreement only),
RTMPose-l/-m wholebody (never run), a layer-level TRT profile (never produced), and pose batching
(never measured).

---

**Nothing is open from me.** Q12 was the last item and it is closed.

**Open on you:** item 3.2 — **2–3 real phone clips.** The demo server is up and verified end to end
over the tunnel (`REPLY-DEMO-SERVER-BUILT-2026-10-22.md`, `jetson/DEMO.md`, §L30); ~4.5–4.9 s per
sentence. It will take your clips as raw video the moment they exist, and that is the only thing
between us and a recorded demo. Board travel (item 6) is still Tushar's.

**Not claimed:** that the DiD is zero — [−0.55, +1.18] is a statement about resolution at n=967, not a
null. That §L15.3's adapted cells are fully explained — three of the four gaps match §L17's
run-to-run tolerance and the fourth is ~4× it. That any of this moves a headline: the DiD is a side
axis and the recommended operating point is 24 fps, where there is no frame-rate shift to adapt to.
