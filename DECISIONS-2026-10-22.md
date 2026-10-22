# The decision ledger — every choice we made, what we rejected, and the number that decided it

*Started 2026-10-22 by the LM track. **This is a draft with the pose half deliberately left open** —
see §10. Written for the two of us, because a lot of this project was carried by tooling and neither
of us can currently recite the decision chain end to end.*

---

## 0. How to read this, and the rules for extending it

This file is **not** new evidence. It is a ledger: one row per decision, pointing at the section of
`results/RESULTS.md` that already holds the measurement. If this file and `RESULTS.md` ever disagree,
**`RESULTS.md` wins and this file is stale.**

Three conventions, and they are load-bearing:

1. **Every claim carries a section reference** (`§L5.3`, `§2.5c`, …). A claim with no reference is not
   a claim, it is a guess, and it does not belong here.
2. **"Established" means a 95 % confidence interval that excludes zero.** Everything else is
   "not established", which means *unknown*, **not** "no effect". We got this wrong once in a way that
   shaped three weeks of work (§4.2 below), so the wording is strict.
3. **`[OPEN — POSE TRACK]`** marks something the LM track does not know. Those blocks are questions,
   not placeholders. **Do not fill one with a plausible number.** If the answer is "we never ran it",
   that is a real and acceptable answer — write that.

### Two companion files that already exist, and how this one differs

| file | what it is |
|---|---|
| `ORIENTATION.md` | the **map** — what flows into which stage. Written 2026-10-03, patched since. |
| `RESEARCH-STORY-2026-10-03.md` | the **narrative** — how we got here, in order, including the retractions. Three weeks stale. |
| **this file** | the **ledger** — chose / rejected / why, with nothing in chronological order. |

---

## 1. The one fact that explains almost every choice

Two models do all the real work, and their costs scale on different axes:

| | pose model (RTMW) | language model (mT5) |
|---|---|---|
| cost scales with | **number of frames** | **number of sentences** |
| measured per unit | 13.8 ms / 127 mJ per frame (§1, P2 row) | ~1.9 s / ~11 J per sentence (§2.9C) |
| share of system energy | **~75 %** | ~25 % |

Everything downstream follows from that asymmetry:

- **Fewer frames is the only big lever.** It attacks 75 % of the budget.
- **Decoder tricks are small levers**, because they attack 25 %. Measured at the system level:
  dropping 30 → 24 fps costs **103.5 J per BLEU-4 point** sacrificed; narrowing beam 4 → beam 2 costs
  **1.0 J per BLEU-4 point** (§2.9, "What this means for the system"). Frame rate is **10–100×** the
  more efficient lever.
- **The live demo says the same thing, more bluntly.** In `demo_server.py`, pose is **58 %** of a
  request, and beam 4 → greedy saves 531 ms of decoder but only **12 %** of total request time
  (§L30).

If you remember one thing from this file, remember that the project's headline recommendation is a
**frame-rate** result, and that we spent real effort on quantisation and pruning that turned out to
matter mostly for *fitting in memory*, not for energy.

---

## 2. What we actually shipped

The cleanest statement of the final configuration is the argparse block of the thing that runs
(`jetson/demo_server.py:358-378`). Every default there is a decision with a measurement behind it:

| knob | shipped value | decided in |
|---|---|---|
| pose model | `rtmw-l-m_256x192` | §C2, §2.5c |
| pose precision | **FP16** TensorRT engine | §1 (P2), §2.5c, §P4 |
| LM checkpoint | `openasl_pose_only_slt_pruned.pth` — vocabulary-pruned mT5-base | §L5.2, §L5.3 |
| LM runtime | **PyTorch, FP32** — *not* TensorRT, *not* INT8 | §L8.2, §L9.1, §L30 |
| frame rate | **24 fps** | §2.9, §L13, §5.3 |
| beam width | **4** | §L6.1 |
| `max_new_tokens` | **64** | §L3.2 |
| `max_length` (encoder frame cap) | **256** | §L7.1 |
| crop | native-resolution bbox, **board-side**, no square/pad/resize | §3.1, §2.5b, `jetson/DEMO.md` |
| non-finite pose frames | **dropped and counted**, never translated | §L27, §L28, §L30 |

Measured end to end on the board, 4.53 s of video thinned to 109 frames: **4.5–4.9 s per sentence**,
of which pose is 58 % (§L30).

---

## 3. The closed decisions, one at a time

### 3.1 Which pose extractor — RTMW-l-m 256×192, and it was not the obvious pick

**Chosen:** RTMW-l-m at 256×192.
**Rejected:** RTMPose-x at 384×288, and the two rtmlib bundles (lightweight / performance).

The honest order of events matters here. We did **not** pick RTMW for speed. We ran §C2 to find out
*which extractor the Uni-Sign authors actually used to make their released keypoints* — the archive
folder is literally named `pose-rtmpose-192` — and matched it. RTMPose-x 384×288 then became the
"oversized comparison" row rather than the baseline.

The paper specifies RTMPose-x, so choosing otherwise was logged as an accuracy **risk**. §2.5c
reversed that worry, at n=30 clips from 30 distinct videos:

| config | BLEU-4 | ROUGE-L |
|---|---:|---:|
| authors' own poses (the ceiling) | 18.17 | 45.03 |
| **RTMW FP16 + coordinate-frame fix** | **18.52** | 43.91 |
| RTMW FP32 | 16.55 | 40.72 |
| RTMPose-x FP32 | **9.42** | 32.99 |

Paired bootstrap, RTMW FP32 → RTMPose-x FP32: **−7.13 BLEU-4 [−10.94, −3.07]**, **−7.73 ROUGE-L
[−12.10, −3.57]** — established on both metrics. The cheap choice was also the accurate one, which we
were not entitled to assume.

> This does **not** contradict the paper. Their RTMPose-x ran at 192 width on their own square crops;
> ours ran at 384×288 on native crops. What is established is that *within our pipeline* RTMW is much
> the better of the two.

### 3.2 Pose precision — FP16, and it became mandatory rather than merely cheap

**Chosen:** FP16. **Rejected:** FP32 (too slow, and later: does not fit), INT8 (never built — §5.1).

Measured on the board at 15 W (§1):

| | TRT ms | total ms | energy |
|---|---:|---:|---:|
| RTMW-l-m FP32 | 25.6 | 33.0 | 275 mJ/frame |
| **RTMW-l-m FP16** | **13.8** | **19.7** | **127 mJ/frame** |
| RTMPose-x FP32 | 66.7 | 79.2 | 733 mJ/frame |

Accuracy cost of FP16: **−0.13 BLEU-4 [−1.48, +1.21]** (§2.5c) — free within noise, and 99.81 % of
the keypoints Uni-Sign actually consumes land within 5 px of FP32 (§P4).

**It later stopped being optional.** The FP16 engine is 68 MB against FP32's 159 MB, and the same
end-to-end run with the FP32 engine runs out of memory partway through LM decode (§3.2 of
`ORIENTATION.md`). FP16 is now a *fitting requirement*, not an efficiency choice.

**Three traps inside this one decision, all worth keeping:**

1. **TensorRT turns TF32 on by default on Ampere, so an "FP32" engine built with defaults is not
   FP32.** With TF32 on, the correctness gate fails at 1.55e-3; with `BuilderFlag.TF32` cleared it
   passes at 7.2e-6 with all 2416 confident keypoints at 0.0 px. `common/trt_runner.py` clears it.
2. **Blanket FP16 on RTMPose-x is broken by one layer overflowing** — `/mlp/mlp.0/ReduceSum` peaks at
   142,293 against FP16's maximum of 65,504. Keeping 7 of 465 layers in FP32 fixes it for under 3 %
   latency.
3. **A max-pixel-error gate cannot tell a rounding flip from an overflow** — both print "FAIL". The
   metric that separates them is *agreement*: fraction of consumed keypoints within 5 px, judged only
   where the FP32 score ≥ 0.3. And the ≤1 px / ≤2 px columns are **not comparable across clips**,
   because the underlying SimCC bin size is bbox-dependent (1.71–2.45 px over five clips), which
   swings the ≤2 px column by 13 points on geometry alone. **Only the ≤5 px column compares.**

**One limitation we could not design away:** hands are the floor in *every* configuration (~99 %
within 5 px, p99.9 ≈ 30 px) and are **completely unchanged** by the precision fix. So hand error is
argmax instability in the pose head, not a precision problem, and no choice of number format fixes it.

### 3.3 The crop — native-resolution bbox, cropped on the board

**Chosen:** crop to the OpenASL signer bbox at **native resolution and native frame rate**, no square
pad, no resize to 224. **Rejected:** OpenASL's own square-pad-resize-224 recipe.

This is the decision most likely to be got wrong by whoever builds the phone side, so it is written
into `jetson/DEMO.md` as well. RTMW does its **own** affine transform internally. Cropping to a 224
square first resamples twice and feeds RTMW's affine black padding — a path **nothing we have
measured covers**. The fetch indices record `crop_style=native`, "no square/pad/resize".

**Consequence for the demo:** the phone sends **full frames plus a box as four fractions**; the server
crops. The box may fall outside 0–1 — OpenASL's own boxes routinely do — and is clamped (§L30, §5 of
`REPLY-DEMO-SERVER-BUILT-2026-10-22.md`).

**The expensive bug that came out of this.** Our crop and the authors' crop define coordinates in
*different frames*, a **1.68× normalisation mismatch**. We found it because the keypoint disagreement
correlated **0.968** with the predicted ratio (§2.5d). `common/renorm_to_openasl.py` fixes it by exact
arithmetic — body-keypoint disagreement 0.1328 → 0.0055, a 24× reduction. With it applied, our best
configuration (18.52 BLEU-4) is statistically indistinguishable from the ceiling (18.17) **at n=30**.

That last clause is the correction we owe ourselves: **at n=931, the gap to the ceiling is established
at −1.41 BLEU-4 [−2.19, −0.63]** (§2.5k). The frame fix removed *most* of the gap, not all of it, and
30 clips were too few to see the remainder.

### 3.4 The converter — only 69 of 133 keypoints are ever used

`common/pose_to_unisign.py` turns keypoints into Uni-Sign's four groups (body / left hand / right hand
/ face), root-normalised, confidence threshold 0.3. It is **bit-exact** against the authors' own
`load_part_kp` / `crop_scale`.

The count people keep getting wrong — `common/pose_to_unisign.py:35-38` is the authority:

| group | indices | count |
|---|---|---:|
| body | `[0] + range(3, 11)` | 9 |
| left hand | 91–111 | 21 |
| right hand | 112–132 | 21 |
| face | every other of 23–39, then 83–90, plus 53 | 18 |
| | | **69** |

So we extract 133 and spend compute on 133, but the model sees 69. **We never tested extracting only
the 69** — see §5.3.

### 3.5 Vocabulary pruning — the big LM win, and it is a memory win

**Chosen:** slice mT5's embedding and output head down to the tokens OpenASL actually uses.

OpenASL's 98,419 sentences use only **26,078 of mT5's 250,112 tokens** (§L5.1). Slicing `shared` and
`lm_head` to that keep set (§L5.2):

| | params | checkpoint |
|---|---:|---:|
| full | 587.7 M | 1,187 MB |
| **pruned** | **243.6 M** | **571 MB** |

Cost: **−0.28 BLEU-4 [−0.63, +0.05]**, P(pruned < full) = 0.95 (§L5.3). Not an established loss, but
probably a real small one, bounded at about 0.6. No retraining. 321 of 976 predictions change, because
dropping 224 K rows from the softmax shifts beam scores even where the argmax is unchanged — greedy on
the single demo clip was token-identical.

**It is also worth ~12 % of system energy** — 37.7 vs 42.9 J/sentence at 24 fps / beam 4 (§5.4) —
which had never been measured end to end; before that it was only ever a memory and accuracy argument.

**The methodology error inside it, which we keep rather than hide.** The first token census included
the **test** split, so the vocabulary was selected using test data. Re-censused over train+dev only
(§L5.4): **26,025** tokens, 53 dropped, **0 added**, and all 53 occur *zero* times in train+dev. The
leak was worth **−0.03 BLEU-4 [−0.11, +0.01]** — a **precise** null, not an underpowered one, because
the two models differ by 53 of 26,078 rows so most sentences are word-identical.

One of the 53 was `▁Bitcoin`, the key content word of the clip we used as this project's standard
single-clip demo. It was in the deployed vocabulary *only because it appears in the test split*.

> **Awkward fact, stated rather than buried:** the checkpoint actually on the board is the **leaky
> 26,078** keep set (`REPLY-Q10-TEST-POSES-TOO-2026-10-07.md:36`). The leak-free 26,025 build
> (`openasl_pose_only_slt_pruned_traindev.pth`) is the LM-track Mac's training base. Since the leak is
> a bounded null at ~0.1 BLEU-4 this changes no conclusion, but the report must not claim the
> demo runs a leak-free vocabulary.

### 3.6 The output-length cap — free, and verifiably so

**Chosen:** `max_new_tokens = 64`. **Rejected:** 48 (a real small loss), 100 (no benefit).

§L3.2, 976 test clips: cap 64 is **bit-identical** to cap 100 — 0.00 delta, 0 of 976 sentences
changed, because the longest prediction in the whole split is **62** tokens. Cap 48 costs
**−0.14 BLEU-4 [−0.30, −0.03]**, statistically real, truncating 4 % of sentences.

So 64 is free and bounds the worst case. The saving is in **tail** latency, not the mean — the average
sentence is 21 tokens and is unaffected.

### 3.7 Beam width — the obvious lever, and the wrong one

**Chosen:** beam 4. **Rejected:** greedy, beam 2 — on accuracy, not on cost.

§L6.1, 976 clips, pruned model:

| decoding | BLEU-4 | paired delta vs beam 4 | speed |
|---|---:|---|---|
| beam 4 | 22.87 | — | — |
| beam 2 | 22.06 | **−0.81 [−1.29, −0.39]** | 1.7× faster |
| greedy | 20.88 | **−2.00 [−2.63, −1.41]** | 2.7× faster |

Both are **established losses**. Greedy is a real 2-point loss, not the "well under a point" our
original plan assumed.

**Why we keep beam 4 anyway**: because of §1. Greedy saves 2.16 J of LM energy, which is **5 % of
system joules**, for 2.00 BLEU-4 — about **1.1 J per BLEU-4 point** against **103.5** for frame rate
(§2.9). Calling beam width "the frontier" was true only of the LM *in isolation*. The live server
reproduced this at the system level: **12 % of request time** for the whole beam-4 → greedy move
(§L30).

### 3.8 Frame rate — the actual result of the project

**Chosen:** 24 fps. **Rejected:** 30 (source — more energy for nothing), **16 fps** (which we
recommended for two weeks and then retracted), 12, 8.

**Energy, measured on the board** (§2.9A). Pose energy per second of video at 30 / 24 / 16 / 12 / 8
fps: **4.25 / 3.36 / 2.27 / 1.67 / 1.17 J/s** = **1.00 / 0.79 / 0.53 / 0.39 / 0.27×**. Energy per
*frame* stays flat — which is the whole point. **You save by processing fewer frames, not cheaper
ones.**

**Accuracy, measured offline at n=976** (§L13): 24 fps is free (test −0.07, dev +0.26); 16 fps costs
−1.1 to −1.3 BLEU-4 and −1.3 to −1.6 ROUGE-L; 12 fps −2.5; 8 fps −8.0.

**The operating point:**

| cell | system J | BLEU-4 | verdict |
|---|---:|---:|---|
| beam 4 @ source | 43.4 | 22.87 | reference |
| **beam 4 @ 24 fps** | **36.1** | **22.80** | **−17 % energy for −0.07 [−0.53, +0.39]** — free |
| beam 4 @ 16 fps | 27.0 | 21.54 | −41 % energy for −1.33 [−2.00, −0.64] — **established loss** |

**Why the saving lands on the pose side and not the LM.** Shortening the frame sequence shortens the
LM's *encoder*, which costs 39–50 ms, while the *decoder* costs 1062–1538 ms. Cutting the sequence
from 256 to 68 frames — 73 % fewer — saves the LM only **12 %** (§2.9C). So frame rate is a pose lever
that the LM barely notices, and **a report that presents it as an LM saving is claiming ~12 % where
the pose side delivers 47–73 %.**

### 3.9 Adaptation — we built the harness; the result is one unfinished computation away

**Chosen:** fine-tune the pose stack + projection (5.35 M of 243.5 M params), **mT5 frozen**, FP32, no
AMP, 20,000 train clips, 1 epoch, lr 1e-5, label smoothing **0.0**, warmup 0.1 (§L15).

**Rejected:** label smoothing 0.2 and lr 1e-4 — and the way we rejected them is the useful part.

Our first recipe appeared to show that adaptation *hurt* at 16 fps. The control with **no distribution
shift at all** degraded by 1.04 BLEU-4 too, which proved the number was never about adaptation — it
was a broken recipe (§L15.1). Cause: `--label-smoothing 0.2` over a 26 K vocabulary parks the reported
loss at a ~3.35 smoothing floor that looks exactly like a training curve. At 0.0 the true
cross-entropy is ~0.36, and at lr 1e-5 fine-tuning *improves* both metrics.

**Status as of today**, and this is the one genuinely unfinished result in the LM track:

- **Frame-rate adaptation (does training recover what 16 fps costs?)** — the 2×2 exists in §L15.3 and
  all four eval JSONs are in `results/block4/`. `adapt_ci.py` has not been run, so there is **no
  confidence interval** and it is **not quotable**. Recomputed from the files the
  difference-in-differences is **+0.348 BLEU-4 / +0.441 ROUGE-L**; §L15.3's prose says +0.22 / +0.31
  and we have not run that discrepancy down (`REPLY-Q12-EVAL-JSON-EXISTS-2026-10-22.md`, deferred
  until after the demo).
- **Pose adaptation (does training on *our* keypoints close the −1.41 pose gap?)** — **answered, and
  it is a null.** §L26: **−0.1263 BLEU-4 [−0.4973, +0.2488]**, and the interval **excludes** the
  +0.5490 §L17 predicted and the +0.7785 bar §L21 set. So a benefit above **~18 %** of the −1.41
  deficit is excluded at 95 %. The pose deficit is attributed to architecture, crop and normalisation
   — with quantisation excluded to within 11 % (§L25) and adaptation to within ~18 %.

### 3.10 The demo shape — thin client, board does the work

**Chosen:** phone captures and uploads; **board runs both models**; English comes back as text.
**Rejected:** running the models on the phone (Core ML), and a pre-recorded simulator demo.

The board half is built and measured: `jetson/demo_server.py`, one long-lived process holding the
RTMW FP16 engine **and** the pruned LM resident, `POST /translate` with video bytes in the body and the
person box as four query fractions (§L30).

Two findings from it that no composed estimate would have produced:

1. **The first request costs 11.7 s; every one after it costs 4.5 s** — a **2.6× penalty paid entirely
   by whoever goes first**, which in a live demo is the audience, on the one sentence everybody is
   watching. Warming the TensorRT engine is **not enough**; the LM's first `translate()` pays its own
   kernel setup. A synthetic warm-up sentence at start-up costs 6.1 s of load time and removes it.
2. **The pose stage costs 21.5–22.8 ms/frame in the live service against 13.8 ms standalone — about
   1.6×.** That is §2.2b's contention effect showing up in a deployed service: CPU-side stages slow
   when the GPU takes more of the shared 15 W budget, and it lands on the stage that charges *per
   frame*.

**Memory is the binding constraint and there is almost none left:** MemFree goes 4,802 → 4,650 (pose
engine) → 193 (LM) → 75 (warm-up) → **91–130 MB steady**. The first launch *failed* with
`NVML_SUCCESS == r INTERNAL ASSERT FAILED` after reclaiming to only 3,500 MB; it needs a reclaim to
5,400 first. **This is why the LM runs as PyTorch and not as TensorRT engines** — the FP32 engine set
is 2.07 GB resident (§L8.2) and there is not 2 GB of room.

---

## 4. Rejected choices, consolidated — including three where our first reason was wrong

### 4.1 Rejected on the evidence

| option | why it lost | where |
|---|---|---|
| **RTMPose-x 384×288** as the pose baseline | −7.13 BLEU-4 [−10.94, −3.07] vs RTMW in our pipeline | §2.5c |
| **FP32 pose engine** | 2× the latency, 2.2× the energy, and 159 MB does not fit alongside the LM | §1, §3.2 |
| **mT5 in FP16** | **not a quality loss, a dead model** — every step returns token 0 at logprob −10.169, which is exactly −ln(26078), i.e. a uniform distribution | §L8.2 |
| **mT5 in BF16** | runs, produces coherent text, but diverges from the FP32 greedy path at step 3 where the margin is small (logprob −1.05). 8-bit mantissa is too coarse for token-exact decoding | §L8.2 |
| **TensorRT for the decoder** | FP32 engines are *token-identical* and 2.9× faster per step (19 vs 55 ms) — but the decoder is bound by the **host-side step loop**, so it is not faster end to end, and 2.07 GB resident does not fit | §L8.2, §L30 |
| **greedy / beam 2** | established losses of 2.00 and 0.81 BLEU-4, for ~5 % of system energy | §L6.1, §2.9 |
| **16 fps** | established loss, −1.33 BLEU-4 [−2.00, −0.64] | §L13 |
| **`max_new_tokens` 48** | −0.14 [−0.30, −0.03], truncates 4 % of sentences, saves nothing on the mean | §L3.2 |
| **OpenASL's square-pad-224 crop** | resamples twice and feeds RTMW's affine padding; no measurement covers that path | §3.1, `jetson/DEMO.md` |
| **Sentence-segmentation heuristic (P12)** | deleted by a *design* choice, not a measurement: the phone uses **hold-to-record**, so the button press is an exact sentence boundary and the heuristic has nothing to infer | `WORKSPLIT.md` P12 |

### 4.2 The three where we rejected the right thing for the wrong reason

These are the ones worth being able to tell properly, because each one is a correction we published
against ourselves.

**1. Weight-only INT8 on the LM.** Our first reason for dropping it was that it would inherit the FP16
failure. **That was wrong.** The implementation is **W8A32** — int8 weights, fp32 activations *and
accumulation* — so it cannot inherit an activation-overflow failure, and the proof is that it scores
**22.79 BLEU-4**, which is impossible under a uniform-logits failure. It works: 293 MB, **4.05×
smaller**, **−0.09 BLEU-4 [−0.34, +0.14]** vs pruned (§L9.1).

**The valid reason to drop it is that the decoder is host-bound**, so INT8 buys memory and not
latency — and the int8-in-RAM runtime is actually **10× slower per step** (2139 vs 197 ms) because 217
layers dequantise on every token. A *fused* W8A32 kernel is the version that would save both, and we
did not build one.

**2. "16 fps is free."** The original n=30 curve put 16 fps at Δ −0.15 BLEU-4 with a CI of
**[−3.20, +2.89]** and it was written up as "no measured accuracy cost". That interval is wide enough
to contain a 3-point disaster. At n=976 the real cost is −1.33 [−2.00, −0.64]. **A wide interval is
ignorance, not good news** — this is the single most expensive lesson in the project, and the
operating point moved from 16 to 24 fps because of it.

**3. "OpenASL is 30 fps."** Both tracks hardcoded ~29.97. In our 931 clips, **76.5 % are ~30 fps,
22.2 % are ~24 fps, and 7 are 59.94** — so a 24 fps clip labelled "16 fps" was really subsampled to
~12.8, and **6 of the first 30 clips were off-rate**. One constant invalidated the first frame-rate
curve on *both* tracks. The rate is now derived per clip as `n_frames / duration_s`, and frame
selection is `round(duration × target)`, verified identical between our `keep_idx` and their
`fps_ratio_for_clip`.

### 4.3 Also withdrawn, and not to be resurrected

- **"LM load peaks near 4.2 GB and memory is binding at load."** An **occupancy artefact** — the
  probes ran on a shared board with no co-tenant check. Measured peak device allocation is **0.98 GB**
  (§5.2). The consequence that matters: *this is the finding whose retraction closed INT8*, and it is
  closed.
- **"RTMW FP16 is translation-neutral (5/5 identical sentences)."** Retracted as a five-sentence coin
  flip and replaced with the powered version in §2.5c (§2.5b).
- **Our own FP32 throughput correction** — retracted in §L24, and it was wrong in the direction our
  own argument predicted.

---

## 5. Rejected by omission — things we never ran, stated as gaps

These are not decisions, they are absences. They belong in the report's limitations section, and
calling them "future work" without saying they were in the plan would be dishonest.

| item | state | note |
|---|---|---|
| **INT8 PTQ on the pose model (P5)** | **never built** | Corrected 2026-10-22: `common/trt_runner.py:27` has only an INT8 dtype *mapping* — there is **no calibrator, no INT8 build path, no artifact**. The 361 frames in `data/calib_frames/` are the only ready part. Dropped deliberately (`results/RESULTS.md:835-836`) because frame rate dominates it 100× at the system level. |
| **Dropping the face group (P9)** | **neither side ran it** | 51 fewer keypoints, would need a re-adaptation pass (feature width 4C → 3C). |
| **Extracting only the 69 consumed keypoints** | never tried | We pay for 133 and use 69 (§3.4). Whether a 69-point head is even available is a pose-track question — §10.4. |
| **PTQ vs QAT on the pose front-end (P10)** | not started | |
| **GPU preprocessing (P11)** | not started | The cost *is* quantified: imread 6.02 ms + preprocess 5.11 ms per frame, 19 % of frame time (§2.2b). |
| **Same-model resolution row (P7)** | partial | We compared *across* models (RTMPose-x 384 vs RTMW 256), never the same model at two resolutions. |
| **A board-measured BLEU** | **never produced** | *Every* accuracy number in this project is offline on the Mac. The only bridge is a 30-clip batch-size-1 agreement check. This is the biggest genuine methodological gap we have. |
| **Streaming / overlapped stages** | unmeasured | M1 ran all frames *then* the LM, sequentially, and the 15 W contention we predicted did not appear. **M1 is not evidence that streaming is free** — a pipeline decoding sentence N while extracting N+1 would contend. §L30's 1.6× pose slowdown is the first hint of what that costs. |
| **iOS app (P14)** | nothing built | The board half is done; the phone half is not. |

---

## 6. The measurement rules, and the mistake that bought each one

Every rule here exists because we got something wrong first. This is the part of the project most
worth writing up, because it generalises past this system.

1. **Paired bootstrap, one set of draws for every run.** Score every configuration on the *same*
   resample draws, so deltas and differences-of-deltas are comparable. Subtracting two
   independently-bootstrapped CIs is both wrong and less sensitive.
2. **Report both metrics — either one can be blind to your effect, and the one that resolves it can
   be the one furthest from the truth.** The frame-rate effect shows on ROUGE-L and not BLEU-4; the
   pose-substitution effect showed on BLEU-4 and not ROUGE-L **at n=400** — then at the full 931 clips
   **both** establish it and they converge (−1.41 and −1.33, §2.5k). So that asymmetry was a *power*
   artefact, and the metric that did resolve it at n=400 **overstated it by 0.83**. It is **not** that
   one metric is more powerful: BLEU-4's interval is the *narrower* of the two in all four comparisons
   on shared draws. What differs is whether the effect registers on that metric at all.
3. **"Not established" means unknown.** See §4.2.2.
4. **n=300 can flip the sign of a known effect**, and did — at n=300 the same model scored +1.10
   BLEU-4 *higher* at 16 fps than at source. All adaptation work uses the full 967-clip dev split.
5. **Run the no-shift control before any adaptation claim.** §4.2 / §3.9.
6. **Seed spread is not a confidence interval.** It bundles seed effects with GPU non-determinism,
   which is real here: three runs at the *identical* seed gave final losses 0.3765 / 0.3742 / 0.3788.
7. **One success is not a threshold.** We said LM load needs ~4 GB MemFree, inferred from M1's single
   success at 4,625 MB. Measured properly it **fails at 5,214 and succeeds from 5,268** — a cliff just
   above 5.2 GB, which our reclaim target of 5,000 straddled (§5.3). The same error shape as the
   occupancy episode.
8. **Check and record board occupancy on every run.** An unrecorded co-tenant invalidated the 09-29
   memory probes, and that bad result propagated across tracks before it was caught.
9. **Batch size changes the output.** `eval_openasl.py` batches 8 clips and pads; the board runs one.
   At `--batch-size 1` the eval reproduces M1's sentence exactly, and 2 of 30 clips differ between
   batch 8 and batch 1. Smaller than anything we reason about — but stated, not discovered later.
10. **End-to-end energy must be quoted per *sentence*, not per frame.** `mJ_per_frame` *rises* under
    subsampling (169.07 → 189.70) because the per-sentence LM cost spreads over fewer frames (§5.3).
    Per-frame is still correct for the pose stage alone, where it was defined.
11. **No hardware number is ever estimated.** If it is in `RESULTS.md`, a run on this board produced
    it. Accuracy numbers are the Mac's; latency, power and energy are the board's.

---

## 7. Hardware facts that kept catching us out

- **Memory is unified** — CPU and GPU share one 8 GB pool, there is no separate VRAM, and a GPU
  allocation failure surfaces as `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
  CUDACachingAllocator.cpp:1017`. **Read that as "out of memory."**
- **15 W mode caps the GPU at 612 MHz on this board.** Our earlier 44.4 ms / 11.6 W pose numbers match
  the 25 W cap (918 MHz) exactly — a board-and-mode difference, not a code difference. TensorRT latency
  scales almost exactly with the clock cap, so **every row must carry its measured `gpu_MHz`**.
- **No sudo, ever.** Course-owned board. So no 7 W mode, no `jetson_clocks`, no pinned clocks. Power
  budgets in this project are *measured average watts*, not a hardware cap.
- **CPU stages slow by ×1.281 when the GPU takes more of the 15 W budget.** "CPU time" is not a
  constant. §L30 measured this as a 1.6× pose slowdown in the live server.
- **Two energy bases and two speed metrics — always say which.** Total mJ/frame includes the 3.6–4.0 W
  idle draw; dynamic excludes it. `fps_compute = 1000/total_ms`; `fps_end_to_end` includes JPEG decode
  (~5.3 ms, a dataset artefact a camera pipeline would not pay).
- **The board is shared scratch.** Results get pulled to the Mac and the bulk deleted.

---

## 8. Data integrity findings — the part we would rather not have found

Two of eight inspected dev/test clips are **good video with the bounding box on the wrong thing**
(§L29, `REPLY-DEV-CLIP-QUALITY-2026-10-22.md`). The worst, `wn3kpVhXTLQ-00:02:11.433-00:02:15.233`,
has a 128 × 178 box sitting on the signer's **torso** — all 114 frames are a close-up of the lace
panel of her top, and the pose head dutifully returned 133 keypoints for it.

**Why this evades every check either track runs**, which is the generalisable part:

- **It produces no NaN.** A pose of a torso is finite and in range. Our scan finding *0 of 967
  non-finite* is a true result and does **not** mean the clips are good.
- **It produces no confidence *change*.** A box that is wrong from frame 0 to the last frame is
  flatly mediocre throughout, so a change-detector never fires. Both verified cases surfaced only
  because their crops were also unusually small.
- **So the published rate is a floor**, not an estimate. A wrong box over a *large* region trips
  neither rule, and 98 % of clips are unaudited.

**Why it does not threaten the headline:** both arms eat the same bad clips — the control arm trains on
the authors' released poses, made from the same videos using the same published `bbox-v1.0.json` — so
the defect is **symmetric across the arms** and cancels out of `ours − theirs`. And the test side is
measured, not argued: removing all 16 flagged test clips moves the pose deficit from
**−1.4077 [−2.1859, −0.6307]** to **−1.3147 [−2.1182, −0.5187]**, about one eighth of its own
half-width, both intervals still excluding zero.

**Write-up obligation:** arm 2 trained on **917 clips of which 11 are known-degraded**, not on 917
clean clips.

Separately, §L27/§L28: **a NaN confidence left a finite-looking coordinate behind** for 7 frames of one
dev clip (indices 9, 12, 13, 14, 15, 19, 22) — caused by **FP16** in the pose head, with FP32 on
identical frames producing none. The frames are B-roll with no signer in them. The guard now drops and
counts non-finite frames, and §L30 demonstrated it live: **12 of 150 frames dropped**, sentence
translated from the remaining 138. On live phone video this will fire constantly, and it is the
difference between a dropped frame and a confident translation of nothing.

---

## 9. Still genuinely open

1. **`adapt_ci.py` has not been run**, so the frame-rate difference-in-differences has no interval and
   is not quotable. Unblocked as of today (`REPLY-Q12-EVAL-JSON-EXISTS-2026-10-22.md`).
2. **§L15.3's adapted cells do not reproduce** from the committed eval JSONs. Low priority, deferred
   until after the demo, recorded so nobody quotes the table in the meantime.
3. **No board-measured BLEU exists.** §5.
4. **The phone half of the demo does not exist.** §10.
5. **Whether the board may physically leave the lab** — asked, still unanswered (`ASK-IOS-DEMO-BOARD`
   item 6).
6. **The phone's WireGuard `AllowedIPs` must include `192.168.1.0/24`.** Unverified, and the
   likeliest thing to go wrong on the day — the failure is **indistinguishable from a blocked port**.

---

## 10. `[OPEN — POSE TRACK]` — what the LM track cannot write

**To whoever completes this:** append your answers under each heading. Keep the §-reference rule from
§0. **"We never ran it" is a correct and welcome answer** — say so rather than reasoning toward a
number. And if any row above is wrong, correct it in place and say so; the LM track inferred several
pose-side statuses from committed artifacts rather than being told them.

### 10.1 The variant sweep, in full
§2.5c gives RTMW vs RTMPose-x. `WORKSPLIT.md` P6 also lists **rtmlib lightweight and performance**
bundles as swept. Those rows are not in the ledger above because the LM track does not have them
broken out. **What did they score, and on what — keypoint agreement, BLEU, or latency only?**

### 10.2 Why *l-m* specifically
RTMW ships in several sizes. §C2 established that the authors used *a 256×192 RTMW/RTMPose model* and
we matched the resolution. **Was `l-m` chosen because it is what the SDK zip shipped, or was it
compared against `x-l` 384×288 and other sizes? If it was never compared, that belongs in §5 as a gap,
not in §3.1 as a decision.**

### 10.3 The 7 FP32 layers
§3.2 trap 2 says 7 of 465 RTMPose-x layers stay in FP32. **How were those 7 picked — a measured
activation-range sweep, or the one overflowing layer plus its neighbours?** The procedure is the
reusable part, and it is currently unrecorded. (Note this applies to the *rejected* RTMPose-x path;
say explicitly whether RTMW FP16 needed any layer kept in FP32, or whether it was clean.)

### 10.4 Is a 69-keypoint extraction even possible
§3.4: the model consumes 69 of 133. **Is there a wholebody head that emits a subset, or is 133 the
only granularity RTMW offers?** If extraction cost is dominated by the backbone rather than the head,
say so — that closes the question permanently instead of leaving it as future work.

### 10.5 Where does the person box come from on the phone
This is the biggest hole in the demo and it is a **design** question, not a measurement.
`RESULTS.md` has **no person detector anywhere** — every box in the project comes from OpenASL's
published `bbox-v1.0.json`, and `03_infer_frames.py:11` defaults to the whole frame. The server takes
a box as four fractions. So on the day, **what sends them?**

Options as the LM track sees them, with no preference asserted:
- the signer frames themselves and we send the whole frame as the box;
- a fixed box drawn on the phone's preview that the signer stands inside;
- an on-phone detector (Vision framework), which is new, unmeasured code;
- something else.

**Whichever it is, the crop geometry §3.3 depends on it.** A whole-frame box is a far larger crop than
anything in the energy table (§2.9D: 15.1× the crop area costs 1.34× the energy, so cost is **85 %
fixed** — `mJ/frame = 109.7 + 49.5/Mpx`), so it is affordable, but it is also a crop distribution
nothing has been measured on.

### 10.6 The pose-stage decisions this ledger may have mis-stated
The LM track wrote §3.1–§3.3 from `RESULTS.md` and `ORIENTATION.md`, not from the pose track.
**Specifically check:** that FP16 was chosen on energy *and* became a fitting requirement only later
(§3.2); that the native crop was a deliberate deviation from OpenASL's recipe rather than an
accident (§3.3); and that P5/P9/P10/P11 in §5 are genuinely un-run rather than run and unreported.

### 10.7 Anything the pose track decided that is not in this file at all
The LM track has no visibility into: engine build settings beyond TF32, batching on the pose side,
JPEG decode choices, how `max_frames=900` was picked, or the thread model in `demo_server.py`.
**Add a §3.11 for anything that was a real choice with a real alternative.**

---

*Authority: `results/RESULTS.md` over this file, always. This file over `ORIENTATION.md` and
`RESEARCH-STORY-2026-10-03.md` where they conflict on a **decision**, because both predate
§L26–§L30 — but those two remain the better read for the map and the narrative respectively.*
