# LM track: complete findings, ablations and lessons — 2026-09-30

**Who this is for.** The pose track, the course report, and anyone (human or LLM) picking up the LM
track cold. It is self-contained: every number states its metric, its denominator and the file it
came from, because the two most expensive mistakes on this project both came from comparing numbers
that looked comparable and were not.

**How to read the numbers.**
- Every accuracy figure is real model output on real OpenASL clips. Nothing is estimated, scaled or
  extrapolated. Where something is composed from separately measured parts, it says so.
- Every latency and energy figure comes from a real Jetson run at 15 W, measured by the pose track.
- "Emulated" frame rates thin already-extracted 30 fps poses. A real 16 fps capture differs in
  exposure and motion blur. This is labelled everywhere it appears.
- Deltas carry 95 % CIs from a paired clip-level bootstrap (1000 resamples) unless stated otherwise.
  Where a CI is missing, the text says so — that is a gap, not an oversight.
- Retracted claims are struck through and kept in place, so nothing is rebuilt on them.

---

## 1. The system

Sign-language video → RTMPose/RTMW keypoints → Uni-Sign ST-GCN pose encoder → mT5-base → English
text. Deployed on a course-owned Jetson Orin Nano at 15 W (power mode 0, fixed — no sudo on that
board). The deliverable is an **accuracy-versus-energy frontier**.

**Model.** 587.7 M parameters = mT5-base 582.4 M + pose stack 5.35 M. Four keypoint parts (body 9,
left hand 21, right hand 21, face 18); the hands share weights. Encoder input is the prefix
*"Translate sign language video to English: "* (8 tokens) followed by up to 256 pose tokens.

**Work split.** LM track (this document) owns everything from pose tensors to text: pruning,
quantization, decoding, frame-rate sensitivity, adaptation training. The pose track owns keypoint
extraction and **all** Jetson runs. Colab is used only for training.

---

## 2. Baseline: the model reproduces, and 22 BLEU-4 is what state of the art looks like

| | BLEU-1 | BLEU-4 | ROUGE-L |
|---|---|---|---|
| Paper, OpenASL test, pose-only | 49.10 | 22.67 | 42.77 |
| Recomputed from the authors' released predictions | 49.09 | 22.64 | 42.83 |
| Our run, Colab T4, bf16, beam 4, cap 100 | 48.83 | **22.53** | 42.68 |
| Our run, Mac CPU, fp32, released checkpoint | 49.61 | **23.16** | 43.17 |

976 test sentences. The −0.14 gap to the paper is within the loader's random frame subsampling for
long clips; treat **22.5–23.2 as the FP32 reference band**.

**What 22 BLEU-4 actually means.** First test clip — reference: *"I am Ed Bosson, and I am 67 years
old and now retired."* Prediction: *"Hello, I'm Howard Rosenblum, and I'm a fourteen-year-old
immigrant."* Fluent, grammatical, and wrong in every detail. Worth keeping in mind before treating a
0.5 BLEU-4 difference as meaningful to a user.

**Gotcha that cost real time:** the released prediction files prefix every line with
`sample: <clip>, ground-truth: ` / `prediction: `. Scoring them raw gives a bogus 58.7 BLEU-4.

---

## 3. Compression ablations

All on 976 OpenASL test clips, Mac CPU, unless stated. Paired bootstrap, 1000 resamples.

### 3.1 Vocabulary pruning — the best single lever

A token census over OpenASL keeps **26,078 of 250,112 mT5 tokens (10.4 %)**. Coverage of token
occurrences: top 5 K = 94.6 %, 10 K = 98.1 %, 20 K = 99.7 %. `shared` embedding and `lm_head` are
sliced, the tokenizer is wrapped to remap ids both ways.

| | params | checkpoint (bf16) | BLEU-4 | ROUGE-L |
|---|---|---|---|---|
| full | 587.7 M | 1187 MB | 23.16 | 43.17 |
| **vocab-pruned** | **243.6 M (−59 %)** | **571 MB (−52 %)** | **22.87** | 42.98 |

**Δ −0.28 BLEU-4, 95 % CI [−0.63, +0.05]**, P(pruned < full) = 0.95. Report this as *"not established
as a loss"*, not as "−0.28", which reads like a measured penalty.

655 of 976 predictions are identical; 321 change because dropping 224 K tokens from the softmax
shifts beam scores even where the argmax doesn't move (greedy on a single clip was identical).

### 3.2 Weight-only INT8 — works, but was dropped

**Precision contract, stated because the name caused a real misreading.** Weights are int8, the
per-row scale is *stored* fp16 and cast up before use, and activations, matmul and accumulation are
all **fp32** (`compute_dtype=torch.float32`). No activation is ever fp16. This was labelled "W8A16"
until 2026-09-28 and renamed **W8A32**.

Per-row symmetric int8 (absmax/127) on every mT5 2-D weight ≥ 1e5 elements: 220 tensors, 278.3 M of
285.2 M stored values. Pose stack, layer norms and relative-attention bias stay fp32. Max per-tensor
relative error 4.2e-3.

| | file MB | BLEU-4 | ROUGE-L | Δ BLEU-4 |
|---|---|---|---|---|
| released, full vocab | 1187 | 23.16 | 43.17 | — |
| pruned | 571 | 22.87 | 42.98 | −0.28 [−0.63, +0.05] vs released |
| **pruned + W8A32** | **293** | 22.79 | 43.00 | **−0.09 [−0.34, +0.14]** vs pruned, P=0.77 |

Stacked cost versus the released model: **−0.37 BLEU-4 [−0.76, −0.02] for a 4.05× smaller file.**
182 of 976 sentences change wording, so the quantization is not invisible per sentence, only in
aggregate.

**Why it was dropped anyway.** The pose track measured the decoder as **host-bound**, not GEMM-bound:
on the board the encoder is 39–50 ms while the decoder is 1062–1538 ms, dominated by per-step
binding overhead rather than the 238 M-parameter matmuls. Our own runtime agrees — `--w8-runtime
int8` is **10× slower** per decoder step (2139 vs 197 ms) because 217 layers dequantize per token.
Quantizing weights buys memory.

~~And memory is not binding at 2.577 GB peak on an 8 GB board.~~ **CORRECTED by the pose track
2026-09-30: memory IS binding, at load rather than in steady state.** The pruned-checkpoint load path
peaks near **4.2 GB** — more than loading the *full* checkpoint costs, because `load_model`
materialises the 250 K-vocab mT5 and then slices it — against `MemAvailable` capping around 5.3 GB.
Four probes: the TRT engine alone works at 3988 MB free; **the LM alone fails** at 3863 MB and again
at 4641 MB after maximum reclaim. It is why their end-to-end sustained run has never completed.

**This partly reopens INT8.** Our justification was "buys memory, and memory isn't binding". The
second clause is wrong, so a 293 MB W8A32 checkpoint would help materially — *if* loaded from a
pre-pruned directory that avoids the 250 K materialisation. That is exactly what their
`attach_pruned_tokenizer` fast path does, and it is unverified. The host-bound-decoder argument
against INT8 *for latency* still stands; the memory argument does not.

**Correction to the pose track's stated reason.** Their handoff argued INT8 was dead because W8A16
keeps activations in fp16 and fp16 breaks mT5. The premise is wrong about our implementation (see
the precision contract above), and the empirical disproof is cleaner: our INT8 config scored **22.79
BLEU-4**, which is impossible under the fp16 failure mode. Their *second* reason — host-bound decoder
— is correct and sufficient on its own. **Right decision, wrong justification.**

### 3.3 mT5 in fp16 is broken (pose track measurement, board)

| engine set | tokens vs PyTorch | max abs Δlogprob | verdict |
|---|---|---|---|
| FP32 | **identical** | 5.6e-4 | passes |
| FP16 | **broken**: 64× token 0 | 10.2 | unusable |
| BF16 | diverges at step 3 | 1.1 | unusable |

The fp16 failure is diagnostic, not random: every step returns token 0 at logprob −10.169, which is
exactly −ln(26078) — a uniform distribution over the pruned vocabulary, i.e. saturated logits. This
is the documented T5/mT5 fp16 overflow. **Consequence for training: no fp16 autocast anywhere.**

### 3.4 Decoder strategy — a real accuracy axis, not a free lever

| decoding | BLEU-4 | ROUGE-L | Δ vs beam 4 | eval wall (CPU) |
|---|---|---|---|---|
| beam 4 | 22.87 | 42.98 | — | 919 s |
| beam 2 | 22.06 | 42.22 | **−0.81 [−1.29, −0.39]** | 529 s (1.7× faster) |
| greedy | 20.88 | 41.56 | **−2.00 [−2.63, −1.41]** | 342 s (2.7× faster) |

Greedy costs a full 2 BLEU-4 — not "well under a point" as the original plan assumed. This is also
the number that prices the TensorRT decision: the TRT path is **greedy-only** (`trt_decode.py`
hardcodes `num_beams=1`; beam through explicit-KV-cache engines needs per-beam cache reordering), so
"PyTorch encoder + TRT greedy decoder", the fastest configuration measured, costs 2 BLEU-4 by
construction.

### 3.5 Output-length cap — free at 64

| cap | BLEU-4 | Δ vs cap 100 | preds at/over cap | sentences changed |
|---|---|---|---|---|
| 100 | 22.87 | — | 0 | — |
| **64** | **22.87** | **0.00 (bit-identical)** | 0 / 976 | **0** |
| 48 | 22.73 | −0.14 [−0.30, −0.03] | 39 / 976 | 28 |

Prediction length: mean 21.2 mT5 tokens, median 19, p95 44, max 62. Cap 64 is the deployment default
and is free. **Scope restriction added 2026-09-30: this was measured on the *un-adapted* model. An
adapted checkpoint visibly hits the cap** (see §8.4), so the equivalence is not established for
adapted models.

### 3.6 Frame rate — measured twice, and the first measurement was wrong

**First attempt (L7.1), using a length cap as a frame-rate proxy:**

| cap L (≈ fps) | clips subsampled | BLEU-4 | Δ vs 256 |
|---|---|---|---|
| 256 (30) | 305 / 976 | 22.79 | — |
| 205 (24) | 410 | 22.62 | −0.17 [−0.61, +0.26] |
| 137 (16) | 602 | 21.17 | −1.61 [−2.36, −0.89] |
| 103 (12) | 716 | 19.52 | −3.26 [−4.10, −2.41] |
| 68 (8) | 819 | 14.76 | −8.02 [−9.20, −6.86] |

**Why this was wrong: OpenASL is not one frame rate.** Measured as frames ÷ duration over 400 test
clips, **73 % are 30 fps, 21 % are 24 fps**, the rest 25/31/60. A single length cap therefore does not
emulate a slower camera — it only touches long clips, and hits those harder.
`fps_ratio_for_clip` derives each clip's own rate from its frame count and duration and keeps
`round(duration × target_fps)` frames.

**Corrected measurement (L7.2), true per-clip emulation:**

| target fps | BLEU-4 | ROUGE-L | Δ BLEU-4 |
|---|---|---|---|
| source (30/24) | 22.79 | 43.00 | — |
| **24** | 22.80 | 43.22 | **+0.02 [−0.38, +0.45]**, P=0.45 |
| **16** | 21.66 | 41.56 | **−1.12 [−1.71, −0.48]** |
| 12 | 20.26 | 39.15 | −2.52 [−3.22, −1.85] |

**24 fps is free.** 16 fps costs ~1.1 BLEU-4 un-adapted (not 1.6 — the cap overstated it).

### 3.7 Does the frame-rate cost depend on INT8? (difference-in-differences)

All of §3.6 was measured on the quantized model, so the penalty could have been an INT8 artefact.
`unisign/did_ci.py` compares both penalties on **identical bootstrap resamples**:

| 16 fps penalty measured on | Δ BLEU-4 |
|---|---|
| pruned, no INT8 | −1.33 [−2.00, −0.64] |
| pruned + W8A32 | −1.12 [−1.71, −0.48] |
| **interaction** | **+0.21 [−0.14, +0.57]**, P(<0) = 0.13 |

The interaction straddles zero and lies inside the 0.6 sensitivity band. **Quantization does not
change the cost of dropping frame rate.** Stated honestly: *no interaction large enough for us to
see*, not "the levers are independent" — a ±0.6 CI cannot rule out an interaction the size of the
INT8 penalty itself.

---

## 4. The accuracy surface (976 test clips, pruned FP32, cap 64)

Full 3×3 grid, with **one bootstrap resample draw scoring every cell**, so all nine are mutually
comparable rather than only comparable to the reference (`unisign/grid_table.py`).

| decoder | fps | BLEU-4 | ROUGE-L | Δ BLEU-4 vs beam 4 @ source |
|---|---|---|---|---|
| **beam 4** | source | **22.87** | 42.98 | reference |
| beam 4 | **24** | **22.80** | 43.13 | **−0.07 [−0.53, +0.39]** |
| beam 4 | 16 | 21.54 | 41.35 | −1.33 [−2.00, −0.64] |
| beam 2 | source | 22.06 | 42.22 | −0.81 [−1.29, −0.39] |
| beam 2 | 24 | 21.93 | 42.29 | −0.94 [−1.45, −0.39] |
| beam 2 | 16 | 21.19 | 40.89 | −1.68 [−2.35, −0.98] |
| greedy | source | 20.88 | 41.56 | −2.00 [−2.63, −1.41] |
| greedy | 24 | 20.66 | 41.18 | −2.22 [−2.83, −1.60] |
| greedy | 16 | 19.83 | 39.53 | −3.04 [−3.74, −2.36] |

**The two levers are additive.** Adding the beam-4 frame-rate cost (−1.33) to the source-rate greedy
cost (−1.99) predicts 19.55 for greedy @ 16 fps; measured is **19.83**, i.e. +0.28 sub-additive —
inside the 0.6 band, so no detectable interaction. Treat the grid as additive for planning and quote
the measured cell when reporting.

---

## 5. Board energy (measured by the pose track, 15 W, mode 0, INA3221 on VDD_IN)

### 5.1 LM stage — encoder length is nearly useless to shrink

| beams | frames used | encoder ms | decoder ms | J/sentence |
|---:|---:|---:|---:|---:|
| 1 | 215 | 49 | 1228 | 8.88 |
| 2 | 215 | 50 | 1447 | 10.26 |
| 4 | 215 | 49 | 1538 | 11.04 |
| 1 | 137 | 39 | 1236 | 8.55 |
| 4 | 137 | 41 | 1459 | 9.89 |
| 1 | 68 | 41 | 1168 | 7.79 |
| 4 | 68 | 41 | 1370 | 8.94 |

Cutting T from 256 to 68 — a 73 % reduction in pose frames — saves the LM only 8.88 → 7.79 J (12 %).
**Reduced frame count is a pose-stage saving, not an LM one.**

### 5.2 Pose stage — frame rate scales properly

| fps | mJ/frame | **J per second of video** | vs 30 fps |
|---:|---:|---:|---:|
| 30 | 141.6 | **4.25** | 1.00× |
| 24 | 139.9 | **3.36** | **0.79×** |
| 16 | 141.8 | **2.27** | **0.53×** |
| 12 | 139.5 | 1.67 | 0.39× |
| 8 | 145.9 | 1.17 | 0.27× |

`mJ/frame` is flat by construction; **`J per second of video` is the only comparable column** across
rates.

### 5.3 First on-device end-to-end run (M1)

9052.8 ms ± 335.6 per sentence (pose 6640.7, LM 2386.5), **50.37 J/sentence**, peak GPU 2.577 GB,
8.51 s of video. Total is 1.06× slower than real time; the pose stage alone is 0.78×. **The LM is
what breaks real time.** LM load time is 64.4 s, which is what an audience notices in a live demo.

### 5.4 CPU/GPU contention under the 15 W cap

Two independent CPU-only stages slowed by **the identical factor to three digits** (×1.281) when GPU
load rose, with `cpu0_MHz` dropping ×1.165. **Every standalone LM latency figure is an optimistic
bound** on what it gets in the real pipeline, and the host-bound decoder is exactly the work that
suffers when the CPU downclocks.

---

## 6. The frontier (composed, `unisign/frontier.py`)

System joules are **composed** from two separately measured stages — `pose_J_per_s × clip_seconds +
LM_J` — **not measured end to end**. Validated against the one real end-to-end run: composition
predicts 47.2 J where **50.37 J** was measured, so it runs **~6 % low**.

**Two biases in the absolute column, in opposite directions.** Neither affects the *relative*
comparisons across cells — every cell shares the same clip and rates — and the relative ordering is
the deliverable. But these joules are not a typical clip's:

1. **~6 % low.** The 3.16 J the composition misses. ~~Attributed to the §5.4 contention.~~
   **CORRECTED by the pose track 2026-09-30:** their M1 run measured *no* contention (25.10 ms/frame
   end-to-end vs 25.03 standalone), because the implementation is sequential — all frames, then the
   LM — so the stages never overlap. Likelier: the convert step (25.7 ms, unmodelled here), the idle
   floor during the 64 s LM load, or warm-up. Untested.
2. **~7 % high for a median clip.** The pose energy rate came from one clip whose crop is at the
   **83rd percentile** of crop area across the 931-clip split. Pose cost is ~13.97 ms fixed plus
   ~11.13 ms scaling with crop area, so a median clip is ~7 % cheaper per frame and a p10 clip ~27 %.
   Meanwhile `CLIP_S` is our split's *mean duration* — so the absolute column already mixes a duration
   from our data with an energy rate from their clip.

| decoder | fps | BLEU-4 | ROUGE-L | system J | vs ref | |
|---|---|---|---|---|---|---|
| beam 4 | source | 22.87 | 42.98 | 43.4 | +0 % | **PARETO** |
| beam 2 | source | 22.06 | 42.22 | 42.6 | −2 % | dominated |
| greedy | source | 20.88 | 41.56 | 41.2 | −5 % | dominated |
| beam 4 | **24** | **22.80** | 43.13 | **36.1** | **−17 %** | **PARETO** |
| beam 2 | 24 | 21.93 | 42.29 | 35.5 | −18 % | **PARETO** |
| greedy | 24 | 20.66 | 41.18 | 34.3 | −21 % | dominated |
| beam 4 | **16** | 21.54 | 41.35 | **27.0** | **−38 %** | **PARETO** |
| beam 2 | 16 | 21.19 | 40.89 | 26.5 | −39 % | **PARETO** |
| greedy | 16 | 19.83 | 39.53 | 25.7 | −41 % | **PARETO** |

### The central finding: frame rate is 10–100× the more efficient lever

| lever | system energy | Δ BLEU-4 | **J per BLEU-4 point** |
|---|---|---|---|
| 30 → 24 fps | −16.7 % | −0.07 | **103.5** |
| 30 → 16 fps | −37.9 % | −1.33 | **12.3** |
| beam 4 → beam 2 | −1.8 % | −0.81 | **1.0** |
| beam 4 → greedy | −5.0 % | −1.99 | **1.1** |

Beam width is the *LM's* lever but the **worst** system-level lever: greedy saves 5 % of system
energy for 2 BLEU-4. Frame rate saves 17 % for nothing measurable. The reason is in §5.1 and §5.2 —
the decoder is host-bound so beam count barely moves LM joules, the LM is only ~25 % of system
energy, and pose energy scales with frame count.

**This corrects the pose track's framing.** Their §0b concludes "beam width is the LM's real lever,
and that single trade is the accuracy–energy frontier." True of the LM in isolation, false of the
system.

**Recommended operating point: beam 4 @ 24 fps** — 17 % less system energy, BLEU-4 −0.07
(statistically tied), ROUGE-L +0.15. Free.

---

## 7. Validation work

### 7.1 The 16 fps result did not replicate on dev — and why that mattered

Re-ran the whole battery on the 967-clip dev split, which we had not selected on.

| knob | test (n=976) | dev (n=967) |
|---|---|---|
| reference (beam 4, source) | 22.87 / 42.98 | 23.11 / 42.90 |
| 24 fps | −0.07 BLEU-4 | **+0.26** |
| **16 fps** | **−1.33 [−2.00, −0.64]** | **−0.34 [−0.97, +0.53]**, P=0.77 |
| beam 2 | −0.81 | −0.97 |
| greedy | −2.00 | −2.37 |

**Beam width replicated cleanly.** The 16 fps BLEU-4 cost did not.

First hypothesis — different clip distributions — was **tested and rejected**:

| | test | dev |
|---|---|---|
| frames, mean / median | 217.3 / 171.5 | 199.8 / 158.0 |
| seconds, mean | 7.61 | 6.94 |
| source fps, mean | 29.07 | 29.11 |
| over the 256-frame cap | 31.25 % | 28.65 % |

Too similar to explain a 1-point divergence.

**The actual answer: it is a metric artefact.** Looking at ROUGE-L instead:

| 30 → 16 fps | BLEU-4 | **ROUGE-L** |
|---|---|---|
| test | −1.33 | **−1.63** |
| dev | −0.34 | **−1.30** |

**ROUGE-L replicates — same sign, 0.33 apart.** BLEU-4 is 4-gram precision; a handful of sentences
flipping a single 4-gram match moves it, and at n≈970 that is the whole effect size. ROUGE-L is
longest-common-subsequence recall and averages over far more evidence per sentence.

**Standing consequence: any effect below ~1 BLEU-4 at n≈1000 should be carried by ROUGE-L, and
BLEU-4 quoted with the caveat that it cannot resolve it.**

This got worse at smaller n. At **n=300**, the same un-adapted model scored **+1.10 BLEU-4 higher at
16 fps than at source rate** — the opposite sign of a known effect. That is why every later
evaluation uses the full 967-clip dev split.

### 7.2 Honest framing of the dev work

This validates that **the same knobs would have been chosen on dev**. It is **not** "selected on
held-out dev", because test results were seen first.

---

## 8. Adaptation training (guide Block 4)

### 8.1 The vocabulary was leaking the test set

The keep set in §3.1 was built from a census over **train + dev + test** (98,419 sentences). A model
whose embedding matrix and LM head were *selected using the test set* has seen the test set.
`unisign/vocab_census.py` rebuilds it from **train + dev only** (97,443 sentences):

| | keep ids | checkpoint |
|---|---|---|
| train+dev+**test** census | 26,078 | 571 MB |
| **train+dev census** | **26,025** | **570 MB** |

**53 tokens dropped, 0 added.** Each occurs **zero** times outside test. They include **`▁Bitcoin`**
— the key content word of this project's standard demo clip, present in the deployed vocabulary only
because it appears in the test split.

**Validated on dev:** corrected checkpoint scores 23.13 / 42.93 against the leaky one's 23.11 /
42.90. Identical within noise, as expected since none of the 53 occur in dev.

**Still outstanding: the test-split score of the corrected checkpoint.** Those 53 tokens occur 55
times across 46 of 976 test sentences, so a small drop from 22.87 is expected, and that would be the
true unleaked number. **Every test figure in this document still comes from the leaky keep set.**

### 8.2 The first adaptation runs made the model worse — and the control proved why

Initial recipe used the authors' `--label-smoothing 0.2` with no warmup (n=300 dev clips):

| run | train/eval rate | before | after | Δ BLEU-4 |
|---|---|---|---|---|
| 16 fps | 16 fps | 18.27 | 16.57 | **−1.70** |
| **control** | **source** | 17.17 | 16.13 | **−1.04** |

**The control degraded too.** With no distribution shift to adapt to, fine-tuning still cost 1.04
BLEU-4 — so the 16 fps number was never evidence about adaptation, only about a broken recipe.

### 8.3 Recipe sweep (control condition, 967 dev clips)

Un-adapted baseline **23.13 / 42.93**.

| recipe | BLEU-4 | ROUGE-L | Δ BLEU-4 | Δ ROUGE-L |
|---|---|---|---|---|
| ls 0.2, lr 1e-4, no warmup | — | — | −1.04 *(at n=300)* | not measured |
| ls 0.0, lr 1e-4, warmup 0.1 | 22.75 | 43.51 | −0.38 | +0.58 |
| **ls 0.0, lr 1e-5, warmup 0.1** | **23.31** | **43.63** | **+0.18** | **+0.70** |

**Label smoothing was the problem.** At 0.2 over a 26 K vocabulary the reported training loss sits at
~3.35 and barely moves; at 0.0 the true cross-entropy is ~0.36. **The 3.35 was mostly the smoothing
floor, not the model's fit** — a constant being read as a training curve for two sessions.

At lr 1e-5 fine-tuning **improves** the model on both metrics. This is the first evidence the C8
harness actually trains rather than merely runs.

**Caveat:** the isolating control (ls 0.2 at lr 1e-5, which would separate smoothing from learning
rate) was killed by a Colab disconnect after 10 log lines. "Label smoothing was the problem" is the
best available reading, not an isolated result — the working recipe changed three things at once.

### 8.4 Adapted vs un-adapted at 16 fps (967 dev clips, seed 42)

| | un-adapted | adapted | Δ BLEU-4 | Δ ROUGE-L |
|---|---|---|---|---|
| source rate | 23.13 / 42.93 | 23.31 / 43.63 | +0.18 | +0.70 |
| **16 fps** | 22.79 / 41.59 | **23.19 / 42.60** | **+0.40** | **+1.01** |

- **Adaptation helps more under frame-rate shift than without it.** Difference-in-differences
  **+0.22 BLEU-4, +0.31 ROUGE-L** — the part attributable to frame-rate adaptation specifically.
- **It recovers most of the frame-rate loss.** Un-adapted, 16 fps costs **−1.34 ROUGE-L**. Adapted,
  the 16 fps model sits **−0.33 ROUGE-L** below the un-adapted full-rate baseline — about **75 %
  recovered** — and **+0.06 BLEU-4 above** it.
- **This moves the frontier.** 16 fps / beam 4 is a **38 % system energy saving** (§6). Un-adapted
  that carries a real accuracy cost; adapted, the cost nearly vanishes.
- The un-adapted 16 fps figure independently reproduces the Mac result (22.79 / 41.59 vs 22.77 /
  41.60) on different hardware with a different checkpoint.

**Output truncation, found by reading the samples.** One adapted prediction ends mid-phrase — *"where
they can establish a"* — at `max_new_tokens 64`, while the un-adapted version completes the sentence.
§3.5's cap equivalence was measured on the un-adapted model only. Truncation costs n-gram matches, so
**+0.40 / +1.01 may be an underestimate.** A cap-100 re-evaluation is queued.

### 8.5 What is NOT established about §8.4

1. **No confidence intervals.** `train_adapt.py`'s built-in eval logs summary metrics, not the 967
   predictions, so none of these deltas can be bootstrapped. Every other comparison in this document
   carries a paired CI; these do not.
2. **One seed.** Guide row 4.3 (seed variance) has not run. "+0.40 vs +0.18" is uninterpretable
   without the seed-to-seed spread, which is a *different* noise source from the clip-resampling CI.
3. 20,000 of 96,477 train clips, one epoch.
4. Frame-rate emulation is a proxy for a real slower camera.

---

## 9. Retracted or superseded

- ~~"16 fps is free (Δ −0.15, CI [−3.20, +2.89])"~~ **SUPERSEDED.** That was n=30, a CI ~6 BLEU-4
  wide. At n=976 and n=967 the ROUGE-L cost is −1.63 and −1.30. The pose track's §0c conclusion that
  "at 16 fps and above there is nothing to recover" does not survive, and neither does §0d's
  "16 fps / beam 4, ~38 % system saving at no measured accuracy cost" — the energy half is right.
- ~~"INT8 is dead because W8A16 inherits mT5's fp16 failure"~~ **WRONG PREMISE** (§3.2). Right
  decision, different reason.
- ~~"16 fps costs 1.6 BLEU-4"~~ superseded by true per-clip emulation: ~1.1–1.3 (§3.6).
- ~~"The pose front-end costs 6.33 BLEU-4"~~ **RETRACTED by the pose track** at n=30: +1.61,
  CI [−2.57, +5.94], not established.
- ~~"Synthetic verification at T=264"~~ removed; the ONNX export is verified on real poses only.

---

## 10. Lessons

### Methodological

1. **Run the no-shift control before the intervention.** Two sessions were spent interpreting a
   −1.70 BLEU-4 "adaptation failure" that was a broken training recipe. The control that settled it
   cost 12 minutes and should have come first. An intervention without a control cannot distinguish
   "it didn't work" from "the procedure is broken."
2. **Match the metric to the effect size and the sample.** BLEU-4 cannot resolve sub-1-point effects
   at n≈1000, and at n=300 it flipped the sign of a known effect. Most of a day went into
   "why doesn't this replicate" when the answer was that we were reading the wrong instrument.
3. **A flat loss curve may be a floor, not a fit.** Label smoothing 0.2 over a 26 K vocabulary pins
   the reported cross-entropy near 3.35 regardless of learning. The true loss was 0.36. We read a
   constant as a training curve.
4. **Validate on data you did not select on, and say honestly what that buys.** Dev validation says
   *the same knobs would have been chosen*; it does not say *selected on held-out data*, because test
   came first.
5. **Share bootstrap resample draws across cells.** The 3×3 grid and the difference-in-differences
   both score every cell on one draw, which makes cells mutually comparable instead of only
   comparable to a reference — and is strictly more sensitive than differencing independent CIs.
6. **State negative results as sensitivity, not independence.** "No interaction large enough for us
   to see at ±0.6" — not "the levers are independent."
7. **Check the denominator before comparing.** A 5-clip BLEU-4 and a 976-clip BLEU-4 differ by ~7
   points on the subset alone. No small-n BLEU figure in this project is an absolute score.
8. **Read the sample outputs, not only the metrics.** The `max_new_tokens` truncation in §8.4 is
   invisible in the aggregate numbers and was found by comparing three sentences side by side.
9. **Selection leaks through unexpected channels.** The vocabulary was chosen using the test split.
   Nothing in the training loop touched test data, yet the deployed model's output layer was shaped
   by it.

### Engineering

10. **A proxy is not the thing.** A length cap only truncates long clips; a frame-rate emulation must
    derive each clip's own source rate. OpenASL is 73 % 30 fps and 21 % 24 fps, so there is no single
    `--src-fps`.
11. **Definitions must match across tracks.** `unisign_infer.py` had no `--fps`, so board energy rows
    would have used a different "16 fps" than the BLEU rows, and the frontier cells would not have
    lined up. Caught before it caused damage.
12. **Naming is part of the contract.** Calling weight-only INT8 "W8A16" led the other track to
    conclude the work was dead. A precision contract now sits in the code and in both reports.
13. **Existence is not validity.** Skipping extraction because a file exists let a truncated pkl
    survive a killed writer and crash training 2250 steps in. Validate against a recorded size.
14. **Idempotent scripts beat careful sequencing.** Colab lost `/content` three times in one session.
    Every step now skips work whose output exists, so a disconnect costs only the step in flight.
15. **Line-buffer output from background jobs.** Redirecting unbuffered stdout to a file made working
    jobs indistinguishable from hung ones, three times.
16. **Sparse random access to a remote archive can cost more than downloading it.** A 4 MB read-ahead
    buffer discarded on every non-adjacent seek meant 4.22 MB transferred per 0.5 MB clip. Above
    ~20 % density, downloading the whole 32 GB archive transfers less.
17. **Depend on reproducible sources, not on a shared drive.** Google Drive filled mid-session and
    the project folder was cleared while freeing space, taking the released checkpoint with it. Every
    input now comes from git or a public HuggingFace download; the environment rebuilds in ~20 min.
18. **Checkpoint mid-epoch, atomically.** Epoch-only checkpointing loses hours to a disconnect, and a
    non-atomic write can truncate the very file that was meant to protect the run.
19. **CPU-only development hides GPU bugs.** A CPU lookup table indexed by CUDA tensors ran fine for
    weeks of Mac evaluation and failed on the first Colab run.

---

## 11. Open work

| # | what | cost | why |
|---|---|---|---|
| 1 | Paired CIs for the §8.4 2×2 (`eval_openasl.py` → `bootstrap_ci.py`) | ~20 min | the only result here without error bars |
| 2 | Seed variance, 3 more seeds (row 4.3) | ~55 min | "+0.40 vs +0.18" is one draw |
| 3 | Test-split eval of the leak-free checkpoint | ~15 min | every test figure above uses the leaky keep set |
| 4 | Cap-100 re-eval of adapted models | ~5 min | §8.4 may be an underestimate |
| 5 | Measured end-to-end board energy at a second corner | pose track | our frontier is composed and ~6 % optimistic |
| 6 | Adaptation at 12 fps | ~20 min | 12 fps costs 2.5 BLEU-4 un-adapted; worth trying if 16 fps holds |
| 7 | Full 96 K clips / multiple epochs | ~65 min/epoch | may increase the gain |

---

## 12. File index

**Analysis** — `unisign/grid_table.py` (3×3 surface, shared draws), `unisign/did_ci.py`
(difference-in-differences), `unisign/frontier.py` (composed frontier), `unisign/bootstrap_ci.py`
(paired CI), `unisign/vocab_census.py` (keep-set census).

**Pipeline** — `unisign/model.py` (pruning, `PrunedTokenizer`), `unisign/quant.py` (W8A32 with
precision contract), `unisign/train_adapt.py` (adaptation, mid-epoch atomic checkpointing),
`unisign/eval_openasl.py`, `unisign/unisign_infer.py` (`--fps`/`--src-fps`).

**Colab** — `colab_setup.py` (idempotent environment rebuild), `colab_block4.py` (Block 4 batch).

**Results** — `results/RESULTS.md` (primary record), `results/colab_runs/` (training logs),
`results/openasl_vocab_keep_ids_traindev.json`, `results/vocab_keep_diff.json`,
`results/eval_*.json`.

**Reports** — `ABLATION-REPORT.md`, `REPLY-LM-TRACK-2026-09-28.md` (corrections owed to the pose
track), `PROJECT-GUIDE.md` (row numbering), `WORKSPLIT.md`.

**Checkpoints** (gitignored) — `weights/openasl_pose_only_slt_pruned_traindev.pth` (leak-free base),
`weights/adapt_fps16_lr1e5_seed42.pt` (the §8.4 adapted model; GPU reductions are non-deterministic,
so a retrain is similar but not identical, and this file is the one that produced those numbers).

---

## 13. Corrections received from the pose track, 2026-09-30

Their `SYNC-REPLY-POSE-2026-09-30.md` accepted four of our five corrections and returned five. Three
changed results above and are applied in place (§3.2 memory, §6 the two frontier biases). The other
two are recorded here.

**Our "98,419 poses extracted" unblocks less than we implied.** Those are the **authors'** poses. They
unblock the **frame-rate** and **face-group** adaptations immediately. They do **not** touch the
extractor gap: adapting around that needs train-split poses from *our* extractor, which requires the
~97 K source **videos** — pose pkls do not substitute, and at their measured ~8 s/clip that is still
days of fetching. Keep the two separate in the report.

**We cannot ask them to verify loader changes.** Their Mac has no torch, transformers, cv2 or
safetensors, and their board has been unreachable since 2026-09-28. Anything needing a Python
environment is ours to run.

**Their ask 4, answered:** the pruned TensorRT engines **do** need re-exporting. `results/RESULTS.md`
L8.1 records the three ONNX graphs as exported from `weights/mt5-base-openasl-pruned`, **vocab
26,078** — the leaky keep set (§8.1). Their L8.2 *latency* rows (19.2 ms/token, 2.9× the pruned
PyTorch decoder, tokens identical to PyTorch) are unaffected, because token-identity was checked
against a PyTorch model with the same leaky vocabulary. Any claim about that engine's **output
quality** describes a model whose vocabulary was selected using the test set.

**What they found that we had missed, in both codebases:** the hardcoded source frame rate. Their
`subsample_pkl.py` and `03_infer_frames.py --keep-fps` assumed `src_fps=29.97`, so **6 of the 30 clips
in their frame-rate curve were off-rate** — rows labelled "16 fps" contained clips thinned to 12.8.
Their split measures 76.5 % at ~30 fps and 22.2 % at ~24, replicating our 73/21. They also verified
our two `--fps` definitions are the same function (`round(duration × target)`) on five cases, so the
frontier's cells line up across tracks.
