# Ablation report and critical re-evaluation

**Project:** pose-based sign-language translation on a 15 W Jetson Orin Nano
**Date:** 2026-09-26 · **Track:** language model (Stage 3), with the pose stage where it interacts
**Purpose:** state every ablation performed so far in full, then audit them — what the evidence
actually supports, what is confounded, and what cannot yet be claimed.

Read §3 before quoting any number from §2. Several results are weaker than their table looks.

---

## 1. Protocol

**What varies, what doesn't.** The test set is the OpenASL test split, 976 clips, fixed across every
row. Poses are the authors' released keypoints (`pose-rtmpose-192`), so the pose stage is *held
constant* for all LM ablations — this is a language-model study conducted on frozen pose inputs, not an
end-to-end study. Metric code is the authors' own `translation_performance` (sacrebleu 13a tokenizer,
ROUGE-L F), so our numbers are directly comparable to the published ones.

**Significance.** BLEU is a corpus-level metric, so per-sentence significance tests don't apply. We use
a **paired bootstrap** ([unisign/bootstrap_ci.py](unisign/bootstrap_ci.py)): resample the 976 clips with
replacement 1000 times, recompute corpus BLEU-4 for both systems on each resample, and report the
distribution of the difference. Paired, because both systems see the same resampled clips, which
removes clip-difficulty variance and is far more sensitive than comparing two independent intervals.

**Noise floor.** Three legitimate runs of the *same checkpoint* — the authors' evaluation on a T4 in
bf16, ours on a T4, and our standalone loop on Mac CPU in fp32 — span **22.53 / 22.53 / 23.16 BLEU-4**,
a spread of 0.6. We treat 0.6 BLEU-4 as the threshold below which an effect needs a confidence interval
to be believed at all. ⚠️ See §3.2: this band is *not* run-to-run noise, and the distinction matters.

**Decoding defaults** unless a row says otherwise: beam 4, `max_new_tokens` 64, frame cap 256,
deterministic (uniform) frame subsampling, fp32 on Mac CPU.

---

## 2. The ablations

### 2.0 Baseline chain

Each ablation was measured against the configuration immediately before it, not against a fixed
reference. The chain:

```
released checkpoint (587.7 M, 1187 MB, 23.16 BLEU-4)
   └─ A. vocabulary pruning        → 243.6 M, 571 MB, 22.87
        └─ B. weight-only INT8     → 293 MB,          22.79
             └─ C. frame-rate cut  → 24 fps 22.80 · 16 fps 21.66 · 12 fps 20.26
   ├─ D. decoding (beam/greedy), measured on the pruned model
   └─ E. output-length cap, measured on the pruned model
```

This is a **path through the configuration space, not a factorial design.** §3.1 explains why that
limits what the deltas mean.

---

### 2.A Vocabulary pruning (L5.1–L5.3)

**Hypothesis.** mT5-base ships a 250,112-token multilingual vocabulary. The task is English-only, so
most of the embedding and output projection are dead weight that still costs memory and a
per-token matmul.

**Census.** Tokenizing the whole OpenASL corpus (98,419 sentences) gives **26,075 distinct tokens,
10.4 % of the vocabulary.** Occurrence coverage: top 5 K covers 94.6 %, 10 K → 98.1 %, 20 K → 99.7 %.

**Method.** Slice the rows of the shared embedding and `lm_head` to the keep set, sorted by original id
so that pad/eos/unk (0/1/2) stay at 0/1/2 and `decoder_start`/`eos` semantics are unchanged. Wrap the
tokenizer to remap ids in both directions; out-of-set tokens map to `<unk>`. **No retraining.**

**Result (976 clips, beam 4):**

| | params | ckpt MB | BLEU-1 | BLEU-4 | ROUGE-L | identical predictions |
|---|---|---|---|---|---|---|
| full | 587.7 M | 1187 | 49.61 | 23.16 | 43.17 | — |
| pruned | 243.6 M | 571 | 49.25 | **22.87** | 42.98 | 655 / 976 |

**Paired bootstrap: −0.28 BLEU-4, 95 % CI [−0.63, +0.05], P(pruned worse) = 0.95.**

**Verdict.** −59 % parameters and −52 % checkpoint for a cost that is small and may be zero. The
strongest single result in the project. The CI touching zero means we should say "at most about
0.6 BLEU-4", not "0.28".

**Honest detail.** Greedy decoding on a probe clip was *token-identical* before and after, yet 321 of
976 beam-search predictions change. Removing 224 K tokens from the softmax changes the normalization
and therefore beam *scores*, even where the argmax is unchanged. The aggregate barely moves; a third of
individual sentences do.

⚠️ **Flaw found in re-evaluation — see §3.3.** The keep set was built from train **+ dev + test**.

---

### 2.B Weight-only INT8, W8A32 (L9.1)

**Method.** Weights int8, scale stored fp16 and cast up before use, activations and accumulation
**fp32** — not fp16, despite the old "W8A16" label (renamed W8A32 on 2026-09-28). Per-row symmetric
int8 (absmax/127) with an fp16 scale, on every 2-D mT5 weight with
≥ 1e5 elements: 220 tensors, 278.3 M of 285.2 M stored values. Pose stack, layer norms and
relative-attention bias stay fp32. Max per-tensor relative error 4.2e-3.

**Result:**

| | file MB | BLEU-4 | ROUGE-L | paired Δ | sentences changed |
|---|---|---|---|---|---|
| pruned (baseline) | 571 | 22.87 | 42.98 | — | — |
| pruned + INT8 | **293** | 22.79 | 43.00 | **−0.09 [−0.34, +0.14]**, P = 0.77 | 182 / 976 (19 %) |
| vs released | 1187 → 293 | 23.16 → 22.79 | | −0.37 [−0.76, −0.02] | |

**Verdict.** Free within noise: the CI is centred near zero and P = 0.77 means we cannot even order the
two systems. Stacked with pruning: **4.05× smaller on disk for −0.37 BLEU-4.**

**What this row does *not* establish.** Two runtime modes exist: `dequant` (int8 on disk, float in RAM —
what the BLEU above used) and `int8` (int8 in RAM, dequantized per forward). The second is **10× slower
on CPU** (2139 ms vs 197 ms per sentence) because 217 layers dequantize on every token. So the file-size
win is real, the *memory* win costs speed, and the win that matters for energy — smaller and faster
together — requires a fused INT8 kernel that we have not built or measured. Quoting "4× smaller" as an
energy result would be wrong.

---

### 2.C Frame rate (L7.1, superseded → L7.2)

The pose stage's energy is linear in frames processed, making frame rate the largest energy lever in
the system. Two attempts were needed to measure its cost, and the first was wrong.

#### First attempt (L7.1) — length cap, now superseded

Capping clips at L frames and reading L as a frame rate gave: 205 → −0.17, 137 → −1.61, 103 → −3.26,
68 → −8.02 BLEU-4. **Two errors:**
1. A cap only shortens clips *longer* than L; shorter clips pass through untouched. A slower camera
   thins **every** clip. So the cap concentrates all the damage on long clips and hits them harder.
2. It assumed one source frame rate. **OpenASL has no single frame rate.**

#### Measuring the source rate

Frames ÷ duration (duration from the timestamps in each clip's filename), 400 test clips:

| source rate | share |
|---|---|
| 30 fps | 73 % (292/400) |
| 24 fps | 21 % (82/400) |
| 25 / 31 / 32 / 60 fps | 6 % |

Mean 29.04, median 30.00, range 23.98–61.22. **Any single assumed source rate mis-thins about a fifth
of the data.**

#### Corrected method (L7.2)

`fps_ratio_for_clip` ([common/pose_to_unisign.py](common/pose_to_unisign.py)) derives each clip's own
rate and keeps `round(duration × target_fps)` frames; the 256-frame cap then applies as before.
Verified: a 30.00 fps clip (T = 152 → 81) and a 24.11 fps clip (T = 131 → 87) both land on 16.0 fps.

**Result (pruned + INT8, 976 clips, beam 4, cap 64):**

| target fps | frames kept | BLEU-4 | ROUGE-L | paired Δ vs source rate |
|---|---|---|---|---|
| source (30/24) | 100 % | 22.79 | 43.00 | — |
| **24** | ~80 % | **22.80** | 43.22 | **+0.02 [−0.38, +0.45]**, P = 0.45 |
| **16** | ~53 % | **21.66** | 41.56 | **−1.12 [−1.71, −0.48]**, P = 1.00 |
| 12 | ~40 % | 20.26 | 39.15 | −2.52 [−3.22, −1.85], P = 1.00 |

**Verdict.**
- **24 fps is free.** CI centred on zero, and the point estimate is *positive*. About 20 % fewer frames
  on 30 fps clips, no retraining, no accuracy cost. The cheapest saving available in the project.
- **16 fps costs 1.12 BLEU-4**, not the 1.61 the cap method reported — the cap overstated the damage by
  38 %. This is the gap fine-tuning must close.
- **12 fps costs 2.5** and needs adaptation to be viable at all.

**Limitation (important).** This emulates a slower camera by *thinning already-extracted 30 fps poses*.
A real 16 fps capture differs: longer exposure, more motion blur per frame, and possibly worse keypoint
accuracy. We are measuring "the LM sees fewer frames", not "the camera ran slower". The pose owner's
board path drops frames before extraction, which produces the same LM input but not the same imagery.
Unmeasured.

---

### 2.D Decoding strategy (L6.1)

**Result (pruned model, 976 clips):**

| decoding | BLEU-4 | ROUGE-L | paired Δ vs beam 4 | eval wall | speed-up |
|---|---|---|---|---|---|
| beam 4 | 22.87 | 42.98 | — | 919 s | 1.0× |
| beam 2 | 22.06 | 42.22 | −0.81 [−1.29, −0.39] | 529 s | 1.7× |
| greedy | 20.88 | 41.56 | **−2.00 [−2.63, −1.41]** | 342 s | 2.7× |

**Verdict, and a corrected planning assumption.** The project plan assumed greedy decoding would cost
"well under a point". **It costs two.** Beam width is a real accuracy–energy axis, not a free saving.

Supporting timing (Mac CPU, single clip, median of 3) shows why — the decoder is 60–80 % of LM time and
scales with beam width, while the GCN and encoder run once per sentence:

| model | beams | GCN | encoder | decoder | total | decoder share |
|---|---|---|---|---|---|---|
| full | 1 | 92 | 86 | 268 | 446 ms | 60 % |
| full | 4 | 103 | 108 | 1000 | 1250 ms | 80 % |
| pruned | 1 | 102 | 105 | 191 | 396 ms | 48 % |
| pruned | 4 | 125 | 139 | 772 | 1035 ms | 75 % |

**Lever order on the LM side: beam width > vocabulary pruning > anything in the encoder.** Pruning cuts
decoder time 20–30 % (the shrunken output matmul); beam 4 → 2 saves ~25 % of total LM time for
−0.8 BLEU-4; greedy saves ~60 % for −2.0.

---

### 2.E Output length cap (L3.2)

| cap | BLEU-4 | paired Δ vs cap 100 | predictions at/over cap | sentences changed |
|---|---|---|---|---|
| 100 | 22.87 | — | 0 / 976 | — |
| **64** | 22.87 | **0.00, byte-identical output** | 0 / 976 | 0 |
| 48 | 22.73 | −0.14 [−0.30, −0.03], P = 0.994 | 39 / 976 | 28 |

Prediction lengths: mean 21.2 tokens, median 19, p95 44, **max 62**. So cap 64 is provably free and is
the deployment default. Cap 48 truncates 4 % of sentences for a small but statistically real loss.

**Verdict.** The cap bounds *tail* latency only — the average sentence is 21 tokens and never touches
it. It is a safety guard, not an energy lever. Beam width remains the lever that moves average cost.

---

### 2.F Pose extractor identification (C2) — an ablation over the *input*

Not an LM ablation, but it determines the pose baseline. Candidates were compared against the authors'
own keypoints for one clip, in the model's normalized space:

| candidate | body | left hand | right hand | face | resulting sentence quality |
|---|---|---|---|---|---|
| **RTMW-l-m 256×192 (lightweight)** | **0.091** | **0.053** | **0.041** | **0.012** | closer, more confident (mean logprob −0.81) |
| RTMW-x-l 384×288 (performance) | 0.141 | 0.139 | 0.056 | 0.020 | worse (−1.20) |

**Result: the smaller model is both closer to the reference and produces better translations.** Raw
pixel gaps (~57 px) were framing, not error — a per-axis scale/offset maps our crop onto the authors',
and Uni-Sign's normalization removes it before the model. **Decision: the 256×192 model is the
baseline**, which is also the cheaper one.

**Caveat carried forward:** an MMPose RTMPose-m/l wholebody at 256×192 would also fit the folder name.
Distinguishing them needs the board-side variant sweep, not more clips.

---

### 2.G Deployment-path equivalence (L8.1) — a correctness ablation

The pruned mT5 exported to three ONNX graphs (encoder, decoder-init, decoder-step with 48 past K/V
tensors; 340 + 614 + 557 MB, fp32, opset 17). Verified against PyTorch on real poses: **identical output
tokens, max |log-prob difference| 2.9e-5.** Synthetic check at T = 264 over 12 steps: logits within
2.2e-5. This establishes that the deployment runtime is not a new source of accuracy change — any BLEU
difference on the board will come from precision or the pose stage, not the export.

---

### 2.H Training-side ablations (C8) — **harness only, no scientific result yet**

Two runs, both on a 300-clip Mac debug slice, evaluated on 200 held-out test clips:

| run | before | epoch 0 | epoch 1 | reloaded |
|---|---|---|---|---|
| no input change (C8.1) | 14.71 | 14.08 | 14.49 | 14.55 |
| 16 fps (C8.2) | 13.79 | 13.78 | 13.55 | — |

**These establish exactly two things**, both about the code: the training loop, checkpoint save and
reload are correct (the saved checkpoint round-trips through the standard evaluation script to within
0.06 BLEU-4), and **±0.6 BLEU-4 is the noise floor at this slice size**, so neither run shows a signal.

**They establish nothing about adaptation.** 300 clips out of 96,476 cannot teach the model a new frame
rate. Any reading of "fine-tuning didn't help at 16 fps" from C8.2 would be wrong.

Trainable surface: mT5 frozen; `proj_linear`, `gcn_modules`, `fusion_gcn_modules`, `part_para`,
`pose_proj` trainable = **5.35 M of 243.6 M parameters (2.2 %)**. Recipe follows the authors' published
settings (AdamW eps 1e-9, wd 1e-4, cosine, clip 1.0, label smoothing 0.2, targets ≤ 50 tokens), with lr
lowered 3e-4 → 1e-4 because we adapt a converged checkpoint.

⚠️ **C8.2 is additionally confounded:** it used a uniform 0.667 ratio, which on 30 fps clips emulates
~20 fps and on 24 fps clips 16 fps. Its "16 fps" was a mixture. Superseded by the per-clip method.

---

## 3. Critical re-evaluation

This section is the answer to "re-evaluate your results". Each item is a weakness in the evidence as it
stands, not a hypothetical.

### 3.1 The design is a path, not a factorial — so the deltas cannot rule out interactions

Every delta was measured against the configuration immediately preceding it. The deltas therefore
**telescope by construction**: −0.28 (pruning) − 0.09 (INT8) − 1.12 (16 fps) sums to −1.50, and the
end-to-end difference 23.16 → 21.66 is also 1.50. That arithmetic is *guaranteed* by how the runs were
chained. **It is not evidence that the levers are independent**, and I should not have presented the
stack as if the costs simply add.

What is genuinely untested: whether the frame-rate penalty is larger on a full-vocabulary model, whether
INT8 degrades more when the encoder sees fewer frames, whether greedy decoding costs more at 16 fps than
at 30. With three levers we tested one path through eight or more cells.

**Action taken, now resolved.** The 16 fps run on the pruned-but-not-quantized model has completed
(`results/eval_test_pruned_truefps16.json`, all 976 test clips, 14 min). Comparing the frame-rate
penalty with and without INT8, on the same bootstrap resamples (difference-in-differences, 1000
resamples, N = 976):

| 16 fps penalty measured on | BLEU-4 30 → 16 fps | paired delta [95 % CI] |
|---|---|---|
| pruned, **no** INT8 | 22.87 → 21.54 | **−1.33** [−2.00, −0.64] |
| pruned, **with** INT8 | 22.79 → 21.66 | **−1.12** [−1.71, −0.48] |
| **interaction (difference of the two)** | | **+0.21** [−0.14, +0.57], P(<0) = 0.13 |

The interaction CI straddles zero and its whole range sits inside the 0.6 sensitivity band, so **there
is no detectable interaction between quantization and frame rate**: the 16 fps cost is the same whether
or not the model is quantized. Quoting the two deltas separately is therefore defensible for this pair.
Two caveats remain. First, this tests one of the several untested cells listed above — pruning × frame
rate and decoding × frame rate are still unmeasured. Second, absence of a detectable interaction at this
sample size is not proof of independence; the CI admits effects up to ±0.6, which is the same size as
the INT8 penalty itself. The honest statement is *"no interaction large enough for us to see"*, not
*"the levers are independent."*

Script: `unisign/did_ci.py` (paired resampling shared across all four runs, so the DiD is measured
on identical clip draws rather than by differencing two independent CIs).

### 3.2 The "0.6 BLEU noise floor" is a methodology-sensitivity band, not run-to-run variance

The three runs that define it differ in framework, numeric precision *and* frame-subsampling policy.
That makes 0.6 a useful conservative threshold, but it is measuring something other than what its name
suggests. In fact our standalone loop is **deterministic** (uniform subsampling, fixed beam), so
repeating it yields bit-identical output and zero variance. We therefore have:

- no measured variance for the evaluation path (it is exactly zero by construction), and
- **no measured variance for the training path at all** — seed-variance runs are still pending.

Consequence: every training-side number that follows will need its own noise band before any adaptation
result can be called real. The inference-side thresholds cannot be borrowed for it.

### 3.3 The pruned vocabulary was built using the test set — quantified

The keep set came from train + dev + **test** sentences, so tokens appearing only in test were retained.
In deployment you cannot know them. This mildly flatters the pruning result. Measured today:

| | tokens |
|---|---|
| keep set from train + dev only | 26,022 |
| keep set as actually built (incl. test) | 26,075 (**+53 test-only tokens**) |
| test token occurrences affected | 55 of 20,883 = **0.263 %** |
| test sentences containing ≥ 1 such token | **46 / 976 (4.7 %)** |

**Assessment: real but small.** A clean protocol would send those 55 occurrences to `<unk>`, plausibly
costing well under 0.1 BLEU-4 — inside the noise band, and much smaller than the −0.28 already
attributed to pruning. So the conclusion survives, but the report should either re-run with a
train+dev-only keep set or state this bound explicitly. **It should not be left unmentioned.**

### 3.4 Decode knobs were selected on the test set

Beam width, `max_new_tokens` and frame rate were all chosen by looking at test-set BLEU. There is no
held-out set behind those choices, so the final configuration is tuned on the data it is reported on.
With a corpus this size and effects this large the practical risk is low, but the protocol is wrong, and
a dev split exists — its poses were simply never extracted. Fixable.

### 3.5 Multiplicity is unaccounted for

Roughly fifteen paired comparisons have been run, each at 95 %. Expecting about one spurious
"significant" result is the correct prior, and none of the intervals is corrected. The two findings I
would flag as most vulnerable are the ones with intervals nearest zero: pruning's −0.28
[−0.63, +0.05] and the cap-48 result −0.14 [−0.30, −0.03]. The large effects (greedy −2.00, 16 fps
−1.12) are not at risk.

### 3.6 Training-slice evaluations use a harder subset, and are not comparable to the headline numbers

The C8 runs evaluate on the first 200 test clips, which score **14.71** against **22.87** on all 976 —
verified against the full evaluation, so the subset is genuinely harder, not a bug. Any comparison
across those two scales is invalid. Every C8 number must be read only against another C8 number.

### 3.7 Latency evidence is thin, and energy evidence barely exists

- All LM timings are **one clip** (299 frames), median of 3, on a Mac CPU. Decoder time depends on
  output length, which varies from 5 to 62 tokens, so a single clip cannot represent the distribution.
- Mac timings are ordinal only. They rank levers; they cannot be converted to joules.
- On the board we have **one configuration, one run, one clip, one signer**: FP32 pose extraction at
  56.4 ms/frame, 754 mJ/frame. The project's own results protocol demands 3 runs with mean ± std, ≥ 3
  signers, and a 30-minute sustained run. None of that is done.
- **No LM stage has ever been measured on the board.** Not latency, not power.

Consequence: **the accuracy–energy frontier plot — the deliverable the report is built around — cannot
be drawn yet.** Every BLEU number is ready; the joule axis is almost entirely missing. This is the
biggest gap in the project, and it is larger than any remaining accuracy question.

### 3.8 Minor confounds worth recording

- The released-model baseline used `max_new_tokens` 100 while the pruned rows used 64. Cap 64 was proven
  byte-identical to 100 *on the pruned model*; it was never verified on the released one (whose
  predictions could in principle run longer). The pruning delta rests on that untested assumption,
  though the measured 62-token maximum makes it very likely safe.
- BLEU-4 carries confidence intervals everywhere; **ROUGE-L never does.** ROUGE-L differences are quoted
  as bare numbers throughout and should not be interpreted as significant.
- INT8 BLEU was measured in `dequant` mode. The `int8`-in-RAM mode was verified token-identical on one
  clip only, not across the test set.

### 3.9 Ablations that do not exist yet

Missing from the LM side: adaptation at any frame rate on real data (the headline experiment), seed
variance, face-group dropping, pose-model variant swap, INT8 KV cache, activation quantization, prompt
ablation, LLM post-correction, context conditioning. Missing entirely: any interaction cell (§3.1), and
every board-side energy row (§3.7).

---

## 4. What the evidence supports today

| Claim | Support | Confidence |
|---|---|---|
| Vocabulary pruning: −59 % params, −52 % file, ≤ 0.6 BLEU-4 cost | 976 clips, paired CI | **Strong** (see §3.3 for a small caveat) |
| Weight-only INT8 on top: 4.05× smaller file, cost within noise | 976 clips, paired CI | **Strong** for file size; **not** for speed or energy |
| 24 fps is free | 976 clips, per-clip emulation, CI centred on zero | **Strong**, for LM input only |
| 16 fps costs 1.12 BLEU-4 un-adapted | 976 clips, CI [−1.71, −0.48] | **Strong**, same caveat |
| Greedy decoding costs 2 BLEU-4; beam width is a real axis | 976 clips, CI | **Strong** |
| `max_new_tokens` 64 is free | byte-identical output, max length 62 | **Proven** |
| ONNX deployment path is numerically equivalent | tokens identical, 2.9e-5 | **Proven** |
| The 256×192 pose model is the right baseline | normalized-space distance + sentence quality | **Reasonable**, one clip only |
| FP32 pose extraction cannot hit 30 fps on the board | 56.4 ms/frame measured | **Strong**, single run |
| Adaptation recovers accuracy at reduced frame rate | — | **No evidence either way** |
| Any configuration's energy per sentence | — | **Not measured** |

**Honest one-line summary of the project's state:** the accuracy side of the frame-rate/compression
trade-off is well characterized with proper intervals; the energy side, which is the actual subject of
the course, is one measured row on the board.

---

## 5. Queued checks, in priority order

1. ~~**16 fps without INT8** — tests the interaction in §3.1.~~ **Done**: no detectable interaction, +0.21 [−0.14, +0.57] (§3.1).
2. **Board energy rows for the LM** — the missing axis of the deliverable plot. Highest value of
   anything on this list.
3. **Seed variance** for the training path (§3.2) — required before any adaptation claim.
4. **The real adaptation run at 16 fps** on all 96,476 clips — the experiment this whole thread exists
   to set up.
5. **Re-prune from train+dev only** (§3.3) — bounds a known leak at a cost of one 15-minute run.
6. **Extract dev poses** so knobs stop being selected on test (§3.4).
7. **Repeat pose-stage rows** to the project's own protocol: 3 runs, ≥ 3 signers, sustained (§3.7).

---

## 6. Source files

Measurements: [results/RESULTS.md](results/RESULTS.md) (authoritative log, with per-run JSON alongside).
Per-ablation raw output: `results/eval_test_*.json` — `released`, `pruned`, `pruned_w8`,
`pruned_greedy`, `pruned_beam2`, `pruned_mnt{48,64}`, `w8_len{205,137,103,68}`, `w8_truefps{24,16,12}`.
Code: [unisign/eval_openasl.py](unisign/eval_openasl.py) (scoring path),
[unisign/model.py](unisign/model.py) (pruning), [unisign/quant.py](unisign/quant.py) (INT8),
[unisign/train_adapt.py](unisign/train_adapt.py) (fine-tuning),
[unisign/bootstrap_ci.py](unisign/bootstrap_ci.py) (intervals),
[common/pose_to_unisign.py](common/pose_to_unisign.py) (pose→input, frame-rate emulation).
