# Sign-language translation on a 15 W Jetson — work completed to date

**Course:** CS 6301 Low-Power Computer Vision · **Last updated:** 2026-09-26
**Scope of this document:** everything finished so far, what each result means, and what is still open.
Two people share the project; this covers both tracks but goes into detail on the language-model
(LM) track, which is Atisri's.

---

## 1. What the system does

A video of someone signing goes in, English text comes out, on a course-owned Jetson Orin Nano
running at 15 W.

```
video frames
   │
   ├─ Stage 1: pose extraction (RTMPose / RTMW, COCO-WholeBody, 133 keypoints per frame)
   │            → 2-D keypoints + confidences
   │
   ├─ Stage 2: pose encoder (Uni-Sign ST-GCN, 4 parts: body / left hand / right hand / face;
   │            the two hands share weights) → one embedding per frame
   │
   └─ Stage 3: mT5-base decoder, prompted "Translate sign language video to English: "
                → an English sentence
```

The model is Uni-Sign's released **pose-only** OpenASL checkpoint. Pose-only matters for power: the
heavy RGB video branch is never run, and the LM only ever sees keypoints.

**The central question of the project** is not "how accurate can this be" but *how much accuracy does
each watt buy*. Every result below is therefore a pair: a quality number (BLEU-4 / ROUGE-L) and a cost
(parameters, megabytes, milliseconds, or joules).

### Ground rules we work under

- **The Jetson is course-owned and we have no sudo, permanently.** So the power mode is fixed at 15 W,
  and `nvpmodel` / `jetson_clocks` are unavailable. A "power budget" here is a *measured average*, not
  a hardware cap. All work happens inside the provided `lpcv-work` Docker container, where pip works
  without root.
- **Every hardware number comes from a real board run.** Nothing is estimated or scaled from a laptop.
- **Quality numbers can come from anywhere; timing numbers cannot.** BLEU depends only on the model and
  its inputs, so the same sentences appear on a Mac, on Colab or on the board. Latency and watts depend
  on the hardware, so they only come from the Jetson. Mac timings in the logs are labelled
  "indicative only" and are used solely to rank levers against each other.
- **Training never happens on the board** (too slow, and installing torch there is off-limits). Training
  runs on Colab; the Mac holds only a 300-clip debug slice for proving code correct.

---

## 2. Reference points established

| Measurement | Result |
|---|---|
| Paper, OpenASL test, pose-only | 49.10 BLEU-1 / 22.67 BLEU-4 / 42.77 ROUGE-L |
| Recomputed from the authors' own released predictions | 22.64 BLEU-4 — metric code confirmed correct |
| **Our reproduction** (Colab T4, bf16, beam 4) | **22.53 BLEU-4** / 42.68 ROUGE-L, 976 test clips, 5 min |
| Our standalone evaluation loop (Mac CPU, fp32, deterministic subsampling) | 23.16 BLEU-4 |

The 0.6 BLEU-4 spread across those three runs comes only from frame-subsampling policy and numeric
precision. **That spread is the project's noise floor:** any claimed effect smaller than about
0.6 BLEU-4 needs a confidence interval before it can be believed. We built
[unisign/bootstrap_ci.py](unisign/bootstrap_ci.py) (paired bootstrap, 1000 resamples) for exactly this,
and every accuracy claim below carries one.

What 22 BLEU actually looks like, since the number alone is misleading — reference: *"I am Ed Bosson,
and I am 67 years old and now retired."* Prediction: *"Hello, I'm Howard Rosenblum, and I'm a
fourteen-year-old immigrant."* Fluent, grammatical, wrong in every specific. That is the field's state
of the art on this task.

---

## 3. Results: the pose stage (Stage 1)

### 3.1 Correctness ladder, then speed

We refused to report a latency number before proving the fast path computes the same keypoints as the
reference implementation:

| Check | Result |
|---|---|
| Our preprocessing vs mmpose's pipeline | identical (max diff 0.0) |
| ONNX Runtime vs PyTorch | max diff 8.0e-6 |
| TensorRT "FP32" with **TensorRT's defaults** | max diff 1.55e-3 — **FAIL** |
| TensorRT FP32 with TF32 explicitly disabled | max diff 7.2e-6, 2416/2416 keypoints agree — PASS |

**Methodology finding worth reporting on its own:** TensorRT silently enables TF32 on Ampere, so an
engine you asked for in FP32 is not FP32. Our baseline would have been quietly wrong, and every
quantization comparison measured against it would have been skewed. `common/trt_runner.py` now clears
the flag.

### 3.2 Baseline cost on the board (15 W, measured)

RTMPose-x (384×288), one 299-frame clip, batch 1:

| pre | TensorRT | post | **total** | avg W | idle W | **mJ/frame** |
|---|---|---|---|---|---|---|
| 10.6 ms | 44.4 ms | 1.5 ms | **56.4 ms** | 11.62 | 3.73 | **754** |

Reading: 56 ms/frame is about 18 fps, but the source video is 30 fps, so **the FP32 pose stage cannot
keep up in real time.** That single fact motivates every optimization that follows. One 10-second
sentence costs 225 J in pose extraction alone. Preprocessing (JPEG decode + affine, on CPU) is 19 % of
frame time and does *not* shrink when you quantize the GPU model — a ceiling that pure GPU work can't
break through.

### 3.3 Which pose model produced the released poses

The released poses live in a folder named `pose-rtmpose-192`, which is a hint, not an answer. We fetched
the authors' own keypoints for one clip (18 MB via HTTP range requests, instead of downloading the
32 GB archive — see [data/openasl_pose_fetch.py](data/openasl_pose_fetch.py)) and compared candidates
in the model's own normalized space:

| Candidate | distance: body / left / right / face |
|---|---|
| **RTMW-l-m, 256×192 (rtmlib "lightweight")** | **0.091 / 0.053 / 0.041 / 0.012** ← closest on every part |
| RTMW-x-l, 384×288 ("performance") | 0.141 / 0.139 / 0.056 / 0.020 |

Raw pixel gaps of ~57 px turned out to be *framing*, not extractor error: a per-axis scale and offset
maps our tight crop onto the authors' padded square crop, and Uni-Sign's own normalization removes it
before the model ever sees it. **Decision: the smaller 256×192 model is the baseline**, which is also
the cheaper one — a rare case where matching the reference and saving energy agree.

A pleasing cross-check: the *lightweight* poses produce a more confident and more accurate sentence
than the heavier 384×288 ones, consistent with the archive name. Bigger pose model, worse translation.

---

## 4. Results: the language model (Stage 3) — the main body of work

Levers were evaluated in order of expected payoff. All numbers: 976 test clips, paired bootstrap CIs.

### 4.1 Vocabulary pruning — the single biggest win

mT5-base carries a 250 K-token multilingual vocabulary. This task is English-only, and the entire
OpenASL corpus (98,419 sentences) uses **26,075 distinct tokens — 10.4 %** of it.

So we sliced the embedding and output-projection rows down to the tokens that actually occur, and
wrapped the tokenizer to remap ids both ways. No retraining.

| | full | pruned |
|---|---|---|
| parameters | 587.7 M | **243.6 M (−59 %)** |
| checkpoint on disk | 1,187 MB | **571 MB (−52 %)** |
| BLEU-4 | 23.16 | 22.87 |

Cost: **−0.28 BLEU-4, CI [−0.63, +0.05]** — small, possibly zero. For more than half the model gone.
The decoder also got 20–30 % faster, because the per-token matmul against the output layer shrank by
a factor of ten.

One honest subtlety: 321 of 976 sentences change wording even though greedy decoding on a test clip was
bit-identical. Dropping 224 K tokens from the softmax shifts *beam* scores even where the top choice is
unchanged. The aggregate is nearly unchanged; individual sentences are not.

### 4.2 Weight-only INT8 (W8A32), stacked on top

Per-row symmetric int8 on every large 2-D mT5 weight (220 tensors); the pose stack, layer norms and
attention biases stay in float.

| Checkpoint | file MB | BLEU-4 | Δ vs previous |
|---|---|---|---|
| released | 1,187 | 23.16 | — |
| pruned | 571 | 22.87 | −0.28 [−0.63, +0.05] |
| **pruned + INT8** | **293** | 22.79 | −0.09 [−0.34, +0.14], within noise |

**Total: 4.05× smaller on disk for −0.37 BLEU-4.** INT8 on its own is free within our noise floor.

Caveat we're careful about: 19 % of sentences change wording, so the quantization is invisible only in
aggregate. And keeping weights as int8 *in memory* is currently 10× slower on CPU, because 217 layers
dequantize on every token — the memory saving and the speed saving need a fused kernel to arrive
together. That's a board-side task.

### 4.3 Decoding: not a free lever

| Decoding | BLEU-4 | Δ vs beam 4 | wall time |
|---|---|---|---|
| beam 4 | 22.87 | — | 919 s |
| beam 2 | 22.06 | −0.81 [−1.29, −0.39] | 529 s (1.7× faster) |
| greedy | 20.88 | **−2.00 [−2.63, −1.41]** | 342 s (2.7× faster) |

Our project plan had assumed greedy decoding would cost "well under a point". **It costs two points.**
Beam width is a genuine accuracy–energy axis, not a free saving.

Timing breakdown explains why: the decoder is 60–80 % of LM time and scales with beam width, while the
GCN and encoder run once. **Lever order on the LM side: beam width > vocabulary pruning > anything in
the encoder.**

A `max_new_tokens` cap of 64 is free (the longest prediction is 62 tokens, and output is byte-identical
to no cap). Cutting to 48 truncates 4 % of sentences for −0.14 BLEU-4. The cap bounds worst-case
latency; it does nothing for the average.

### 4.4 Frame rate: the largest energy lever, and where a measurement error was caught

Pose energy is linear in frames processed, so running the camera slower is the biggest saving
available anywhere in the pipeline. The question is what it costs in quality.

Our first pass approximated this by capping clip length, and concluded 16 fps costs 1.6 BLEU-4. **That
was wrong twice over,** and finding out why produced one of the more useful results here:

1. A length cap only shortens clips *longer* than the cap. A real slower camera thins **every** clip.
2. **OpenASL is not one frame rate.** Measured across 400 test clips (frames ÷ duration, from the
   timestamps in each filename): **73 % are 30 fps, 21 % are 24 fps**, the rest 25, 31 or 60. Any single
   assumed source rate mis-thins about a fifth of the data.

The fix derives each clip's own rate and keeps `round(duration × target_fps)` frames
(`fps_ratio_for_clip` in [common/pose_to_unisign.py](common/pose_to_unisign.py)). Verified: a 30.00 fps
clip and a 24.11 fps clip both land on 16.0 fps. Re-measured honestly:

| target fps | BLEU-4 | Δ vs source rate (95 % CI) |
|---|---|---|
| source (30/24) | 22.79 | — |
| **24** | **22.80** | **+0.02 [−0.38, +0.45] → free** |
| **16** | **21.66** | **−1.12 [−1.71, −0.48]** |
| 12 | 20.26 | −2.52 [−3.22, −1.85] |

**Two conclusions we can act on:**
- **24 fps is free.** The interval is centred on zero. On 30 fps clips that's 20 % fewer frames for no
  measurable accuracy cost, with no retraining. Claimable today, pending the board's energy figure.
- **16 fps costs 1.12 BLEU-4** — about 47 % fewer frames. This is a precise target for fine-tuning to
  close, and it is the reason the adaptation run exists.

### 4.5 Deployment path: ONNX with a KV cache

The pruned mT5 exports to three ONNX graphs (encoder, decoder-init, decoder-step with 48 past K/V
tensors). Verified against PyTorch on real poses: **identical output tokens, max log-probability
difference 2.9e-5.** Budgeted three days, took one attempt. TensorRT engines on the Jetson are the
remaining step.

### 4.6 Fine-tuning harness

[unisign/train_adapt.py](unisign/train_adapt.py) follows the authors' published recipe (AdamW, cosine
schedule, gradient clipping 1.0, label smoothing 0.2, targets truncated to 50 tokens), with two
deliberate departures: a **lower learning rate** (1e-4 rather than 3e-4), because we adapt an
already-converged checkpoint rather than training from scratch; and **mT5 frozen**, so only the pose
side trains — **5.35 M of 243.6 M parameters**, which keeps checkpoints at 64 MB and fits Colab.

Verified on a 300-clip slice with *no* input change, as a test of the code and not of the science:
BLEU went 14.71 → 14.08 → 14.49, and the saved checkpoint reloads through the standard evaluation
script to within 0.06 BLEU. Training, evaluation, saving and reloading are all correct, and **±0.6 BLEU
is the noise floor at that slice size.**

A second dry run at 16 fps produced no gain, as expected from 300 clips. Its purpose was to exercise
the frame-rate path end to end before spending Colab time.

**One trap documented so nobody re-reports it as a finding:** the first 200 test clips score 14.71,
while all 976 score 22.87. The subset is simply harder — verified against the full evaluation, not a
bug. Only compare numbers computed on the same clips.

---

## 5. Cross-cutting lessons

1. **Prove correctness before measuring speed.** The TF32 default would have invalidated every later
   comparison.
2. **Publish a noise floor first, then respect it.** Three legitimate runs of the same model span
   0.6 BLEU-4. Without that band, INT8 quantization (−0.09) would look like a real regression, and it
   isn't.
3. **Check the assumption you built the axis on.** "16 fps" was reported before anyone confirmed the
   source frame rate — which turned out not to be a single number. The corrected cost is 1.12, not 1.6.
4. **Separate what's hardware-dependent from what isn't.** Quality is portable; time and power are not.
   Keeping this explicit lets most work happen off the board, on a device the two of us share.
5. **Assumptions in the plan don't survive contact.** Greedy decoding was assumed nearly free; it costs
   two BLEU points.

---

## 6. Status

**Done:** reproduction of the published result · metric verification · pose-extractor identification ·
FP32 board baseline with power · vocabulary pruning · INT8 quantization · decode-knob sweep ·
`max_new_tokens` sweep · true frame-rate costs with CIs · ONNX export with KV cache · bootstrap CI
tooling · standalone inference and evaluation scripts · fine-tuning harness, smoke-tested.

**Next, and blocked on Colab:** fetch the full training set (about 97 K clips, 30 GB) into storage,
then run seed-variance training (to establish the training noise band) and the **16 fps adaptation
run**, which either closes that 1.12 BLEU gap or doesn't. That single result decides whether 16 fps or
24 fps is the deployment frame rate.

**Next, on the board (pose owner):** copy back the FP32 keypoints, the first on-device end-to-end
translation, TensorRT engines for the LM, and power logging for each configuration.

**Open questions:**
- Does fine-tuning recover the 1.12 BLEU-4 that 16 fps costs? Unknown until the Colab run.
- Do the Mac's accuracy numbers hold on the board, where the model may run at lower precision? Each
  final configuration needs one confirming board run.
- Can INT8 weights save memory *and* time together? Needs a fused kernel, not a Python dequantization.

**The final deliverable** is an accuracy–energy frontier: measured joules per sentence against BLEU-4
for every configuration, with the whole report built around that plot.

---

## 7. Where things live

| Path | Contents |
|---|---|
| [results/RESULTS.md](results/RESULTS.md) | Every measurement, with method and caveats. The authoritative log. |
| [PROJECT-GUIDE.md](PROJECT-GUIDE.md) | Step-by-step plan with status markers |
| [WORKSPLIT.md](WORKSPLIT.md) | Task ownership across the two tracks, and the board queue |
| [common/pose_to_unisign.py](common/pose_to_unisign.py) | Keypoints → model input; frame-rate emulation |
| [unisign/eval_openasl.py](unisign/eval_openasl.py) | Standalone evaluation loop (the scoring path for all ablations) |
| [unisign/train_adapt.py](unisign/train_adapt.py) | Fine-tuning harness |
| [unisign/model.py](unisign/model.py) · [unisign/quant.py](unisign/quant.py) | Vocabulary pruning · INT8 quantization |
| [unisign/bootstrap_ci.py](unisign/bootstrap_ci.py) | Paired bootstrap confidence intervals |
| [data/openasl_pose_fetch.py](data/openasl_pose_fetch.py) | Fetches single pose files from the 32 GB archive over HTTP ranges |

Large files (weights, pose data, training runs) are deliberately not in git; they live on Drive and the
board, mirroring these paths.
