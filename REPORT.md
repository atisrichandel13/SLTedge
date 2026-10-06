# Sign-language translation on a 15 W Jetson: an accuracy–energy frontier

**CS 6301 Low-Power Computer Vision · C11 report · drafted 2026-10-05**

Atisri Chandel (language-model track) · Tushar Goyal (pose track, all board runs)

> **Status.** Every number here is measured and traceable to a file in `results/`. Two items are
> outstanding and are marked where they appear: **J9 step 2** (the adaptation training, the one
> remaining accuracy lever — step 1, the 918-clip dev pose extraction, is delivered and verified,
> §2.5l) and the **demo video**. Nothing in this draft depends on either.
>
> **Updated 2026-10-06.** The accuracy numbers here are at **n=931**, not the n=400 subset an earlier
> draft used, and the pose deficit is **−1.41 [−2.19, −0.63]**, not −2.24 (§2.5k, §L19). J10 has also
> run (§2.9E) with a negative result, so the frontier's beam-width caveat is **narrowed, not closed**.

---

## 1. What we built, and the one fact that shapes it

American Sign Language video → English text, on an NVIDIA Jetson Orin Nano 8 GB held at **15 W**
(`nvpmodel` mode 0, which caps the GPU at 612 MHz on our board).

```
video frames → RTMW-l-m pose (133 keypoints) → Uni-Sign ST-GCN → mT5-base → English
```

The pose stage never passes pixels onward; everything downstream sees only coordinates, and only
**69 of the 133** keypoints reach the ST-GCN.

The deliverable is not "accurate" or "fast" but an **accuracy–energy frontier**: for each sensible
configuration, how good the translation is and what it costs in joules. The point of a frontier is
that it separates cheapening that costs accuracy from cheapening that is free.

**The structural fact behind every decision:** the two models have opposite cost shapes.

| | pose stage | language model |
|---|---|---|
| cost scales with | **frames** | **sentences** |
| measured on-device | 13.8 ms, 127 mJ / frame | 1.9 s, 11.9 J / sentence |
| share of system energy | **~75 %** | ~25 % |

Because the pose stage dominates and charges per frame, **frame rate is the efficient lever and
decoder width is not**: per BLEU-4 point sacrificed, reducing 30→24 fps costs 103.5 J against 1.0 J
for beam 4→2 (§2.9). Calling beam width "the frontier" was true of the language model alone and false
of the system.

---

## 2. Baseline clarity

Nothing is claimed as an improvement until the starting point reproduces.

| check | BLEU-4 | ROUGE-L |
|---|---:|---:|
| Uni-Sign paper, OpenASL pose-only | 22.67 | 42.77 |
| authors' released predictions, our scorer | 22.64 | 42.83 |
| authors' checkpoint, our Colab run | 22.53 | 42.68 |
| authors' checkpoint, our own eval loop | 23.16 | 43.17 |

Those four bracket the noise floor, so **22.5–22.7 BLEU-4 is the FP32 reference band** and the
released checkpoint is a reference rung nothing overwrites.

One trap worth recording: the authors' released predictions carry a `sample: …, prediction:` prefix.
Scored raw they give **58.7 BLEU-4** — a 36-point error from a text prefix, and the kind of thing that
would have invalidated the whole project silently.

We also identified which extractor produced the released keypoints (§C2) rather than assuming: a
**256×192 RTMW/RTMPose** model. That fixed our baseline pose model and made RTMPose-x 384×288 the
oversized comparison rather than the default.

---

## 3. Compression decisions, each with an accuracy axis

The rubric asks for an accuracy number against every efficiency row. Each decision below has one.

### 3.1 Pose model: RTMW-l-m 256×192

Chosen to match the extractor §C2 identified, not for speed — and it turned out to be the accurate
choice too: at n=30 on the board, **RTMW beats RTMPose-x by 7.13 BLEU-4 [−10.94, −3.07]**. The cheap
model was also the better one, which we were not entitled to assume.

### 3.2 Pose precision: FP16

| engine | TRT ms | total ms | mJ / frame |
|---|---:|---:|---:|
| RTMW-l-m FP32 | 25.6 | 33.0 | 275 |
| **RTMW-l-m FP16** | **13.8** | **19.7** | **127** |
| RTMPose-x FP32 | 66.7 | — | 733 |

Accuracy cost **−0.13 BLEU-4 [−1.48, +1.21]**, and 99.81 % of the consumed keypoints land within 5 px
of FP32. FP16 later proved to be a *fitting* requirement as well: the same end-to-end run with the
FP32 engine runs out of memory during LM decode.

> **Honest limit.** That −0.13 is n=30 and its interval is ±1.5. §8 lists re-testing it at n≈400 as
> the cheapest open accuracy question, precisely because an n=30 null is what hid the pose gap in §5.

### 3.3 Vocabulary pruning

OpenASL uses only 26,078 of mT5's 250,112 tokens, so the embedding and output head are sliced:
**587.7 M → 243.6 M parameters, 1187 MB → 571 MB.**

| axis | result |
|---|---|
| accuracy, n=976, authors' keypoints | **−0.28 BLEU-4 [−0.63, +0.05]** — not established |
| accuracy, n=400, our keypoints | −0.51 [−1.08, +0.05] — not established |
| **system energy, measured end to end** | **−12.1 %** (42.94 → 37.75 J/sentence) |
| peak GPU | 2.56 → **1.07 GB** |
| model load | 58 → **29 s** |

The energy result was new: pruning had only ever been argued on memory and accuracy grounds. And the
saving is **3–4× larger at beam 4 than at greedy**, which is the mechanism confirming itself — the
layer pruning shrinks is the output projection, evaluated once per beam per token.

A methodology catch inside it: the first token census included the **test** split, so the vocabulary
was selected using test data. Re-pruned on train+dev, the leak was worth **−0.03 BLEU-4
[−0.11, +0.01]** — a precise null. It changed no result and is recorded as an error anyway.

### 3.4 Frame rate — the main lever

Pose energy per second of video: **4.25 / 3.36 / 2.27 / 1.67 / 1.17 J/s** at 30 / 24 / 16 / 12 / 8 fps
(1.00 / 0.79 / 0.53 / 0.39 / 0.27×). Energy *per frame* stays flat — the saving is from processing
fewer frames, not cheaper ones.

Accuracy, un-adapted, per-clip true-rate emulation at n=976:

| rate | BLEU-4 vs source | verdict |
|---|---|---|
| **24 fps** | **−0.07 [−0.53, +0.39]** | **free** |
| 16 fps | −1.33 [−2.00, −0.64] | established loss |
| 12 fps | −2.5 | avoid |
| 8 fps | −8.0 | floor |

Confirmed independently on **our own** board keypoints at n=400: **+0.13 [−0.49, +0.74]**, a tight
null on a different pose source.

Shortening the sequence shortens the LM *encoder* (39–50 ms) while the *decoder* (1062–1538 ms)
dominates, so cutting 256→68 frames saves the LM only 12 %. **Frame rate is a pose-stage saving the
language model barely notices.**

### 3.5 Rejected, with reasons

- **mT5 in FP16: dead, not degraded.** Every step returns token 0 at logprob −10.169 = −ln(26078), a
  uniform distribution.
- **Weight-only INT8 works but was dropped.** 293 MB (4.05× smaller), −0.09 BLEU-4 [−0.34, +0.14].
  Dropped because the decoder is host-bound, so INT8 buys memory not latency, and the int8 runtime is
  **10× slower per step** (2139 vs 197 ms) as 217 layers dequantize per token. Our first stated
  reason for dropping it was wrong about the implementation (it is W8A32, not W8A16) and is corrected
  in the record.
- **TensorRT for the decoder: a clean negative.** FP32 engines give identical tokens at 19 ms/token
  against 55 for PyTorch, but the host-side step loop bounds it, so it is not faster end to end. FP16
  engines overflow to uniform logits; BF16 diverges at step 3.

---

## 4. Deployment validation at 15 W

### 4.1 The recommended operating point, measured end to end

**Pruned checkpoint, RTMW-l-m FP16, 24 fps, beam 4, cap 64** — one process, both models resident,
sharing the 15 W budget. One test clip, 8.51 s of video, 3 timed sentences:

| | value |
|---|---|
| sentence latency | **6999.1 ms** |
| ├ pose (204 frames) | 5038.8 ms |
| └ language model | 1939.5 ms |
| **energy** | **37.75 J / sentence** |
| per stage (own power windows) | pose 25.26 J at 5.09 W, LM 11.86 J at 6.23 W |
| peak GPU | 1.073 GB |
| model load (outside the window) | 28.9 s |

**It runs at ~0.82× real time** for this clip. The pose stage alone is comfortably live; because LM
cost is per *sentence* and pose is per *frame*, long sentences get better and short ones worse.

Per-stage energy is integrated from each stage's own samples, not apportioned from the total by
latency share — the stages do not draw equal power (5.09 W vs 6.23 W), so an apportionment would have
mis-assigned about a quarter of the LM's energy.

### 4.2 It does not throttle

Three 30-minute sustained runs:

| load | drift (1st→5th fifth) | Tj max | verdict |
|---|---:|---:|---|
| pose FP16 only | — | 53.33 °C | no throttling |
| pose FP32 only | −0.26 % | 57.44 °C | no throttling |
| **end-to-end, deployable config** | **−0.56 %** | **51.75 °C** | **no throttling** |

Over **254 consecutive sentences** the deployable configuration drifted **−0.56 %** — marginally
*faster* at the end — and works out to **38.48 J/sentence sustained against the 37.75 J measured in a
short window, +1.9 %.**

**This is the result the rest of the table needed.** Every latency and energy row in the project came
from an 8–25 s window with Tj never above 53 °C, and nothing had tested continuous operation. They are
not optimistic, and the frontier's energy axis does not move under realistic duty cycle.

### 4.3 Memory is the binding constraint, and it is unified

CPU and GPU share one 8 GB pool, so a GPU allocation failure surfaces as
`NVML_SUCCESS == r INTERNAL ASSERT FAILED` — which must be read as *out of memory*.

- The **full** checkpoint needs **>5.2 GB** of free memory to load, and that threshold is for the
  checkpoint *plus a resident TensorRT engine*; the eval path, which holds no engine, loads it at
  4.1 GB.
- The **pruned** checkpoint needs ~1.5 GB, and it runs a cell the full checkpoint cannot: 255 frames
  at beam 4 failed three times on the full checkpoint at up to 5768 MB free, and succeeded on the
  pruned one at **4238 MB**.
- Free-page count alone is **not** a sufficient readiness check, and an explicit page-cache reclaim
  sized to the *target* is mandatory before any load.

---

## 5. The accuracy of the system we actually ship

This is the part most easily overstated, so it is separated out.

Every accuracy figure in §2–3 is scored on the **authors' released keypoints**. The deployed system
feeds keypoints from **our own** extractor. Measured on 400 test clips from 400 distinct videos, our
own board extraction, pruned checkpoint, 24 fps, beam 4, batch 1:

| n=931, **all paired test clips** | BLEU-4 | ROUGE-L |
|---|---:|---:|
| **the deployed system** | **21.73** | **41.82** |
| same clips, authors' keypoints | 23.14 | 43.15 |
| **pose-substitution cost** | **−1.41 [−2.19, −0.63]** | **−1.33 [−2.23, −0.49]** |

**Both are established, and this is the project's one real accuracy deficit.**

> **UPDATED 2026-10-05 from the n=400 subset**, which gave −2.24 [−3.50, −1.09] on BLEU-4 and did not
> establish ROUGE-L at all. J12 extended our board pose set to all **931** test clips that exist on
> disk (45 of the 976 `labels.test` names have no clip: 43 dead links and 2 excluded by a stale
> `--exclude-yid` default, §L20 — the true ceiling is 933 and the difference is ~0.001), and the
> full-width answer is
> **−1.41**. Nothing earlier was wrong — −1.41 sits inside the n=400 interval — but that estimate sat
> near its pessimistic end. Two things changed with it, and one claim of ours died:
>
> - **The "monotonic drift" argument is withdrawn.** We wrote that the cost moved monotonically away
>   from zero as the sample grew — −0.35 at n=30, −1.89 at n=100, −2.24 at n=400 — and read that as
>   the signature of a real effect. The next rung is **−1.41**, so the drift reversed. The effect *is*
>   real and *is* established at full width; the monotonicity was sampling noise that happened to
>   point one way three times. We keep the finding and drop the reasoning.
> - **ROUGE-L now establishes it too**, −1.33 [−2.23, −0.49], and the two metrics converge: **1.41
>   against 1.33**, where at n=400 they differed by about 2×.
>
> We had also called this cost "nothing detectable" at n=30, which was reading a near-zero point
> estimate inside a ±4 interval as evidence of absence. `results/ci_n931_posesub_fps24.json`, §2.5k.

**Where it is not.** We spent the diagnostic effort before the compute:

- Our keypoints sit a median of **0.47 % of the frame** from the authors', with a mean signed offset of
  0.0036 — essentially zero. The coordinate frame is right and the information is present.
- Swapping **only the hands** costs 1.60 BLEU-4, but *repairing* our hands recovers only
  **+0.35 [−0.69, +1.61]** — not established, at most a fifth of the gap. Degrading hands alone
  (−1.60) and body+face alone (−1.40) nearly equal degrading both (−1.75): strongly **sub-additive**,
  so no keypoint group carries it.
- The authors' poses came from the **same** 256×192 RTMW/RTMPose family we use, so there is no
  better-architecture upgrade available.

**So the gap is a training-distribution shift, not an extraction failure** — the model was fitted on
the authors' keypoints and never saw our extractor's noise. That makes **adaptation** the lever and a
pose-model swap a poor use of board time.

> **Outstanding (J9).** Fine-tuning the pose encoder on dev-split poses from our own extractor, then
> evaluating on the test clips, is specified in `ASK-ADAPT-TO-OUR-POSES-2026-10-05.md` and has not
> been run. **The target is 21.73 with a ceiling of 23.14, on all 931 paired clips** (updated
> 2026-10-05 from 21.57/23.81 at n=400). Supporting evidence that it should work: the harness already
> *improves* the model (+0.18 BLEU-4 / +0.70 ROUGE-L) with **no** distribution shift to adapt to.
> **Counter-evidence that it may not be resolvable:** §L17 sizes the expected recovery at ~39 % of the
> gap, which is +0.55 against the measured ±0.778 half-width at n=931, so an effect of the predicted
> size would come back *not established* even at full width. Open as Q9 to the LM track.

---

## 6. The frontier

`results/frontier.png`, `results/frontier.csv`. Nine cells over {source, 24, 16 fps} × {beam 4, 2,
greedy}. Accuracy measured at n=976; system energy **composed** as
`pose_J_per_s × clip_seconds + LM_J` from board measurements of each stage.

| cell | system J | BLEU-4 | vs reference |
|---|---:|---:|---|
| beam 4 @ source | 43.4 | 22.87 | reference |
| **beam 4 @ 24 fps** | **36.1** | **22.80** | **−17 % energy for −0.07, CI [−0.53, +0.39]** |
| beam 4 @ 16 fps | 27.0 | 21.54 | −41 % energy for −1.33 [−2.00, −0.64] |

**The recommendation is beam 4 at 24 fps**: free within noise on both metrics, for 17 % less energy.
16 fps is an established loss and is not recommended.

**What the absolute energy column is and is not.** The *relative ordering* is the deliverable and is
robust — every cell shares one clip set, one pose source and one set of resample draws. The absolute
joules carry four known caveats, and the honest position is that they should not be quoted to better
than a few percent:

| caveat | size | status |
|---|---|---|
| composition vs measured end-to-end | ~6 % low | run-to-run variation is ~3 % at 1σ; ~2σ, consistent but not demonstrated |
| decoder-width term | **understated 2.1–2.3×** | **open** — the checkpoint mismatch explained only about half of it |
| `LM_J` measured on the pruned model, standalone | **no effect** | **closed 2026-10-05** by J10/§2.9E — residency changes nothing, so this is *not* the cause |
| measured clip at the 83.4th percentile of crop area | **−2.6 %** for a median clip | **closed** — was estimated at ~7 %, measured 3× smaller |

**Net, the absolute column is mildly optimistic, not approximately unbiased.** The composition runs
~6 % low and the crop-area term is −2.6 %, so they do not offset — an earlier pairing of these two
assumed the crop term was ~7 %, which J6 measured 3× smaller. The net is roughly **3–4 % low.**

The one we would fix first is the decoder-width term: across two checkpoints and four frame counts the
measured beam-4 penalty is **flat in absolute joules** (6.95–7.02 J full, 4.50–4.65 J pruned) while the
composition makes it grow with encoder length. Decoder-width energy is a per-sentence constant and
should not be modelled as a function of sequence length.

> **J10 ran on 2026-10-05 and the leading explanation is now excluded (§2.9E).** The hypothesis was
> that §2.9C measured the LM *standalone* — one process, no pose engine resident — while every
> end-to-end run holds both models, and that the missing resident footprint compressed the spread.
> Re-running the identical sweep with the RTMW FP16 engine resident, changing nothing else, moves the
> decoder-width term from 2.16 / 1.94 / 1.34 / 2.49 / 1.15 J to **2.15 / 1.80 / 1.21 / 2.40 / 1.13**
> — a ratio of **0.90–1.00×**. The spread *narrows* by ~5 % where the hypothesis needed it to widen by
> ~110 %, and per-cell energy agrees within ±0.15 J, the small shifts offsetting (§L22). **So the 2.1–2.3×
> understatement is real, measurement context is not its cause, and the beam-width axis of the
> frontier stays unquotable.** This is a narrowing, not a fix: what J10 bought is the elimination of
> the one candidate we could test, and the remaining explanation is the modelling error named in the
> paragraph above — `LM_J` interpolated in T when the penalty is flat in T. Incidental and reusable: a
> resident TensorRT pose context costs ~292 MB and **~0 W** at idle; it holds memory, not power.

> **On the 2.1–2.3× figure.** An earlier draft said 1.7–2.3×, twice. The error originated in
> `results/RESULTS.md` §5.4, whose composition row listed the composed value at T=263 as +2.67 J — a
> linear *extrapolation* of `LM_J` past its measured grid top of 215. **`frontier.py` clamps instead
> of extrapolating**, so the value it ships is 11.04 − 8.88 = 2.16 J, and the band is 4.50/2.16 = 2.08×
> to 4.65/2.04 = 2.27×. Quoting 1.7× described a model we do not ship. Fixed at source by the LM
> track; the pose track had also "corrected" the figure in the wrong direction once, which is what let
> it regress.

---

## 7. Empirical rigor

### 7.1 The protocol, and what applying it changed

- **Three runs, mean ± std, as separate *processes*.** Our earlier "3 runs" were three passes inside
  one process, which cannot see process-to-process variance — and §2.2b had already measured that as
  the larger term. Run properly: TRT reproduces at **0.2–0.3 % CV** in both precisions, while
  `imread` and `preprocess` sit at **~9 % CV, and only at FP16**. The GPU is the stable part; the CPU
  is not. FP16 is both the faster configuration and the jittery one, because it is the one the CPU
  bounds.
- **DVFS logged on every run** — `gpu_MHz`, `cpu0_MHz` and `temp_tj_C` in every power file.
- **Sustained runs** — §4.2.
- **Occupancy recorded on a shared board.** An unrecorded co-tenant invalidated a whole set of memory
  probes, and that bad result propagated across both tracks before it was caught.
- **Paired bootstrap, one set of draws scoring every configuration**, so deltas and
  differences-of-deltas are mutually comparable. Subtracting two independently-bootstrapped intervals
  is both wrong and less sensitive.

### 7.2 Report both metrics — either can be blind to your effect

This is the methodological result we would most defend, and it reversed on us once:

| effect | BLEU-4 | ROUGE-L |
|---|---|---|
| frame rate 30→16, n=967 | −0.35 [−1.00, +0.45] — **not** established | −1.34 [−2.42, −0.30] — established |
| pose substitution, n=400 | −2.24 [−3.50, −1.09] — **established** | −1.11 [−2.56, +0.35] — not established |
| **pose substitution, n=931** | **−1.41 [−2.19, −0.63] — established** | **−1.33 [−2.23, −0.49] — established** |

> **UPDATED 2026-10-05, and the update makes this section's argument stronger rather than weaker.**
> At full width the pose row establishes on **both** metrics, so the clean "the metrics swapped roles"
> story does not survive: that asymmetry was itself a power artefact at n=400. What replaces it is a
> sharper version of the same lesson. The single metric anyone would have reported for this effect —
> BLEU-4, the one that resolved it — **overstated it by 0.83**, while ROUGE-L, the metric we called
> blind to it, was nearer the full-width answer all along at −1.11 against −1.33. So reporting one
> metric is unsafe not only because it may miss an effect, but because the metric that *does* resolve
> an effect at small n can be the one furthest from the truth. The frame-rate row is unchanged and
> still establishes on ROUGE-L only.

Same clips, same draws within each row. **Read down a row, not across rows**: the two rows are on
different decode protocols (batch 8 and batch 1), and §L18 measures the ROUGE-L offset between them at
~0.35 — negligible for BLEU-4 at ≤0.05, but not for ROUGE-L. The point each row makes is internal to
it, so the comparison the table exists for is unaffected; comparing −1.34 against −1.11 as magnitudes
is not. Our first reading was "ROUGE-L is more powerful than BLEU-4",
which is **wrong**: BLEU-4's interval is the *narrower* of the two in **all four** comparisons we have
on shared draws (by 10–26 %, each ±~5 percentage points from bootstrap noise — the sign is stable, the
magnitudes are not). Precision never reversed; **effect size** did.

The defensible claim: **BLEU-4 is precise and partly blind to fluency-type degradation; ROUGE-L is
noisier and responds to it.** Our working explanation, untested: frame-rate thinning and adaptation
change fluency and ordering, which longest-common-subsequence recall tracks; pose substitution changes
which content words appear, which n-gram precision is sharp about.

**Why it matters beyond us.** The sign-language translation literature routinely reports sub-1-point
BLEU-4 differences at exactly these sample sizes. At n≈1000 BLEU-4 cannot resolve sub-1-point effects —
and the reversal sharpens rather than weakens the point: **reporting a single metric is unsafe in
either direction**, because the metric you picked may be the one blind to your effect.

A further wrinkle with a practical edge: **subsampling to 24 fps attenuates the ROUGE-L signal of
pose-extraction quality by 48 %** (−2.12 → −1.11) while leaving BLEU-4's almost intact (6 %). 24 fps
is free on accuracy and *not* free on measurement sensitivity — anyone benchmarking pose front-ends on
subsampled data is using a blunted instrument and the accuracy numbers would never show it.

---

## 8. What we got wrong

Kept deliberately; several of these are the project's more useful output.

1. **"16 fps at no measured accuracy cost"** — claimed from n=30 where the interval was ±3. At n=976
   it is a −1.33 loss. We read "not significant" as "no effect".
2. **"Our pose extractor costs nothing detectable"** — the same error, same shape: −0.35 inside a ±4
   interval at n=30, established by n=400 and **−1.41 [−2.19, −0.63] at full width (n=931)**.
   **Third instance**, which is why §7.1's sample-size discipline exists. A fourth lesson sits on top
   of it: the n=400 estimate of −2.24 was itself 0.83 too large, so "established" is not the end of
   the sample-size story — a resolved effect can still have a materially wrong magnitude.
3. **A memory ceiling adopted without asking how it was measured.** The probes behind it ran on a
   shared board with no occupancy check. One track adopted the other's number and retracted a correct
   finding of its own on the strength of it.
4. **"Comfortably brackets the residual"** — a ~10 % swing estimated from the single largest observed
   drift; measured across three processes it is ~3 % at 1σ, so the residual sits at ~2σ and is
   consistent with, not demonstrated by, run-to-run variation.
5. **A derived ratio computed from a rounded figure in prose** rather than from the source JSON. The
   ratio moved 5.5 percentage points. Bootstrap bound noise of ≤0.07 is enough to do that.
6. **Two latent data-integrity bugs**, found by reading rather than by failure. The pose fetcher
   accepted truncated files forever (resume checked existence, not size) — audited, all 1,281 files
   clean, it never fired. The clip fetcher had the same hole and it **did** fire twice, putting a
   4-frame clip for a 7.8-second utterance into our n=400 set, where our pipeline answered "No." to a
   15-word reference. Both now verify sizes and rates; the contaminated clip accounts for 0.05 of the
   **n=400** 2.24 gap and the figures are left as measured. At n=931 one clip carries 0.43× the weight
   it did at n=400, so its contribution to the −1.41 full-width figure is smaller still.

---

## 9. Limitations

- **Accuracy is offline.** Translation quality is scored on a workstation; the board contributes the
  keypoints and the end-to-end timing. The board↔offline bridge is a batch-size-1 agreement check, and
  batch size does change output (2 of 30 clips differ between batch 8 and 1).
- **Reduced frame rates are emulated** by thinning 30 fps keypoints — exact for the pose model, not
  equivalent to a real low-rate camera's exposure and motion blur.
- **The streaming case is unmeasured.** Our implementation is sequential — all frames, then the
  language model — so the stages never overlap and no contention appears. A pipeline decoding sentence
  N while extracting frames for N+1 **would** contend. Our end-to-end numbers must not be cited as
  evidence that streaming is free.
- **Hands are the accuracy floor** in every configuration (~99 % of keypoints within 5 px, p99.9
  ~30 px) and are unchanged by the precision fix, so hand error is argmax instability in the pose
  head — not something a number format or, per §5, a sibling model will fix.
- **One clip for all end-to-end timing.** Energy generalises better than it looks (§6: 85 % of
  per-frame cost is fixed, 15.1× crop area costs 1.34× energy), but latency on a single clip is still
  a single clip.
- **Sample sizes.** Accuracy at n=976 (authors' keypoints) and n=400 (ours); the pose-substitution
  interval is ±1.2 BLEU-4 and the full 931-clip split would roughly halve it.

---

## 10. Where it stands

**Done:** baseline reproduction and the extractor identification; the pose ladder with an accuracy
axis on every row; vocabulary pruning with its leak fix; INT8 and TensorRT-decoder negatives;
the decode-knob and frame-rate studies; the frontier; the full C9 protocol including sustained runs
and process-level repeats; end-to-end deployment of the recommended cell; the accuracy of the shipped
system at **n=931, every paired test clip that exists**, and the diagnosis of its one deficit.

**Outstanding:** **J9 step 2** — the adaptation training itself, the single remaining accuracy lever.
Step 1 is **done**: 918 dev clips extracted on the board in 95.5 min, packaged and checksummed
(§2.5l). And the **demo video**, which needs a recording session rather than compute.

> **Corrected 2026-10-06.** An earlier draft listed "the in-process `LM_J` re-run that would close the
> frontier's last open caveat" as optional future work. **It ran (J10, §2.9E) and it did not close
> that caveat** — residency is excluded as the cause, the decoder-width term is still understated
> 2.1–2.3×, and the beam-width axis is still the one axis §6 tells people not to quote. The remaining
> candidate is the modelling error §6 names: `LM_J` interpolated in T when the measured penalty is
> flat in T. Listing a completed run as optional future work, and crediting it with closing something
> it did not, were both wrong.

**The headline.** On a 15 W Jetson Orin Nano, the pipeline translates a sign-language utterance in
**7.0 s for 37.8 J**, sustains that for 30 minutes without throttling, and scores **21.73 BLEU-4**
against a **23.14** ceiling set by the reference keypoints, on all 931 paired test clips — with the
**1.41**-point difference diagnosed as a training-distribution shift rather than a limit of the
hardware or the extractor.
