# What we actually researched, and why — the whole story in plain words

*2026-10-03. Written to be read start to finish by someone who wants to understand the project, not
to look up a number. Every figure here is measured; where something is composed, emulated or
estimated, it says so. The lookup tables live in `results/RESULTS.md`.*

*Companion: [ORIENTATION.md](ORIENTATION.md) was written the same day from the pose/board side. This
file is the **narrative** — how the research got where it did, in order, with the dead ends. That one
is the **map** — what data flows into which system, every choice with its evidence, and the hardware
facts that keep biting. Read this one first if you want the story; that one if you want to find your
bearings quickly. Where they disagree, `results/RESULTS.md` is authoritative over both.*

---

## 1. The problem, in one page

We are translating **American Sign Language video into English text on a small embedded computer**
(an NVIDIA Jetson Orin Nano) that is allowed to draw only **15 watts**.

The pipeline has two halves:

```
video frames → [POSE MODEL] → skeleton keypoints → [LANGUAGE MODEL] → English sentence
                RTMW/RTMPose                        Uni-Sign encoder + mT5
```

The first half watches the video and reduces each frame to a stick figure — where the hands,
fingers, body and face are. The second half reads that sequence of stick figures and writes a
sentence. **Atisri owns the second half.** The teammate owns the first half and all the hardware
measurement.

The deliverable is not "make it accurate" and not "make it fast". It is an **accuracy-versus-energy
frontier**: a plot showing, for every sensible configuration, how good the translation is and how
many joules it costs. The point of a frontier is that it tells you which trade-offs are *real* —
which cheaper settings genuinely cost accuracy, and which are free.

**Why that framing matters.** Almost any model can be made cheaper. The hard question is whether
the cheapening costs anything you can actually measure. A large part of this project turned out to
be about measuring *well enough to know the difference*, which is a theme that comes back repeatedly
below.

---

## 2. First: get a baseline you can trust

Before optimising anything, we had to be sure we could reproduce the published result. If your
starting point is wrong, every "improvement" you measure afterwards is meaningless.

The Uni-Sign paper reports **22.67 BLEU-4 / 42.77 ROUGE-L** on the OpenASL test set, pose-only.

Two checks:

1. **Score their own released predictions with our scoring code.** Got 22.64 / 42.83 — so our metric
   implementation agrees with theirs. (Gotcha found here: their released files prefix every line
   with `sample: <clip>, ground-truth:`. Scoring them raw gives a bogus 58.7 BLEU-4. A silent
   50-point error from a text prefix.)
2. **Run their checkpoint ourselves.** Got **22.53 / 42.68**. The 0.14 gap is within the loader's
   random frame subsampling for long clips.

**Decision: treat 22.5–22.7 BLEU-4 as the FP32 reference band.** Everything later is measured
against that.

### What 22 BLEU-4 actually looks like

Worth internalising, because it shapes how much any of our deltas matter:

> **Reference:** *"I am Ed Bosson, and I am 67 years old and now retired."*
> **Prediction:** *"Hello, I'm Howard Rosenblum, and I'm a fourteen-year-old immigrant."*

Fluent, grammatical, and wrong in every specific. That is the field's state of the art on this
dataset. We are optimising the energy cost of a system that is not yet usable — which is a
legitimate research goal, but it means "accuracy loss" has to be read in context.

---

## 3. The design decisions, one at a time

Each of these follows the same shape: **a question, an experiment that could have gone either way,
and a decision.** This is the core of the research.

### 3.1 Which pose extractor made the training data?

**Why it mattered.** The language model was trained on keypoints from *some* pose model. If we feed
it keypoints from a *different* pose model at deployment, the inputs are subtly off-distribution and
accuracy drops for a reason that has nothing to do with our compression. We had to identify the
original.

**The experiment.** Take one clip, run several candidate pose models on it, and compare each against
the authors' own released keypoints — not in raw pixels (which differ by cropping) but in the
model's own normalised coordinate space, after fitting out the framing.

| Candidate | distance: body / left hand / right hand / face |
|---|---|
| **RTMW-l-m 256×192** ("lightweight") | **0.091 / 0.053 / 0.041 / 0.012** |
| RTMW-x-l 384×288 ("performance") | 0.141 / 0.139 / 0.056 / 0.020 |

**Decision: RTMW-l-m at 256×192 is the baseline.** Closest on every keypoint group. The bigger,
"better" 384×288 model is *worse* for us — it's the oversized comparison row.

**The lesson, which generalises:** a more accurate pose model is not automatically better for the
downstream task. Matching the training distribution beat raw keypoint quality.

A sub-finding here cost real time: our keypoints initially disagreed with the authors' by a factor
of **1.68**, which looked like a model difference. It was a **coordinate-frame mismatch** — a
cropping convention, not an accuracy problem.

### 3.2 Vocabulary pruning — the big, nearly-free win

**The observation.** The language model is mT5, which was trained to handle **101 languages**. Its
vocabulary is 250,000 tokens. We translate sign language into **English only**, on one dataset.
Almost all of that vocabulary is dead weight — but it is expensive weight, because the token
embedding table and the output layer are both sized by it.

**The experiment.** Count every token that actually appears in OpenASL, keep those plus the special
tokens and the prompt, and physically slice the embedding matrix and output layer down to that set.

| | before | after |
|---|---|---|
| parameters | 587.7 M | **243.6 M** |
| checkpoint size | 1187 MB | **571 MB** |
| accuracy | — | **−0.28 BLEU-4**, CI [−0.63, +0.05] |

**Decision: always prune. This is the single best compression in the project.** It removes 59% of
the parameters for an accuracy change that is not statistically distinguishable from zero.

**And then we found a bug in our own method.** The keep-set was built from train + dev + **test**.
That is a test-set leak: the model's vocabulary was chosen using the data we evaluate on. 53 tokens
appeared *only* in test — including `▁Bitcoin`, which happens to be the key word in our demo clip.

We rebuilt the vocabulary from train + dev only (26,078 → 26,025 tokens) and measured the damage:

> **−0.03 BLEU-4, CI [−0.11, +0.01]**

**This is a precise null, not an underpowered one** — an important distinction. The interval is tight
because the two models differ by only 53 of 26,078 rows, so most sentences come out word-identical.
We can say the leak's effect is bounded at about 0.1 BLEU-4. The honest number is **22.84**, not
22.87, and that is what we now report.

**The lesson:** the leak didn't change the result, but we couldn't have known that without
measuring. Finding it mattered more than fixing it.

### 3.3 The output-length cap — free, and verifiably so

**The question.** The decoder generates up to `max_new_tokens` tokens. Lowering that cap could save
decode time. Does it cost accuracy?

**The experiment.** Score the full test set at cap 100 and cap 64.

**Result: bit-identical.** 0 of 976 sentences changed. Not "close" — *the same*. No prediction was
anywhere near the cap, so lowering it removes a ceiling nothing was touching.

**Decision: cap 64 everywhere.** Free.

This turned out to be unexpectedly useful later: it let us mix cells measured at cap 100 with cells
measured at cap 64 in the same table, because we had *proven* they agree on this model.

### 3.4 Weight-only INT8 — a good idea that lost on the evidence

**The idea.** Store the weights as 8-bit integers instead of 32-bit floats. Standard compression.

**What we built.** Weight-only INT8, properly called **W8A32**: int8 weights, but activations and
accumulation stay in FP32. Checkpoint 571 → **293 MB**.

**Why we dropped it anyway — two independent reasons:**

1. **It is slower, badly.** Our INT8 runtime is **10× slower per decoder step** (2139 ms vs 197 ms),
   because 217 layers have to dequantize on every single generated token.
2. **The decoder is host-bound, not compute-bound.** On the board, the encoder takes 39–50 ms while
   the decoder takes **1062–1538 ms**. The decoder's cost is per-step overhead, not matrix
   multiplication. Making the matrices smaller does not help something that isn't limited by matrix
   size.

**Decision: INT8 is dropped. Pruned FP32 is what deploys.**

**This decision was reversed and then un-reversed, and that history is instructive.** The teammate
reported that memory was critically tight at model load, which would have made a 293 MB checkpoint
valuable regardless of speed. We accepted it and partly reopened INT8. They later found those
measurements had been taken on a **shared board with no occupancy check** — another user's memory
use was being attributed to our model. Re-run on a verified-empty board, the ceiling did not exist.
The claimed "4.2 GB peak" had never been measured at all; it was arithmetic. Measured peak is
**0.98 GB**.

So INT8 stayed dropped, for the original reason. More on what we learned from this in §7.

### 3.5 Exporting mT5 to TensorRT — a clean negative result

**The plan.** Convert the language model to TensorRT (NVIDIA's optimised inference runtime), which
usually means a large speedup, and run it in FP16.

**What happened.** The FP16 engine produced **token 0 at every single step**, with log-probability
−10.169. That number is exactly **−ln(26078)** — a perfectly uniform distribution over our pruned
vocabulary. The model wasn't translating badly; it was outputting *nothing*, having overflowed.

This is a known mT5 characteristic: it has activation magnitudes that exceed the FP16 range.

**Decision: the deployed system is a hybrid** — pose model in TensorRT FP16, language model in
PyTorch. Written up as a deployment reality rather than a failure.

**Why this is a result and not just a setback:** "convert everything to FP16 TensorRT" is the default
advice in embedded inference. For this model family it silently produces garbage. The failure
signature (uniform logprob = −ln(vocab size)) is a diagnostic worth knowing.

### 3.6 Beam width — the obvious lever that turned out to be the wrong one

**The question.** Beam search explores several candidate sentences; greedy decoding takes the best
token each step. Beam 4 is standard. Is it worth it?

**Accuracy cost** (976 clips, paired bootstrap):

| | Δ BLEU-4 vs beam 4 |
|---|---|
| beam 2 | −0.81 [−1.29, −0.39] |
| greedy | −2.00 [−2.63, −1.41] |

Both established losses. **Energy saved:** beam 4 → greedy drops the LM from 11.04 J to 8.88 J.

That looks like a good trade — **24% less LM energy** for 2 BLEU-4 — and for a while we reported it
as the main lever.

**That was wrong, and the correction is the most important insight in the project.** The language
model is only about **25% of total system energy**. The pose model, running on every frame, is most
of the rest. So saving 24% of the LM is saving ~5% of the *system*.

Priced properly, per BLEU-4 point sacrificed:

| lever | joules saved per BLEU-4 point given up |
|---|---|
| frame rate 30 → 24 | **103.5 J** |
| frame rate 30 → 16 | 12.3 J |
| beam 4 → beam 2 | 1.0 J |
| beam 4 → greedy | 1.1 J |

**Frame rate is 10–100× the more efficient lever.** "Beam width is the frontier" was true of the
language model *in isolation* and false of the system. **Lesson: optimise the stage that dominates
the budget, and always check what fraction of the system your stage actually is.**

### 3.7 Frame rate — the real lever

**The idea.** Sign language is slower than video. Do we need every frame? Running the pose model on
fewer frames saves energy linearly, because it runs once per frame.

**An important methodological catch.** OpenASL is **not one frame rate**. Measured across 400 test
clips: 73% are 30 fps, 21% are 24 fps, the rest 25/31/60. So "drop every other frame" means
different things on different clips. We wrote `fps_ratio_for_clip`, which reads each clip's own rate
from its filename timestamps and thins it to hit a true target rate.

**Results** (976 clips, paired bootstrap):

| target fps | Δ BLEU-4 | verdict |
|---|---|---|
| **24** | **+0.02 [−0.38, +0.45]** | **free** |
| 16 | −1.12 [−1.71, −0.48] | established loss |
| 12 | −2.52 [−3.22, −1.85] | large loss |
| 8 | −8.02 | unusable |

**Decision: 24 fps is free and should be taken.** On 30 fps clips that is 20% fewer frames, so ~20%
off the pose stage's energy, for an accuracy change centred on zero.

**Caveat we are careful to state every time:** these rates are **emulated** by thinning
already-extracted 30 fps keypoints. That is mathematically exact for the pose model, because pose
extraction is per-frame independent — a kept frame gets exactly the keypoints it would have got. But
a *real* 16 fps camera has different exposure and motion blur. We label this as emulation everywhere.

---

## 4. Adaptation: can training recover what compression costs?

16 fps saves 47% of pose energy but costs about 1.1–1.3 points. **Can we train the model to cope
with the lower frame rate and get that back?**

The setup: freeze mT5 entirely, train only the pose-encoding stack (5.35 M of 243.6 M parameters) on
frame-rate-thinned inputs.

### The first attempt failed, and the control is what saved it

The first 16 fps adaptation run came back *worse* than no adaptation. The tempting conclusion is
"adaptation doesn't work here."

Instead we ran a **no-shift control** — the identical recipe, but trained on normal-rate data, where
by construction there is nothing to adapt to. It also degraded, by −1.04.

**That proved the problem was the training recipe, not the frame rate.** The culprit was label
smoothing at 0.2, far too aggressive for fine-tuning a frozen-decoder model on a small set.

**This is the single most valuable debugging move in the project.** Without the control we would
have published a false negative about adaptation.

### The result, after the fix

| | effect | verdict |
|---|---|---|
| cost of 16 fps, un-adapted | **−1.34 ROUGE-L [−2.35, −0.33]** | established |
| recovery from adaptation | **+1.04 ROUGE-L [+0.25, +1.80]** | established |
| *does adaptation help more under bigger shift?* | +0.42 [−0.58, +1.53] | **NOT established** |
| spread across 3 training seeds | 0.05 BLEU-4 / 0.15 ROUGE-L | the noise floor |

So: **adaptation works and recovers most of the 16 fps cost.** But the appealing story — "adaptation
specifically corrects frame-rate shift" — is *not* supported. The difference-in-differences interval
includes zero. We claim the first and explicitly refuse to claim the second.

The seed spread matters too: 0.15 ROUGE-L across retrains means the +1.04 gain is ~7× the noise, so
it isn't a lucky seed.

---

## 5. Putting it together: the frontier

The deliverable, `results/frontier.png`. Nine configurations — 3 decoder settings × 3 frame rates —
plotted as accuracy against energy per sentence, with confidence intervals.

| | cell | system J | BLEU-4 | vs reference |
|---|---|---:|---:|---|
| reference | beam 4 @ source rate | 43.4 | 22.87 | — |
| **knee** | **beam 4 @ 24 fps** | **36.1** | **22.80** | **−17% energy, −0.07 BLEU-4 (CI spans zero)** |
| cheapest | greedy @ 16 fps | 25.7 | 19.83 | −41% energy, −3.04 BLEU-4 |

**The recommendation is beam 4 @ 24 fps.** It is free within measurement noise and saves a sixth of
the system's energy. Below that the frontier gets expensive fast — the remaining 24% of energy costs
about 3 BLEU-4.

**The honest caveat, annotated directly on the figure.** The energy numbers are **composed**, not
measured end to end: `pose J/s × clip seconds + LM J`. We know two biases:

- **~6% low** — the composition predicts 47.2 J where the one real end-to-end run measured 50.37 J.
  It doesn't model the format-conversion step, the idle draw during model load, or warm-up.
- **~2.6% high** — the pose energy rate came from a clip whose crop is at the 83.4th percentile of
  size, while the duration is the mean clip's. **This figure was ~7% until 2026-10-04**, when the
  pose track measured pose energy across five clips spanning **15.1× of crop area** (§2.9D / J6) and
  found 15.1× the area costs only **1.34× the energy** — the cost is **85% fixed**, fit
  `mJ/frame = 109.7 + 49.5/Mpx`. The old ~7% assumed all ~11.1 ms of per-frame CPU time scaled with
  area; only ~4.5 ms does. An overestimate of about 3×.

They push in opposite directions but **no longer cancel**: at ~6% low against ~2.6% high the net is
**~3–4% low**, so the absolute energy column is mildly optimistic rather than roughly unbiased.
*Relative* comparisons across cells are unaffected either way, because every cell shares the same
clip and rates. But we do not quote a composed cell as a measured one.

### The one real end-to-end run

The teammate ran the full pipeline in one process on the board: **9052.8 ms ± 335.6 per sentence,
50.37 J**, for an 8.51-second clip. Three findings from it:

1. **It is 1.06× slower than real time, and the language model is why.** Pose alone runs at 0.78×
   real time — comfortably live. The LM's 2.4 s per sentence pushes it over.
2. **No contention penalty** — because the implementation is sequential (all frames, *then* the LM),
   the two stages never overlap. A *streaming* design would contend, and that remains unmeasured. We
   are careful not to cite this as evidence that streaming is free.
3. **FP32 doesn't fit end to end** — it runs out of memory in the attention cache. FP16 for the pose
   stage is a *fitting requirement*, not just an energy choice.

---

## 6. How we learned to measure — the methodology thread

This turned out to be as much of the research as the compression was.

### Confidence intervals on everything, paired

Early on we reported point estimates. With ~1000 clips, a 1-point BLEU difference can easily be
noise. Everything is now a **paired bootstrap**: resample clips, score both systems on the *same*
resample, and take the distribution of the difference. Pairing is much more sensitive than comparing
two independent intervals, because clip difficulty cancels.

For the 3×3 grid we went further: **one set of resample draws scores every cell**, so cells are
comparable with each other, not just each against the reference.

### The finding we think is genuinely publishable — and a reversal that makes it better

Here is the sharpest result in the project, in the form we first found it.

We measured the same effects with BLEU-4 and ROUGE-L, on **identical clips, with identical resample
draws**. For the adaptation experiments:

- **ROUGE-L establishes both effects** (the 16 fps cost and the adaptation recovery).
- **BLEU-4 establishes neither.**

At n=300, BLEU-4 even **flipped the sign** of an effect we could confirm with more data.

**Then on 2026-10-03 we found the reverse, and it is the more interesting result.** On the n=400
board evaluation, the *pose-substitution* effect — swapping the authors' keypoints for our own
extractor's — comes out:

| metric | delta (n=400) | 95 % CI | verdict | **delta (n=931)** | **95 % CI** | **verdict** |
|---|---:|---|---|---:|---|---|
| **BLEU-4** | **−2.24** | **[−3.50, −1.09]** | **established** | **−1.41** | **[−2.19, −0.63]** | **established** |
| ROUGE-L | −1.11 | [−2.56, +0.35] | **not** established | **−1.33** | **[−2.23, −0.49]** | **established** |

Same clips, same paired bootstrap — and at n=400 the metrics had swapped roles.

> **Corrected 2026-10-05 (RESULTS.md §L19).** Once J12 extended the board pose set from 400 clips to
> all 931, **ROUGE-L establishes this effect too.** So the swap was a **power limit at n=400, not a
> property of the pose-source axis** — the clean "each axis registers on its own metric" story does
> not survive the larger sample, and we should not tell it that way.
>
> What *does* survive, and is still the useful lesson: **either metric can be blind to a real effect
> at a sample size you can afford**, so report both. BLEU-4's interval remains the narrower of the two
> here as everywhere (±0.78 against ±0.87). The gap itself also moved, −2.24 → −1.41, because the 531
> added clips are harder for the *authors'* poses; §L19 has the decomposition.

> **SUPERSEDED 2026-10-05; this narrative is dated 10-03 and is kept as written.** At the full 931
> paired test clips the gap is **−1.41 [−2.19, −0.63]** on BLEU-4 and **−1.33 [−2.23, −0.49]** on
> ROUGE-L — **both established**, and converging on each other. So the swapped-roles reading above was
> a power artefact of n=400, and the BLEU-4 figure here is 0.83 too large. `results/RESULTS.md` §2.5k
> is authoritative; this file is not.

**So the claim is not "ROUGE-L is more powerful than BLEU-4."** That was our reading when we had one
family of effects, and it was too simple.

**And "the metrics swapped roles" was also too simple** — the pose track caught this and the
correction matters. *Precision* and *sensitivity* are different things, and only one of them
reverses. Comparing the two metrics within each effect, on identical clips and draws, **BLEU-4's
interval is narrower in all four comparisons we have** — by roughly 26 %, 10 %, 21 % and 17 %,
though none of those should be read to better than about ±5 percentage points, because they are
ratios of two bootstrap-noisy widths and inherit noise from both. The **sign** is what is stable, not the
magnitudes: they cannot be ranked against each other. It is the
tighter estimator in the effect it *fails* to resolve just as much as in the one it resolves.

What reverses is **effect size**: the frame-rate effect is 3.9× larger measured in ROUGE-L, and the
pose-substitution effect 2.0× larger measured in BLEU-4.

So BLEU-4 is the more precise estimator in every comparison we have. But the next part — *which*
effect each metric can see — took one more correction, and the answer turned out not to be about the
metrics at all.

**It is an interaction with subsampling.** ROUGE-L establishes pose substitution perfectly well at
source rate: **−2.12 [−3.46, −0.78]**. The divergence appears *only* at 24 fps. Going from source to
24 fps attenuates the **ROUGE-L** response to pose substitution by **48 %** (−2.12 → −1.11) and the
**BLEU-4** response by **6 %** (−2.39 → −2.24). Frame thinning selectively destroys the ROUGE-L signal
of pose-extraction quality.

**And that produces the most practically useful finding in this whole thread: 24 fps is free on
accuracy but *not* free on measurement sensitivity.** The same subsampling that costs nothing on
either metric halves our ability to detect pose-quality differences with ROUGE-L — and the accuracy
numbers would never reveal it, because they are unchanged. **Anyone comparing pose front-ends on
subsampled data is using a blunted instrument and has no way to tell from the scores.**

That is a stronger and more checkable claim than anything about metric power, and it is the one we
would lead a paper with.

Our working hypothesis for the mechanism, stated as a hypothesis because we have not tested it:
frame-rate thinning and adaptation mostly change *fluency and structure* — the sentence stays about
the same length and says about the same thing, less well — which ROUGE-L's longest-common-subsequence
recall picks up and BLEU-4's n-gram precision smears out. Pose substitution instead changes *which
content words appear*, because different keypoints produce different nouns and names, and that is
precisely what n-gram precision is sharp about.

**Why this matters beyond us.** The sign-language translation literature routinely reports
sub-1-point BLEU-4 differences as findings, at exactly these sample sizes. The original observation —
that BLEU-4 cannot resolve sub-1-point effects at n≈1000 — is a checkable claim about a subfield's
evaluation practice. The reversal sharpens it rather than weakening it: **reporting a single metric
is unsafe in either direction**, because the metric you happened to pick may be the one blind to your
effect. This is still the piece we would build a paper around, and it now needs both metrics on both
families of effect to be the paper.

### "Not measured" beats a plausible number

A standing rule: no estimated or extrapolated figure is ever presented as a measurement, and no
placeholder fills a table cell. If it wasn't run, it says so. The deliverable is a comparison, and
one invented cell invalidates the comparison it sits in.

---

## 7. What we got wrong

Keeping this list is part of the work. Every item was caught and corrected.

1. **"16 fps at no measured accuracy cost."** Claimed from an n=30 experiment where the interval was
   enormous. At n=976 it is a real loss. **24 fps is the free rate, not 16.** The small-sample
   interval technically included zero — we read "not significant" as "no effect", which is wrong when
   the interval spans ±3 points.
2. **"Adaptation helps more under frame-rate shift."** The interval is +0.42 [−0.58, +1.53].
   Retracted.
3. **A truncation hypothesis.** Proposed that the adaptation gain was underestimated because of
   output truncation. Measured: cap 64 and cap 100 give bit-identical results. Retracted.
4. **Blamed the composition gap on CPU/GPU contention.** The teammate's end-to-end run measured *no*
   contention. The reasoning was plausible and the measurement disagreed. Retracted.
5. **Adopted the memory ceiling without asking how it was measured.** We took a cross-track
   measurement, struck a *correct* finding of our own, and partly reopened INT8 on it. The
   measurement was from an uncontrolled shared board. **The lesson cuts both ways: a measurement is
   not evidence because it came from the track that owns the hardware.**
6. **Fabricated a sample size.** An early draft said a result "does not survive n=1943", implying a
   pooled analysis that was never run. Caught when Atisri asked "how did you test this?" Corrected
   with explicit provenance.
7. **"Beam width is the frontier."** True of the LM alone, false of the system. See §3.6.

---

## 8. Where it stands

**Essentially finished (LM side):** baseline reproduction, vocabulary pruning and its leak fix, the
cap sweep, INT8 evaluated and dropped, TensorRT export and its FP16 failure, the decode-knob study,
the frame-rate study, the adaptation harness and result, seed variance, the 3×3 accuracy surface,
and the frontier plot.

**Open, and mostly on the board:**

- **The recommended cell has never been run end to end.** The frontier recommends beam 4 @ 24 fps;
  the only true end-to-end measurement is at source rate. *Updated 2026-10-03 by the pose track:* the
  2×2's first launch did fail all four configs on memory, but that was a missing reclaim step. Re-run
  with reclaim-to-target, **three of the four cells landed** (RESULTS.md §5.3) and the grid has been
  extended to `{source, 24 fps, 16 fps}` per `ASK-E2E-KNEE`, so 24 fps × beam 4 is now in flight.
  The one cell that still failed is `source × beam 4` — and **not** because it does not fit: M1 ran
  that exact cell at *less* free memory. The real finding is that MemFree is not a sufficient
  readiness check. §5.3 has it.
  - What the grid already settles, on the board rather than in the composition model: **frame rate is
    the big knob and beam width the small one.** Source → 16 fps at greedy cuts latency 41.2 % and
    energy 40.2 %; greedy → beam 4 at 16 fps costs 19.3 % latency and 26.9 % energy. It also shows
    `mJ_per_frame` *rises* under subsampling (169 → 190), because the per-sentence LM cost is spread
    over fewer frames — so **end-to-end energy must be quoted per sentence**, not per frame.
- **We have never produced a board-measured BLEU.** All accuracy is offline on the Mac; the only
  bridge to the board is a 30-clip agreement check.
- **The C9 rigor protocol** (3 runs each, DVFS logged, 30-minute sustained thermal run) is not yet
  applied to the final rows.
- Deliberately **not** doing: 12 fps and 8 fps adaptation, the face-group drop, the full 96 K-clip
  training run, and the context-conditioning stretch.

**The report comes last**, once both tracks' results are combined.

---

## 9. Correction to the work-split premise (2026-10-03)

Two constraints this document and `LM-TRACK-REPORT-2026-09-30.md` planned around have both expired,
and one was never true.

**"Anything needing a Python environment is ours to run."** False. The pose track's Mac runs torch
2.8.0, transformers 4.57.6, numpy, safetensors, rouge and portalocker with MPS available — it has
already executed `bootstrap_ci.py` and every paired CI in §2.5f and §2.5g. The claim originated with
them and they have withdrawn it. Only `sentencepiece` (needed for an mT5 tokenizer load) and `cv2`
(needed for any frame or pose path) are genuinely absent, and both are plain pip installs.

**"The board has been unreachable since 2026-09-28."** Expired 2026-10-02.

Consequence for how the remaining work is divided: **verification work that needs a Python
environment but not a GPU can be shared rather than being ours by default.** What is still genuinely
ours alone is GPU training (Colab) and anything touching the adaptation checkpoints; what is still
genuinely theirs is anything on the board.
