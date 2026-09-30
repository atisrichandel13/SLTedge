# Pose-track findings that affect the LM track — 2026-09-26

**Who this is for.** Atisri, and any LLM being asked to reason about the LM track. It is written to be
read cold: every number states its metric and its denominator, because most of the confusion in this
project so far has come from two numbers that looked comparable and were not.

**What it is not.** Not a project overview — that is `BRIEFING-2026-09-26.md`, which is self-contained
and longer. This document is only the delta: what the pose track has learned that changes an LM-track
decision, what we got wrong and retracted, and what we need from you.

**How to treat the numbers.** Every figure here comes from a real run, with the file it came from named.
Where something is inferred rather than measured, it says so. Where a claim was retracted, the
retraction is kept in place rather than deleted, so nothing gets rebuilt on it.

---

## 0. DECISION, 2026-09-28: INT8 is dropped

Tushar's call, on the reasoning in §1. **L9 weight-only INT8 (W8A16) is off the plan.** Nothing needs
to be produced for it, and the accuracy–energy frontier does not need it: the frontier's accuracy axis
is already measured at n=976 with established signs (beam 4 → 22.87, beam 2 → 22.06, greedy → 20.88),
so **beam width is the knob**. The only outstanding piece is board energy per configuration, which the
pose track owns and does not need anything from the LM track.

If it is ever revisited, the only credible variant is INT8 weights with **FP32** activations (PyTorch
dynamic quant), for the reason in §1: FP16 activations are what break mT5.

**Also settled 2026-09-28: M1 exists.** First on-device end-to-end translation, pose engine and LM in
one process: 9052.8 ms ± 335.6 per sentence (pose 6640.7, LM 2386.5), 50.37 J/sentence, peak GPU
2.577 GB. Two results the LM track should know:
- **The LM is what breaks real time.** Total is 1.06× slower than real time; the pose stage alone is
  0.78×. Beam 2 (−0.81 BLEU-4) or greedy (−2.00) would bring the pipeline under real time. That trade
  is now a measured one on both axes.
- **The LM takes 64.4 s to load.** For a live demo that is the number people will notice, and it is
  worth one look at whether it can be cut (safetensors, `low_cpu_mem_usage`, a warm process).
- **Batch size changes output text.** `eval_openasl.py` batches 8 and pads; the deployed path runs 1.
  2 of 30 clips differ, moving BLEU-4 18.52 → 18.64 and ROUGE-L 43.91 → 44.19. Small, but every offline
  BLEU number carries it, so quote the batch size when a number matters.

## 0b. NEW 2026-09-28: your L7 row is measured, and it says encoder length is a weak lever

The pose track ran the LM latency/energy matrix on the board (`unisign/lm_sweep.py`, pruned checkpoint,
one process, each configuration with its own power window). **This is your L7 row — encoder ms and
energy vs T — plus the beam axis.** Raw: `results/lm_sweep_pruned.json`.

| beams | requested T | frames used | total ms | **encoder ms** | **decoder ms** | J/sentence | dyn J | avg W |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 256 | 215 | 1372 | **49** | **1228** | 8.88 | 3.62 | 6.48 |
| 2 | 256 | 215 | 1582 | 50 | 1447 | 10.26 | 4.30 | 6.49 |
| 4 | 256 | 215 | 1676 | 49 | 1538 | 11.04 | 4.78 | 6.62 |
| 1 | 137 | 137 | 1351 | 39 | 1236 | 8.55 | 3.49 | 6.35 |
| 4 | 137 | 137 | 1574 | 41 | 1459 | 9.89 | 3.99 | 6.30 |
| 1 | 68 | 68 | 1264 | 41 | 1168 | 7.79 | 3.08 | 6.19 |
| 4 | 68 | 68 | 1465 | 41 | 1370 | 8.94 | 3.47 | 6.14 |

(Full 15-row matrix in the JSON: beams 1/2/4 × T = 256/205/137/103/68.)

**The finding: encoder length is nearly free to grow and nearly useless to shrink.** The encoder is
**39–50 ms** across the whole range while the decoder is **1062–1538 ms**. Cutting T from 256 to 68 —
a 73 % reduction in pose frames — saves the LM only 8.88 → 7.79 J (12 %), because the decoder dominates
and its cost tracks *tokens generated*, not encoder length.

**Consequences for L7 and the frontier:**
- **Do not present reduced frame count as an LM energy saving.** It is a *pose*-stage saving. The pose
  side scales properly: 4.25 → 1.17 J per second of video from 30 to 8 fps (0.27×). The LM barely moves.
- **Beam width is the LM's real lever**, and both of its axes are now measured: greedy 8.88 J vs beam 4
  11.04 J (2.16 J, 24 % more energy) against the accuracy cost of 2.00 BLEU-4, CI [−2.63, −1.41],
  established at n=976. **That single trade is the accuracy–energy frontier** and it can be plotted now.
- If you want the decoder cheaper, the lever is the step loop, not the input length — L8.2 already
  identified it as host-bound (CUDA graphs, keeping the KV cache device-resident).

## 0c. NEW 2026-09-28: the un-adapted accuracy-vs-frame-rate baseline exists

This is the number your C8/L11 adaptation gains must be measured against, and it did not exist before.
30 clips, released checkpoint, beam 4, batch 1, poses subsampled from full-rate keypoints (exact, since
pose extraction is per-frame independent), paired bootstrap against 30 fps:

| fps | BLEU-4 | ROUGE-L | Δ BLEU-4 | 95 % CI | verdict |
|---:|---:|---:|---:|---|---|
| 30 | 18.64 | 44.19 | — | — | baseline |
| 24 | 20.15 | 46.31 | +1.51 | [−2.32, +5.78] | no measured cost |
| **16** | 18.49 | 43.56 | −0.15 | [−3.20, +2.89] | **no measured cost** |
| 12 | 15.39 | 43.58 | −3.25 | [−7.22, +0.80] | ambiguous |
| 8 | 11.46 | 36.28 | −7.18 | [−12.55, −2.13] | **established loss** |

**16 fps is free within noise and halves pose energy (0.53×).** 8 fps is a real loss, so there is a
floor. 12 fps I would not use: the point estimate is a meaningful −3.25 and n=30 cannot resolve it.
**24 fps looking better than 30 fps is noise** — do not report it as an improvement.

**What this means for C8 (guide rows 4.2/4.4).** The adaptation's job is to recover the loss at the
rates where there *is* one, i.e. 12 fps and below. At 16 fps and above there is nothing to recover on
this evidence, so an adaptation run at 24 fps cannot show a gain and should not be scheduled expecting
one. And per §5b, if C8 trains on the authors' released train poses it closes the frame-rate gap while
leaving the extractor gap untouched.

**One practical warning.** The board had a stale Sep-18 `unisign/bootstrap_ci.py` without the `--out`
flag, which made a board-side CI step fail silently (argparse error, no matching output). It is updated
now, but if you run analysis on the board, check the file you are running is the current one.

## 0d. NEW 2026-09-28: pose-stage energy vs capture rate — the measured joule saving

This is the second half of "is 24 fps free": the energy, as a measured number rather than an inference
from accuracy. Real runs at each rate (`task1_rtmpose/04_infer_power.py --keep-fps`), RTMW-l-m FP16 —
the deployment model — 3 repeats each, board mode 0 (15 W), INA3221 on VDD_IN.

| fps | frames | ms/frame | mJ/frame | dyn mJ | avg W | gpu MHz | cpu MHz | **J per second of video** | **vs 30 fps** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 255 | 19.82 | 141.6 | 40.1 | 4.88 | 309 | 897 | **4.25** | 1.00× |
| 24 | 204 | 21.29 | 139.9 | 33.6 | 4.88 | 308 | 868 | **3.36** | **0.79×** |
| 16 | 136 | 20.68 | 141.8 | 39.1 | 4.85 | 312 | 882 | **2.27** | **0.53×** |
| 12 | 102 | 20.54 | 139.5 | 39.1 | 4.90 | 313 | 901 | **1.67** | 0.39× |
| 8 | 68 | 19.99 | 145.9 | 40.1 | 4.87 | 318 | 1039 | **1.17** | 0.27× |

**So "24 fps is free" is now two measured statements, not one claim:** it costs **21 % less pose energy**
(0.79×) and has **no measured accuracy cost** (§0c, Δ BLEU-4 +1.51, CI [−2.32, +5.78]). And 16 fps is the
better operating point — **47 % less energy**, Δ BLEU-4 −0.15, CI [−3.20, +2.89].

Reading the columns:
- **`J per second of video` is the only comparable column across rates.** `mJ/frame` is flat by
  construction (139.5–145.9) because every frame costs the same to process; what changes is how many
  frames exist per second of video. Quoting mJ/frame as if it showed a saving would show nothing.
- **Do not compare the absolute 141.6 mJ/frame against P2's 127.** Each of the 3 repeats pays 20 warm-up
  inferences and this clip has 255 frames against P2's 299, so the warm-up amortises differently. The
  ratios are unaffected and are what matters here.
- The CPU clock rising at low rates (897 → 1039 MHz) is the 15 W coupling from §3: less sustained GPU
  work leaves the CPU more of the budget.

**Combined with the LM matrix (§0b), for the M1 sentence** (8.51 s of video, 50.37 J measured
end-to-end): pose 36.2 J + LM 11.04 J at 30 fps / beam 4 becomes 19.3 J + 9.89 J at **16 fps / beam 4** —
about **29 J against 47 J, a ~38 % system saving at no measured accuracy cost.** Dropping to greedy saves
a further ~1.3 J and costs 2.00 BLEU-4, which is a trade to present rather than to take silently.

Raw: `results/p8_pose_fps{30,24,16,12,8}.json` and `.csv`.

## 1. Why INT8 was dropped (the original analysis)

**Guide row 3.2 specifies "L9 weight-only INT8 (W8A16) on the pruned mT5". W8A16 keeps activations in
FP16, and FP16 is already known-broken on this model.**

From L8.2 (`results/RESULTS.md` §L8.2, measured on `jetson-lpcv-03`, 15 W):

| Engine set | tokens vs PyTorch | max \|Δlogprob\| | verdict |
|---|---|---|---|
| FP32 | **identical** | 5.6e-4 | passes |
| FP16 | **broken**: 64× token 0 | 10.2 (uniform) | unusable |
| BF16 | diverges at step 3 (22 vs 24 tokens) | 1.1 | unusable |

The FP16 failure is diagnostic, not random: every step returns token 0 with logprob −10.169, which is
exactly −ln(26078) — a uniform distribution over the pruned vocabulary, i.e. saturated logits. This is
the documented T5/mT5 FP16 overflow (feed-forward activations exceed the FP16 max of 65504 in later
blocks; HuggingFace's own fp16 path clamps them). **An INT8 scheme that leaves activations in FP16
inherits this failure.**

Second problem, independent of the first: **INT8 probably buys no latency here.** L8.2 measured the
FP16 decoder step at *no faster* than FP32 (18 vs 19 ms/token) while the FP16 encoder was 3–4× faster
(19–25 vs 82 ms). That asymmetry says the decoder step is **host-bound, not GEMM-bound** — the 19 ms is
dominated by re-feeding 50 input bindings and cloning outputs per step, not by the 238 M-param matmuls.
Quantizing the weights does not make host-side Python faster.

**So the credible variants are:**
- INT8 weights + **FP32** activations (PyTorch dynamic quant — the fallback guide row 3.2 already
  names). Saves memory and bandwidth. Latency gain on the decoder: likely small, for the reason above.
- Leave the LM at FP32 and spend the effort on the step loop instead (CUDA graphs, keeping the KV cache
  device-resident). L8.2 identifies this as the actual next lever.

**What we need from you: confirm or drop "pruned + INT8, beam 4" as the converged config** before
anyone spends a session building engines for it. If it is dropped, the pose track can measure
*pruned FP32, greedy on TRT + beam on PyTorch* end-to-end without any new LM artifact.

## 2. Beam search does not exist on the TensorRT path

`unisign/trt_decode.py` hardcodes `num_beams=1` (line 79), for the reference as well as the engine.
Beam through the explicit-KV-cache engines needs per-beam cache reordering — an implementation task,
not a flag.

`unisign/unisign_infer.py` does take `--num-beams` (PyTorch path), so beam 2 / beam 4 are available
there today.

This interacts awkwardly with another L8.2 finding: **the TRT encoder is slower than PyTorch** (82 vs
57 ms) because with TF32 disabled TensorRT falls back to plain FP32 FFMA kernels while cuBLAS in
PyTorch has better FP32 GEMM tiles. So the genuinely fastest configuration is **PyTorch encoder + TRT
greedy decoder** — which by construction cannot do beam search. Any "beam 4 on TRT" row in a plan is
currently unimplementable, and the config that *is* fastest is greedy-only.

## 3. Power findings that change how LM measurements must be read

**Under the 15 W cap, the CPU and GPU compete, and the loser is measurable.** Measured today
(`results/p2b_split_rtmw_{fp16,fp32}.json`), same input, same code, two engines:

| | RTMW FP16 | RTMW FP32 | ratio |
|---|---:|---:|---:|
| JPEG decode, CPU (`imread_ms`) | 5.33 | 6.83 | ×1.281 |
| preprocess, CPU (`preprocess_ms`) | 4.76 | 6.10 | ×1.281 |
| measured `cpu0_MHz` | 1045 | 897 | ×1.165 |
| measured `gpu_MHz` | 312 | 604 | |

Two independent CPU-only stages slowed by **the identical factor to three digits** when the GPU got
hungrier. The CPU clock drop explains about 60 % of it; DRAM contention with a 604 MHz GPU covers the
rest.

**Why this matters to the LM track, concretely:** every LM number so far was measured with the board
otherwise idle. In M1 (pose → LM in one process) the two stages contend for the same 15 W and the same
8 GB. An LM latency measured alone is an **optimistic** bound on what it gets in the real pipeline, and
the host-bound decoder step (§1) is exactly the kind of work that suffers when the CPU downclocks.
Do not treat the standalone LM rows as end-to-end.

**`cpu0_MHz` is already logged in every power CSV** (`common/power_logger.py`, `aux_avg.cpu0_MHz`). It
had simply never been read next to the timings. Any new LM power run should report it.

## 4. Your results, now with confidence intervals

`unisign/bootstrap_ci.py` (your L12 interface) has been extended, backward-compatibly. Applied to the
existing full-split evals, **976 clips, paired clip-level bootstrap, 1000 resamples**:

| comparison | BLEU-4 | delta | 95 % CI | verdict |
|---|---|---:|---|---|
| released → pruned vocabulary | 23.16 → 22.87 | −0.28 | [−0.63, +0.05] | **not established as a loss** |
| pruned, beam 4 → **beam 2** | 22.87 → 22.06 | −0.81 | [−1.29, −0.39] | **real loss** |
| pruned, beam 4 → **greedy** | 22.87 → 20.88 | −2.00 | [−2.63, −1.41] | **real loss** |

Two things to take from this:

- **Vocabulary pruning is free and the decode knobs are not.** At n=976 the pruning cost cannot be
  distinguished from zero, so report it as "not established as a loss" rather than "−0.28 BLEU-4",
  which reads like a measured penalty. Beam reduction, by contrast, has an established sign in both
  cases — these are genuine accuracy prices, not noise.
- **This puts a number on the TensorRT decision in §2.** The TRT path is greedy-only, and greedy costs
  **2.00 BLEU-4 (CI [−2.63, −1.41])** against beam 4. So "PyTorch encoder + TRT greedy decoder", the
  fastest configuration we have measured, buys a 2.9× decoder speedup at a measured cost of 2 BLEU-4 —
  roughly a third of the entire pose-front-end gap in §5, spent on the decoder alone. Beam 2 on the
  PyTorch path costs 0.81 instead. **That trade is the frontier plot**, and it can be drawn today from
  these three rows plus board energy, with no INT8 at all.

Raw: `results/ci_pruning_976.json`, `results/ci_beam2_976.json`, `results/ci_greedy_976.json`
(976 clips, 1000 resamples, paired on clips, BLEU-4 via sacrebleu 13a).

## 5. The BLEU denominator trap — read this before comparing any two BLEU numbers

All three rows below use **the same released checkpoint and the authors' own released poses**. The only
things that change are which clips, and whose poses.

| | BLEU-4 | what differs |
|---|---:|---|
| 976 test clips | **23.16** | the headline number |
| **our 5 clips** | **16.23** | −6.93, *purely which clips* |
| our 5 clips, **our poses** | **9.89** | a further −6.33, *our pose front-end* |

**UPDATE, same evening, n=30 (this supersedes the −6.33 above).** With 30 clips instead of 5, the pose
front-end gap is far smaller and the 6.33 was a subset artifact:

| | BLEU-4 | ROUGE-L |
|---|---:|---:|
| authors' poses, same 30 clips | 18.17 | 45.03 |
| **ours (RTMW FP16 + frame fix)** | **18.52** | 43.91 |
| ours (RTMW FP32) | 16.55 | 40.72 |

Paired bootstrap at n=30: the raw gap is **+1.61 BLEU-4, CI [−2.57, +5.94] — not established**; only
ROUGE-L was (+4.32, CI [+0.47, +8.30]), and a coordinate-frame correction removes that too. **The best
pose configuration is statistically indistinguishable from the authors' own poses on both metrics.** So
the pose front-end is no longer the project's largest accuracy risk; treat the LM-side decisions in §1
and §2 as the binding ones.

Cross-check worth knowing about: we extracted those same 5 clips *out of* the 976-clip Mac run and
scored them alone — **16.23**, identical to the separate board run to two decimals. So the board LM and
the Mac LM agree exactly on these clips. **Your Mac-side results transfer to the board**; there is no
platform or precision drift in the LM path to worry about.

Consequences:
- **A 5-clip BLEU number and a 976-clip BLEU number are not comparable.** The subset alone is worth
  ~7 BLEU-4. No small-n BLEU figure in this project is an absolute score.
- ~~The pose front-end costs 6.33 BLEU-4 (CI [+0.48, +13.11]), the largest open accuracy risk in the
  project.~~ **Resolved the same evening at n=30: +1.61 BLEU-4, CI [−2.57, +5.94], not established.**
  Those five clips were unusually hard for our poses. A 100-clip run will tighten it further.
- **Also settled at n=30: RTMW beats RTMPose-x by 7.13 BLEU-4 (CI [−10.94, −3.07]).** The paper
  specifies RTMPose-x and we chose RTMW on cost; that choice turns out to be the accuracy-correct one
  on our pipeline, so no front-end switch is coming that would disturb the LM interface.

## 5b. How the C8 adaptation interacts with the gap above

Stated because it decides whether the extractor question matters. **Nothing is training RTMPose** —
RTMPose/RTMW are frozen off-the-shelf keypoint detectors and we only quantize them. C8 (guide rows
4.2/4.4) fine-tunes the *Uni-Sign* pose encoder: the ST-GCN stack plus `pose_proj` and `part_para`,
~4.5 M params, mT5 frozen.

Guide row 4.1 sources the training poses from **the authors' released train poses** (96,477 files, via
`openasl_pose_fetch.py --split train`). If that is what C8 trains on, then:

- **Frame-rate shift is fixed by adaptation.** This is what turns "24 fps is free" from an energy claim
  into an accuracy claim, and it is the right use of the harness.
- **The 6.33 BLEU-4 extractor gap in §5 is *not* fixed by it.** The adapted model is tuned to
  authors'-extractor poses at a new frame rate; our RTMW poses stay out-of-distribution in the same way
  they are now. The ceiling moves down to the new rate and the gap travels along with it.

Closing the extractor gap by adaptation would need train-split poses from **our** extractor — RTMW over
~96,477 clips. At the 8 s/clip we measured for fetch, that is over a week of downloading for video we do
not have. **So the extractor gap has to be closed by choosing the right front-end, not by adapting
around it**, which is why the 30-clip and 100-clip runs stay on the critical path.

Three consequences for how results get recorded:

1. **Every BLEU number needs its checkpoint stamped.** Everything in this document is against the
   released frozen checkpoint (the project's FP32 reference). BLEU against an adapted checkpoint is a
   different quantity and must not share a table with these without a column saying so.
2. **L10 seed variance (row 4.3) has to land before any adaptation claim.** "Adapted vs un-adapted"
   is meaningless without the seed noise band, and that is a *different* noise source from the
   clip-resampling CI in `bootstrap_ci.py`. Row 4.4 asks for a CI; it needs both, and they are not
   interchangeable.
3. **There is a pairing mismatch in the frontier as currently planned.** It would pair *our* pose energy
   at reduced fps (P8, row 2.9) with *their* pose accuracy at reduced fps (L11, row 4.4). The
   un-adapted accuracy-vs-rate curve **on our own poses** does not exist, and it is the correct baseline
   against which an adaptation gain is measured. The pose track can produce it from frames already on
   the board — subsample to 24/16/12/8 fps, RTMW FP16, eval — for about 31 min of board time plus 7 min
   of eval. Treating that as a P8 prerequisite.

## 6. Tooling changes that touch your files

- **`unisign/bootstrap_ci.py`** — I had duplicated it by accident and have deleted my copy. Yours is
  now the single entry point. The original CLI (`a b -n --seed`) is unchanged. Added: alignment by
  clip name when available (falling back to the old positional refs check for older eval output),
  ROUGE-L alongside BLEU-4, `--no-rouge`, `--out`, a distinction between "identical text on every
  clip" and "CI includes 0", and a warning when N is small enough that the resample grid is coarse.
  Verified before merging that both BLEU paths agree: sacrebleu 13a and
  `unisign/metrics.translation_performance` give **9.894897** on the same data, bit-identical to the
  published eval JSON. The CI and the point estimate are therefore the same metric.
- **`unisign/eval_openasl.py`** now records `"names"` — the clip ids it scored. Paired comparisons used
  to align positionally, so a config missing one pkl would have shifted every later sentence against
  its partner **with no error raised**. Old eval JSONs still work via the legacy path, but re-running
  anything you intend to compare is worth it.
- **Metric deps now install on the Mac**: `pip install portalocker rouge`. `unisign/metrics.py` and the
  vendored sacrebleu import them at module level. This means BLEU/ROUGE/CI analysis no longer needs the
  board or the container — only generation does.
- **`jetson/run.sh whoelse`** — new. The board is shared (other accounts: `wjbeksi`, `bzhang`, `zaly`).
  Run it before any timing or power measurement: another tenant inflates latency and power and nothing
  in the result would show it. Checking `docker ps` and `free -m` covers memory safety, not timing
  validity.

## 7. Retracted — do not build on these

- ~~"RTMW FP16 is translation-neutral (5/5 identical sentences)"~~ **RETRACTED.** The same engine pair
  gives 3/5 on a different crop of the same clips while bin-level keypoint agreement is unchanged
  (99.39 % vs 99.76 % within 2 bins). It was a five-sentence coin flip. FP16 remains the choice, on
  keypoint agreement and cost, not on this.
- ~~"The pose→BLEU gap comes from our crop convention"~~ **TESTED, NOT SUPPORTED.** Reimplementing
  OpenASL's exact square/pad/resize-224 recipe moved our keypoint distribution onto theirs but did not
  close the gap: ceiling-identical predictions stayed at 1/5, and BLEU-4 rose (9.89 → 12.38) while
  ROUGE-L fell (43.39 → 38.87) — the metrics disagree in direction. Since their 224 crop *discards*
  resolution relative to our 502–754 px crops, this also rules out pose sharpness. Extractor identity
  is now the leading hypothesis.
- ~~"Preprocess does not shrink with GPU quantization"~~ **CORRECTED.** It shrinks 22 % on RTMW, for
  the power-budget reason in §3.
- ~~"RTMW FP16 runs at the resolution its checkpoint was trained on"~~ **was inferred from a folder
  name and contradicts the paper**, which specifies RTMPose-x. The front-end choice is provisional and
  cost-led, not accuracy-led. See §5 — this is the open risk.

## 8. Memory figures, and which metric each one is

This caused a real misreading, so it is worth stating flatly. **All three of these are the pruned
model.** The TensorRT engines were built from `models/mt5_pruned_onnx/`, which is the pruned export.

| figure | metric | source |
|---|---|---|
| **2.41 GB** | torch **peak-allocated**, *full* 582 M model | L4, `results/l4_jetson_full_fp32.json` |
| **1.03 GB** | torch **peak-allocated**, *pruned* 238 M model | L4, `results/l4_jetson_pruned_fp32.json` |
| **2.07 GB** | **whole-process resident** (MemFree delta), pruned TRT engines | L8.2 |

2.07 > 1.03 for two reasons at once, neither of them "the engine wasn't pruned": it is a different
metric (process-resident vs torch peak-allocated), **and** the 3-graph export duplicates the decoder
weights (`decoder_init.onnx` 616 MB + `decoder_step.onnx` 558 MB both contain the decoder). So
"pruned + TRT FP32" *has* been measured as a combination — at 19.2 ms/token, identical tokens to
PyTorch.

While comparing TRT to PyTorch, use the **pruned** PyTorch row: 19 ms/token vs **55**, a 2.9× decoder
speedup. Comparing 19 against the full model's 69 gives a wrong 3.6×.

## 9. What is on the board right now

- `weights/openasl_pose_only_slt_pruned.pth` (570 MB), `weights/openasl_pose_only_slt.pth` (1.19 GB),
  `weights/mt5-base`, `weights/mt5-base-openasl-pruned`.
- `models/mt5_pruned_onnx/` (1.5 GB): `encoder.onnx`, `decoder_init.onnx`, `decoder_step.onnx`.
- The mT5 **TensorRT engines were cleaned up** (shared board). They rebuild in ~51 s total.
- New test set: **30 clips from 30 distinct YouTube videos, 7299 frames**, plus the authors' released
  poses for all 30 (`data/openasl_pose/`). All 976 test clips have bboxes across 456 videos, so
  scaling further is a matter of download time, not availability.
- The board is otherwise idle and its power mode is 0 (15 W) — every published row is mode 0.

**What the pose track can run for you without any new LM artifact:** pruned PyTorch FP32 latency and
power at beam 1 / 2 / 4, at full and reduced frame counts, plus TRT-FP32 greedy with power.
`common/power_logger.py:run_with_power` is generic, so guide row 1.5 (adding `--power-json` to the LM
scripts) is not a blocker — I can wrap them externally. Say the word and which configs you want.

## 10. Open questions for you

1. **Is "pruned + INT8, beam 4" still the converged config?** (§1 — FP16/BF16 both fail on mT5, W8A16
   inherits the FP16 failure, and the decoder is host-bound so INT8 likely buys memory not latency.)
2. **Is beam on TensorRT worth implementing**, given that the fastest configuration (PyTorch encoder +
   TRT greedy decoder) cannot do beam at all? (§2)
3. If INT8 is dropped, **the frontier plot's LM axis can be the decode knob instead**, and §4 already
   has the accuracy side of it measured with CIs (beam 4 / beam 2 / greedy at 22.87 / 22.06 / 20.88).
   All that is missing is board energy per sentence for those three, which the pose track can measure
   without any new LM artifact. Is that an acceptable substitute for the INT8 axis?
4. Do you want your existing evals **re-run to record clip names** (§6), or is the legacy positional
   path good enough for the comparisons you still plan?
5. **Which pose source is C8 training on** — the authors' released train poses (row 4.1), or something
   else? (§5b) If it is the authors', the adaptation plan closes the frame-rate gap and leaves the
   extractor gap untouched, which is worth being explicit about in the report rather than discovering
   at the frontier plot.
6. **Is L10 seed variance scheduled before the L11/P8 adaptation runs?** (§5b) The adapted-vs-un-adapted
   delta cannot be judged without it.
