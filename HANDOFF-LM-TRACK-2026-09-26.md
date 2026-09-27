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

## 1. The one finding that may invalidate the current LM plan

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

Cross-check worth knowing about: we extracted those same 5 clips *out of* the 976-clip Mac run and
scored them alone — **16.23**, identical to the separate board run to two decimals. So the board LM and
the Mac LM agree exactly on these clips. **Your Mac-side results transfer to the board**; there is no
platform or precision drift in the LM path to worry about.

Consequences:
- **A 5-clip BLEU number and a 976-clip BLEU number are not comparable.** The subset alone is worth
  ~7 BLEU-4. No small-n BLEU figure in this project is an absolute score.
- The pose front-end costs **6.33 BLEU-4** on these clips, 95 % CI **[+0.48, +13.11]** — the sign is
  established, the magnitude is not remotely. At the bottom of that interval it is negligible; at the
  top, if it held at full scale, we would report ~16.8 instead of ~23.16. **This is the largest open
  accuracy risk in the project** and it is a pose-track problem, not an LM one.
- A 30-clip run is in flight tonight to narrow it; a 100-clip run is planned next.

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
