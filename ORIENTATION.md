# Orientation — what flows where, what we chose, and why

*Written 2026-10-03 as a refresher. Plain language on purpose. Every number here is quoted from
`results/RESULTS.md` or `PROJECT-GUIDE.md`; if the two ever disagree, RESULTS.md wins and this file
is stale. Nothing in here is new evidence.*

*Companion: [RESEARCH-STORY-2026-10-03.md](RESEARCH-STORY-2026-10-03.md) was written the same day
from the LM side. That one is the **narrative** — how the research got here, in order, including the
dead ends and the retractions. This one is the **map** — what flows into which system, every choice
with the evidence under it, and the hardware behaviour that keeps catching us out. They were written
independently and overlap on purpose; `results/RESULTS.md` is authoritative over both.*

---

## 1. The system in one paragraph

A video of someone signing goes in. We cut it into frames, crop each frame to the signer, and run a
**pose model** that marks 133 points on the body, hands and face. Those points — not the pixels —
are the only thing the rest of the system ever sees. A **graph network (ST-GCN)** reads the sequence
of points and turns it into a sequence of vectors shaped like text embeddings. **mT5-base**, an
ordinary text model, then decodes those vectors into an English sentence. The whole thing has to run
on a Jetson Orin Nano 8 GB held at 15 watts.

So there are exactly two models doing real work, and they have opposite cost shapes:

| | pose model | language model |
|---|---|---|
| cost scales with | **number of frames** | **number of sentences** |
| per unit | ~24 ms and ~127 mJ per frame | ~1.4–2.4 s and ~9–11 J per sentence |
| share of system energy | ~75 % | ~25 % |

**That asymmetry is the single most important fact in the project.** It is why cutting the frame rate
is our best lever and why fiddling with the decoder is not.

---

## 2. Three machines, and why work lands where it does

| machine | what runs there | why |
|---|---|---|
| **Jetson Orin Nano** (`jetson-lpcv-03`, 192.168.1.73 over WireGuard) | every latency, power and energy number; the TensorRT engines; the end-to-end runs | It is the target. The hard rule is that **no hardware number is ever estimated** — if it is in RESULTS.md, a run on this board produced it. |
| **The Mac** | all accuracy numbers (BLEU/ROUGE), pose export to ONNX, the bootstrap statistics, the frontier plot | Accuracy does not depend on the hardware, and 976-clip evals are far too slow on the board. |
| **Colab / the 4×24 GB rig** | training and adaptation runs | We never train on the board, and never pip-install torch on it. |

The board is **shared scratch**. Results get pulled to the Mac and the bulk gets deleted
(`jetson/run.sh clean-large`). Board repo path is `~/sign-lang-project`, and it is *not* a git repo —
it is an rsync target.

---

## 3. What actually flows in, stage by stage

### 3.1 Getting data in — we have two separate input routes

This confused us early, so it is worth being explicit. There are **two different ways** sign data
enters the system, and they are used for different jobs.

**Route A — the authors' pre-extracted poses.** The Uni-Sign authors published the keypoints they
used. `data/openasl_pose_fetch.py` pulls individual files out of a 32 GB HuggingFace archive using
HTTP range requests, so we never download the whole thing. All 976 test-split pose files = 555 MB.

*Used for:* every accuracy number, and as the ceiling we compare our own pose extraction against.

**Route B — raw video, our own extraction.** `data/openasl_fetch.py` uses `yt-dlp` to download just
the needed section of the YouTube video, then crops each frame to the OpenASL signer bounding box at
native resolution and frame rate.

*Used for:* everything on the board, because the board has to do the pose extraction itself.

The crop recipe matters and bit us (see §3.3). We deliberately kept the baseline clip's recipe —
bbox scaled and clamped to the frame — rather than OpenASL's own square-pad-resize-224.

Data we hold: 5 clips / 1510 frames and 360 calibration frames from 6 train signers (first set), then
30 clips from 30 distinct videos / 7299 frames (the evaluation set), and 931 clips / 6.6 GB on the
Mac waiting for a full-split board pass.

### 3.2 Pose model — which one, and at what precision

**Which model: RTMW-l-m at 256×192.** We did not pick this for speed. We ran an experiment (C2) to
find out *which extractor the authors actually used*, by comparing their released keypoints against
our own runs. The answer was a 256×192 RTMW/RTMPose model (the archive folder is literally named
`pose-rtmpose-192`). RTMPose-x at 384×288 became the "oversized" comparison row rather than the
baseline.

Then the lucky part: at n=30 clips, **RTMW beats RTMPose-x by 7.13 BLEU-4 / 7.73 ROUGE-L**, sign
established. The cheap choice was also the accurate one, which is not something we were entitled to
assume.

**Which precision: FP16.** Measured on the board at 15 W:

| | TRT ms | total ms | energy |
|---|---|---|---|
| RTMW-l-m FP32 | 25.6 | 33.0 | 275 mJ/frame |
| **RTMW-l-m FP16** | **13.8** | **19.7** (51 fps compute) | **127 mJ/frame** |
| RTMPose-x FP32 | 66.7 | — | 733 mJ/frame |

Accuracy cost of FP16: **−0.13 BLEU-4, CI [−1.48, +1.21]** — free within noise. And 99.81 % of the
keypoints Uni-Sign actually consumes land within 5 px of FP32.

FP16 later turned out to be **not just an efficiency choice but a fitting requirement**: the same
end-to-end run with the FP32 engine (159 MB vs 68 MB) runs out of memory partway through LM decode.

**Three traps we hit in the pose stage, all worth remembering:**

1. **TensorRT's default TF32 silently breaks the correctness gate.** With TF32 on, the gate fails at
   1.55e-3. With `BuilderFlag.TF32` cleared, it passes at 7.2e-6 with all 2416 confident keypoints at
   0.0 px. `common/trt_runner.py` clears it.
2. **Plain FP16 on RTMPose-x is broken by one layer overflowing.** `/mlp/mlp.0/ReduceSum` peaks at
   142293 against the FP16 maximum of 65504. Keeping 7 of 465 layers in FP32 fixes it for under 3 %
   latency.
3. **The max-pixel-error gate cannot tell a rounding flip from an overflow.** Both tell you "FAIL".
   The metric that separates them is *agreement*: the fraction of consumed keypoints within 5 px,
   judged only where the FP32 score ≥ 0.3. Also, the ≤1 px and ≤2 px columns are useless for
   comparison because the underlying bin size is bbox-dependent (1.71–2.45 px across five clips),
   which swings the ≤2 px column by 13 points on geometry alone. **Only the ≤5 px column compares.**

One standing limitation: **hands are the floor in every configuration** (~99 % within 5 px, p99.9
~30 px) and are completely unchanged by the precision fix — so hand error is argmax instability in
the pose head, not a precision problem. We cannot fix it by choosing a number format.

### 3.3 The converter — the boring part that was quietly wrong

`common/pose_to_unisign.py` turns keypoints into what Uni-Sign wants: four groups (body, left hand,
right hand, face), root-normalised, confidence threshold 0.3, with `crop_scale`. It is **bit-exact**
against the authors' own `load_part_kp` / `crop_scale`. Only **69 of the 133 keypoints** are actually
consumed.

The bug worth knowing: our crop and theirs define coordinates in different frames, giving a **1.68×
normalisation mismatch**. We found it by noticing the disagreement correlated 0.968 with the predicted
ratio. `common/renorm_to_openasl.py` fixes it by exact arithmetic — body disagreement 0.1328 → 0.0055
— and with it applied, our best configuration (18.52 BLEU-4) is indistinguishable from the ceiling
(18.17).

### 3.4 Language model — mT5-base, shrunk

**Baseline discipline first.** The authors' released predictions carry a `sample: …, prediction: `
prefix. Scored raw they give 58.7 BLEU-4, which is nonsense; stripped, **22.64 / 42.83**. Our Colab
reproduction got 22.53 / 42.68 against a paper number of 22.67 / 42.77, and our own standalone eval
loop gets 23.16 / 43.17. Those four numbers bracket the noise floor, and **the released checkpoint is
the FP32 reference rung — nothing overwrites it.**

**Vocabulary pruning is the big LM win.** OpenASL only ever uses 26,078 of mT5's 250,112 tokens, so
we slice the embedding and the output head: **587.7 M → 243.6 M parameters, 1187 MB → 571 MB**, for
**−0.28 BLEU-4 [−0.63, +0.05]** — not an established loss.

A methodology catch inside that: the first token census included the **test** split, meaning the
vocabulary was selected using test data. Re-pruned over train+dev only, the leak turned out to be
worth **−0.03 BLEU-4 [−0.11, +0.01]** — a precise null. It changed no result, but it was a real
methodology error and it is recorded as one.

**What does *not* work on the LM side:**

- **mT5 in FP16 is broken.** Every step returns token 0 at logprob −10.169, which is exactly
  −ln(26078) — a uniform distribution. Not a quality loss, a dead model.
- **INT8 was dropped, and our first reason for dropping it was wrong.** The implementation is
  **W8A32** (int8 weights, fp32 activations and accumulation), not W8A16, so it does *not* inherit the
  fp16 failure — and it scores 22.79 BLEU-4, which is impossible under that failure mode. It works:
  293 MB, 4.05× smaller, −0.09 BLEU-4 [−0.34, +0.14]. **The valid reason to drop it is that the
  decoder is host-bound**, so INT8 buys memory and not latency, and the int8 runtime is actually
  **10× slower per step** (2139 vs 197 ms) because 217 layers dequantize on every token.
- **TensorRT for the decoder does not pay off.** FP32 engines give identical tokens at 19 ms/token vs
  55 for PyTorch (2.9× on paper), but the decoder is bound by the host-side step loop, so it is not
  faster end to end. FP16 engines overflow to uniform logits; BF16 runs but diverges at step 3.

### 3.5 Decoding — how wide to search

| setting | BLEU-4 delta vs beam 4 | note |
|---|---|---|
| beam 4 | reference | |
| beam 2 | −0.81 [−1.29, −0.39] | 1.7× faster |
| greedy | −2.00 [−2.63, −1.41] | 2.7× faster |

`max_new_tokens`: a cap of 64 is **identical** to 100; 48 costs −0.14 [−0.30, −0.03]. So **we use 64**
— free, and it bounds the worst case.

---

## 4. The one decision everything points at: frame rate

This is where the project's actual recommendation comes from, so it is worth following the chain.

**Energy, measured on the board.** Pose energy per second of video: **4.25 / 3.36 / 2.27 / 1.67 /
1.17 J/s** at 30 / 24 / 16 / 12 / 8 fps — that is **1.00 / 0.79 / 0.53 / 0.39 / 0.27×**. Energy per
*frame* stays flat, which is the point: you save by processing fewer frames, not cheaper ones.

**Accuracy, measured offline at n=976.** Un-adapted: **24 fps is free** (test −0.07, dev +0.26);
**16 fps costs −1.1 to −1.3 BLEU-4 and −1.3 to −1.6 ROUGE-L**; 12 fps −2.5; 8 fps −8.0.

**Why the saving lands on the pose side, not the LM.** Shortening the frame sequence shortens the LM's
*encoder*, which takes 39–50 ms, while the *decoder* takes 1062–1538 ms. Cutting the sequence from
256 to 68 saves the LM only 12 %. So frame rate is a pose lever that the LM barely notices.

**Putting those together gives the headline comparison.** Per BLEU-4 point sacrificed, frame rate is
**10–100× more efficient than beam width** at the system level (30→24 fps costs 103.5 J per BLEU-4
point, against 1.0 for beam 4→2) — precisely because the LM is only ~25 % of system energy.

**So the operating point is beam 4 @ 24 fps**, not 16 fps:

| cell | system J | BLEU-4 | vs reference |
|---|---:|---:|---|
| beam 4 @ source | 43.4 | 22.87 | reference |
| **beam 4 @ 24 fps** | **36.1** | **22.80** | **−17 % energy for −0.07, CI [−0.53, +0.39]** |
| beam 4 @ 16 fps | 27.0 | 21.54 | −41 % energy for −1.33, CI [−2.00, −0.64] |

24 fps is free within noise. **16 fps is an established loss** — its CI excludes zero.

### Two mistakes that shaped this, both mine to own

1. **"Not established" is not "no cost".** My own n=30 curve put 16 fps at Δ −0.15 with a CI of
   [−3.20, +2.89] and I wrote it up as free. That interval is so wide it is compatible with a −3 BLEU
   disaster. At n=976 the LM track measured the real cost. **A wide interval is ignorance, not good
   news.**
2. **OpenASL is not one frame rate.** Both tracks hardcoded ~29.97 fps. In our 931 clips, 76.5 % are
   ~30 fps, **22.2 % are ~24 fps**, and 7 are at 59.94. That single constant invalidated the first
   frame-rate curve on *both* tracks. The source rate is now always derived per clip as
   `n_frames / duration_s`, and frame selection is `round(duration × target)` — verified identical
   between our `keep_idx` and their `fps_ratio_for_clip`, so one rate means the same frames on both
   tracks.

---

## 5. Running it all at once on the board

**M1 (2026-09-28) — the first real end-to-end translation.** One clip, 255 frames = 8.51 s of video,
beam 4, source rate, pose engine and language model in one process:

- total **9052.8 ms ± 335.6** = pose 6640.7 + convert 25.7 + LM 2386.5
- **50.37 J/sentence** total, 14.68 J dynamic, peak GPU **2.577 GB**, LM load 64.4 s (outside the
  timing window)

**It runs at 1.06× slower than real time, and the LM is the reason.** Pose alone is 0.78× real time —
comfortably live. The LM's 2.39 s per sentence pushes the total over. Because LM cost is per sentence
and pose is per frame, **long sentences get better and short ones get worse.**

**The 15 W contention I predicted did not appear — and that is not good news.** The stages are
sequential: all frames, *then* the LM. Having the LM merely resident costs nothing. A streaming
pipeline that decodes sentence N while extracting frames for N+1 **would** contend, and that is
unmeasured. M1 is not evidence that streaming is free.

**The 2×2 grid (2026-10-02)** varied frame count and decoder width independently on one clip, so the
end-to-end cost model could be attributed rather than guessed:

| cell | frames | beams | total ms | pose ms | LM ms | J/sentence |
|---|---|---|---|---|---|---|
| source, beam 4 | 255 | 4 | *failed, see below* | | | |
| source, greedy | 255 | 1 | 7998.0 ± 201.7 | 6260.7 | 1712.6 | 43.11 |
| 16 fps, beam 4 | 136 | 4 | 5616.1 ± 393.8 | 3414.3 | 2184.2 | 32.75 |
| 16 fps, greedy | 136 | 1 | 4705.7 ± 121.4 | 3336.7 | 1355.6 | 25.80 |

What it established:

- **The stages are exactly additive** within a run (pose + convert + LM = total to 0.04 ms), so the
  ~6 % residual we were chasing is a cross-run artefact, not an accounting gap.
- **Pose is invariant to decoder width** (+2.3 %, inside the run-to-run spread) and **linear in frame
  count** at ~23.7 ms/frame. Both are independence the pipeline is supposed to have — a consistency
  check that passed.
- **Frame rate is the big knob, decoder width the small one**, on the board and not just in the model:
  source → 16 fps at greedy cuts latency 41.2 % and energy 40.2 %; greedy → beam 4 at 16 fps costs
  19.3 % latency and 26.9 % energy, almost all of it inside the LM (+61.1 %).
- **`mJ_per_frame` is the wrong denominator for the whole system**, and this run proves it: it *rises*
  under subsampling, 169.07 → 189.70, because the per-sentence LM cost is spread over fewer frames.
  **End-to-end energy must be quoted per sentence.** The per-frame basis is still right for the pose
  stage alone, where it was defined.

**The missing cell is a reproducibility problem, not a limit.** `source × beam 4` died inside beam
search with 5691 MB free on a verified-empty board — but **M1 ran that identical cell successfully at
4625 MB free.** Same cell, more headroom, opposite outcome. The useful conclusion is not "it doesn't
fit" but **"MemFree is not a sufficient readiness check"**. Leading suspicion is fragmentation of the
unified pool, since the tensors beam search adds are tens of MB and not GB. A retry with
`expandable_segments` is queued, scoped to that one cell so it cannot change the allocator under the
cells already measured.

---

## 6. Hardware facts that keep catching us out

- **Memory is unified.** CPU and GPU share one 8 GB pool. There is no separate VRAM to run out of,
  and a GPU allocation failure often surfaces as the bizarre
  `RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at CUDACachingAllocator.cpp:1017`. **Read
  that as "out of memory".**
- **Loading the full checkpoint needs ~4 GB of MemFree** (2.577 GB device peak); the pruned checkpoint
  needs ~1.5 GB (0.98 GB peak). Both measured. `jetson/drop_file_cache.py` must reclaim to the
  **target**, not to the deficit.
- **15 W mode caps the GPU at 612 MHz on this board.** The earlier 44.4 ms / 11.6 W pose numbers match
  the 25 W cap (918 MHz) exactly — a board-and-mode difference, not a code difference. Clocks are
  logged, never pinned, and there is no sudo on this board ever.
- **CPU stages slow down by ×1.281 when the GPU takes more of the 15 W budget.** The power budget is
  shared, so "CPU time" is not a constant.
- **Two energy bases and two speed metrics, always say which.** Total mJ/frame includes the ~3.6–4.0 W
  idle draw; dynamic excludes it. `fps_compute` = 1000/total_ms; `fps_end_to_end` includes JPEG
  decode. The ~5.3 ms JPEG decode is a dataset artefact, not something a camera pipeline would pay.

---

## 7. How we decide whether a number means anything

These rules were all bought with a mistake, which is why they are written down.

- **Paired bootstrap, one set of draws for every run.** Score every configuration on the *same*
  resample draws, so deltas and differences-of-deltas are mutually comparable. Subtracting two
  independently-bootstrapped CIs is both wrong and less sensitive.
- **Effects under ~1 BLEU-4 at n≈1000 must be carried by ROUGE-L.** BLEU-4 simply cannot resolve them.
  The frame-rate result is the demonstration: ROUGE-L −1.34 [−2.42, −0.30] against BLEU-4 −0.35
  [−1.00, +0.45] on the same clips and the same draws.
- **n=300 can flip the sign of a known effect**, and did. All adaptation work uses the full 967-clip
  dev split.
- **Run the no-shift control before any adaptation claim.** Our first adaptation recipe looked like it
  hurt 16 fps; the control with *no* distribution shift degraded by 1.04 BLEU-4 too, so the number was
  never about adaptation — it was about a broken recipe. Cause: `--label-smoothing 0.2` over a 26 K
  vocabulary parks the reported loss at a ~3.35 smoothing floor that looks like a training curve. At
  0.0 the true cross-entropy is ~0.36, and at lr 1e-5 fine-tuning *improves* both metrics.
- **Seed spread is not a confidence interval.** It bundles seed effects with GPU non-determinism,
  which is real here: three runs at the *identical* seed gave final losses 0.3765 / 0.3742 / 0.3788.
- **Check and record board occupancy on every run.** An unrecorded co-tenant is what invalidated the
  09-29 memory probes, and that bad result propagated across tracks before it was caught.
- **Batch size changes the output.** `eval_openasl.py` batches 8 clips and pads; the board runs one.
  At `--batch-size 1` the eval reproduces M1's sentence exactly, and 2 of 30 clips differ between
  batch 8 and batch 1. Every offline BLEU number carries a small batching dependency — smaller than
  anything we reason about, but stated rather than discovered later.

---

## 8. What is still open

**In flight right now:** the 24 fps × beam 4 board cell (the LM track's ask — the frontier's
recommended point had never been run on the board), plus the `expandable_segments` retry of
`source × beam 4`.

**The biggest genuine gap:** **we have never produced a board-measured BLEU.** Every accuracy number
in the project is offline on the Mac. The only bridge between board and offline is the 30-clip
batch-size-1 agreement check. A ~100-clip end-to-end board pass at the recommended cell would close
it, for roughly 15 minutes of board time.

**Also open:**

- **Latency across the crop range.** All pose energy comes from one clip whose crop sits at the 83rd
  percentile of crop area, and the range is 9k–770k px — a 27× spread, with ~11 of 25 ms/frame
  scaling with area. Absolute joules are therefore clip-specific until we have ≥3 clips spanning it.
- **The 931-clip full-split pass.** Clips are on the Mac (6.6 GB); the board has 17 GB free at 85 %
  full, so this needs batching — roughly 100 clips, run, pull, delete, ten times. **Needs your call
  before any data moves**, under the shared-board rule.
- **Adaptation is not yet quotable.** The difference-in-differences is blocked on one eval JSON, and
  adaptation at source rate sits at ROUGE-L +0.60 with P(Δ<0) = 0.044–0.056 across seeds — straddling
  the threshold, so it must not be quoted as either established or null.
- **Not blocked, not started:** INT8 PTQ on the pose model (P5), the pose variant sweep (P6), moving
  preprocessing to the GPU (P11), the sentence-segmentation heuristic for the live demo (P12), the iOS
  app (P14), and the final report (C11).
- **A known trap left in the code:** `openasl_pose_fetch.py` does not compare downloaded sizes against
  `ZipInfo.file_size`, so a truncated file passes silently.
