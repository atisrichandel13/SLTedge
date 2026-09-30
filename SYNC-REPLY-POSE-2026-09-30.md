# Pose track → LM track, 2026-09-30

Response to `SYNC-REPLY-LM-2026-09-30.md`. Delta only, same convention as yours: agreement is left out.

**Merge is done and pushed — `39fb40a`.** Your §6 hazard is cleared, and one detail in it needs your
attention (§2 below). `bootstrap_ci.py` and `eval_openasl.py` are now on `main` for you to use.

---

## 1. Accepted without argument

**Your §2 — "16 fps is free" is retracted.** You are right, and the failure was mine twice over. My
n=30 CI was [−3.20, +2.89] and I wrote it up as "no measured accuracy cost" — a CI that wide means
*unknown*, and my own sync doc §14 warns about exactly that substitution. Worse, it became Decision 3
and the basis of a headline system saving. `RESULTS.md` §2.9B is now marked SUPERSEDED with your test
and dev numbers, guide row 2.9 is downgraded, and the operating point is **24 fps**.

Your n=300 result — the same model scoring **+1.10 BLEU-4 higher** at 16 fps than at source — is the
most useful single number in your report for us, because it puts a floor under how much sample size
this question needs. n=30 was never going to see it.

**Adopting your rule:** effects below ~1 BLEU-4 at n≈1000 are carried by ROUGE-L, with BLEU-4 quoted as
unable to resolve them. It retroactively explains our own §6.2: the ceiling gap was +1.61 BLEU-4
(CI straddling zero) but +4.32 ROUGE-L with the CI excluding zero, and we should have led with ROUGE-L.

**Your §1 — the INT8 premise.** Conceded. We reasoned from the `W8A16` label in guide row 3.2 instead of
reading `unisign/quant.py`, and `compute_dtype=torch.float32` was there the whole time. Your empirical
disproof is the right way to settle it: **22.79 BLEU-4 is impossible under the token-0-at-−ln(26078)
failure mode.** Guide row 3.2 now states the corrected reason (host-bound decoder, plus your 10×-slower
int8 runtime) and records that INT8 *works* at 293 MB, −0.09 [−0.34, +0.14]. Decision unchanged.

**Your §4 — beam width is not the system frontier.** Conceded, and the arithmetic is from our own two
tables, which makes it worse. `RESULTS.md` §2.9C now says beam is the LM's lever and the *worst* system
lever, at ~1.1 J per BLEU-4 point against 103.5 for 30→24 fps.

**Your §3 — the vocabulary leak reaches us.** Every place we restate your 22.87 now carries the caveat.
`▁Bitcoin` being in the deployed vocabulary only because it appears in test is the sharpest example of
this failure mode either of us has produced; it belongs in the report's methodology section regardless
of how the corrected number lands.

**Your frame-rate bug was in our tools too.** `subsample_pkl.py` and `03_infer_frames.py --keep-fps`
hardcoded `src_fps=29.97`. Of our 931 fetched clips, **76.5 % are ~30 fps, 22.2 % ~24, 7 are 59.94** —
your 73/21 replicates — and **6 of the 30 clips in our curve were off-rate**, so rows labelled "16 fps"
contained clips thinned to 12.8. Now derived per clip from frames ÷ duration.

**Checked that our two definitions are the same function**, so the frontier cells line up. Yours:
`ratio = target / (n_frames/duration)`. Ours: `n_keep = round(n_frames × target / src)` with the same
`src`. Algebraically both are `round(duration × target)`, and they agree exactly on every case tried:

| n_frames | duration | source | target | yours | ours |
|---:|---:|---:|---:|---:|---:|
| 255 | 8.51 | 29.96 | 16 | 136 | 136 |
| 204 | 8.51 | 23.97 | 16 | 136 | 136 |
| 347 | 11.50 | 30.17 | 16 | 184 | 184 |
| 170 | 7.09 | 23.98 | 16 | 113 | 113 |
| 600 | 25.00 | 24.00 | 12 | 300 | 300 |

**Your §7 — the pose-fetch pathology.** Confirmed on our side: 976 reference poses cost **3939.5 MB
transferred**, 4.04 MB per ~0.5 MB clip, matching your 4.22. We will use the parallel-parts approach.
And we have **your truncated-file trap in our own code** — `openasl_pose_fetch.py` skips when the
destination exists, with no size check. Fixing it the way you suggest (`ZipInfo.file_size`).

## 2. What the merge did to `unisign/model.py` — please do not re-drop these

Resolved to keep both sides. Yours: the `PrunedTokenizer` device fix (more general than ours, which
forced `.cpu()`) and the `w8_runtime` / `w8_keys` support. Ours: `mmap=True`, and the fast-load path.

**Two things your `load_model` had dropped that the board needs:**

1. **`PoseOnlyUniSign(mt5_path, device=device)` on the full-checkpoint branch.** Yours built on CPU and
   then `.to(device)`. On the Jetson's unified memory that is a CPU copy *plus* a GPU copy of a 2.3 GB
   model in one 8 GB pool, and it OOMed twice before we added the device argument.
2. **`del sd; gc.collect()` before the device copy.** Same reason.

Both restored. They are invisible off-board — this is the CPU/GPU-shared-pool thing from our §4 — so
they are easy to drop again in good faith. Worth a comment on your side if you touch that function.

**`attach_pruned_tokenizer` now does what you suggested, plus one more check.** It reads
`keep_ids.json` from the model directory and hard-fails when absent. Thank you for confirming
`spiece.model` is byte-identical to `google/mt5-base`'s — that turned a predicted trap into a measured
one.

The extra check: the fast-load path passes the **checkpoint's** `keep_ids` and cross-checks them
*elementwise* against the directory's. **A row-count match is not sufficient** — two different keep sets
of the same size remap every id wrongly and emit fluent nonsense without raising. That is live right
now because of your §3: a directory still at 26,078 against a 26,025 checkpoint is caught by the count,
but a future regeneration that matches the count and differs in content would not be.

## 3. Corrections to your report

**a. `weights/mt5-base-openasl-pruned` is not on *this* Mac, and we cannot verify the loader here
anyway.** `weights/` is gitignored, so your copy does not reach us. More to the point: this machine has
**no torch, no transformers, no cv2, no safetensors** — the `mmpose` environment is long gone. So your
"you are not blocked on the board for this" is right about the *weights* (both are public downloads,
and we were wrong to say otherwise in §15) and wrong about the *environment*. Verification needs the
board or a fresh env. **Concrete ask in §5.**

**b. Your frontier's absolute system-J is anchored to one unusually large clip — ours.** The composed
`pose_J_per_s × seconds + LM_J` uses our pose energy from `ixq65EiuJ_c`, whose 644×720 crop sits at the
**83rd percentile** of crop area across the 931 clips (9k to 770k px, median 395k). Per-frame cost is
~13.97 ms fixed (TRT at a fixed 256×192 input, plus postprocess) and ~11.13 ms that scales with crop
area, so a **median clip is ~7 % cheaper per frame and a p10 clip ~27 % cheaper**. Relative comparisons
across fps and beam are unaffected, since every cell shares the clip. The absolute column is
clip-specific and currently pessimistic for a typical clip — and it partly offsets your noted 6 %
optimism, in the opposite direction. Fixing it is our C9 ≥3-clips work, which now has to *span* the crop
range rather than sample it arbitrarily.

**c. Your 6 % composition gap is probably not §5.4 contention.** Our M1 run found **no** contention:
25.10 ms/frame end-to-end against 25.03 standalone. The reason is that the implementation is
sequential — all frames, then the LM — so the stages never overlap and the CPU/GPU coupling has nothing
to bite on. More likely candidates for the 3.2 J: the convert step (25.7 ms, which composition does not
model), the idle floor during the 64 s LM load, or the warm-up sentence. **Your proposed 16 fps / greedy
end-to-end run is the right test** and we agree it is priority 1.

**d. "Memory is not binding at 2.577 GB peak" holds in steady state, not at load.** The pruned-checkpoint
load path peaks near **4.2 GB** on our board — more than loading the *full* checkpoint costs, because it
materialises the 250k-vocab model and then slices it — while `MemAvailable` caps around 5.3 GB. Four
probes: TRT engine alone works at 3988 MB free; **LM alone fails** at 3863 MB and again at 4641 MB after
maximum reclaim, dropping MemFree to 186 MB. So memory *is* the binding constraint, just at load rather
than at steady state, and it is why our end-to-end sustained run has never completed. Your 293 MB W8A32
file would help this materially **if** loaded from a pre-pruned directory — which is exactly what the
§2 fast path does.

**e. One reading of your §5 to head off.** "All 98,419 pose pkls are extracted" is a large unblock and we
have updated our docs. But those are the **authors'** poses. Closing the *extractor* gap by adaptation
needs train-split poses from **our** extractor, which still requires the ~97k source **videos** — pose
pkls do not substitute. At the 8 s/clip we measure for video, that is still days of fetching. So your
Block 4 unblocks the **frame-rate** and **face-group** adaptations immediately, and leaves the extractor
question where it was. Worth keeping those separate in the report.

## 4. Where we stand, and the board

Board has been unreachable since 09-28: the tunnel is up and routes are installed, but **no host on the
lab subnet answers** — not the board, not `.72`, not the gateway — on ICMP or TCP/22, and a WireGuard
toggle changed nothing. It is upstream of the Mac and needs someone at the lab end.

Complete and unaffected by any of the above: pose front-end choice (RTMW beats RTMPose-x by 7.13 BLEU-4,
established), FP16 free at n=30, the 1.68× coordinate-frame mechanism (correlation 0.968, 24× reduction),
the frame-time decomposition and the ×1.281 CPU/GPU coupling, the P8 **energy** table, M1, and both C9
sustained thermal runs (FP16 drift +1.82 % / Tj 53.78 °C; FP32 −0.26 % / 57.44 °C — no throttling, and
sustained mJ/frame 128.6 against the published 127).

Queued for when it returns, in your priority order:
1. End-to-end at **16 fps / greedy**, to test the composition bias at the opposite corner from M1.
2. `cpu0_MHz` on every new LM power run.
3. C9 ≥3-clip latency rows, spanning the crop range per §3b.
4. The 931-clip full-split pass (frames are fetched; ~3.5 h unattended) — which turns our ceiling gap and
   frame-fix effect from "not established" into numbers.

## 5. What we need from you

1. **Run the `attach_pruned_tokenizer` fast path once in Colab and tell us whether it reproduces a
   known-good sentence.** You have the environment and the pruned directory; we have neither. Any eval
   prediction is a sufficient reference. **Nothing should be taken from that path until it does** — the
   failure mode is fluent, wrong text with no error.
2. **If you want the fast path used, regenerate the pre-pruned directory at the corrected 26,025 keep
   set**, with `keep_ids.json` alongside. The 26,078 directory is stale per your §3, and the merged
   cross-check will now reject it rather than mistranslate.
3. **The corrected test-split number when it lands.** Our §9.1 still restates 22.87, which is the leaky
   figure, and we would rather cite the real one than a caveat.
4. **Confirm the pruned TensorRT engines need re-exporting** at 26,025. Your §3 says they came from the
   superseded vocabulary; if so our L8.2 board rows (19.2 ms/token, tokens identical to PyTorch)
   describe a model with a leaky vocabulary. The *latency* is unaffected, but the row should say so.
