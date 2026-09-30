# LM-track reply to the pose-track handoff — 2026-09-28

**Who this is for.** The pose track, in reply to `HANDOFF-LM-TRACK-2026-09-26.md`. Same conventions as
that document: every number names its metric and its denominator, inferences are labelled as
inferences, and retractions stay in place rather than being deleted.

**Read §0 first.** It corrects the premise under question 1, and question 1 is the one you said was
blocking a session of engine-building work.

---

## 0. Correction: our INT8 does not use FP16 activations

Your §1 reads "W8A16 keeps activations in FP16, and FP16 is already known-broken on this model," and
concludes the INT8 plan is dead. **The premise is wrong, and the naming is our fault.**

From `unisign/quant.py`, `W8Linear.__init__(..., compute_dtype=torch.float32)`; the forward casts both
the int8 weight and its scale *up* to `compute_dtype` before the matmul. Activations, matmul and
accumulation are all FP32. The only FP16 object anywhere in the scheme is the **per-row scale in
storage**, which is widened before it is ever used.

The empirical argument is stronger than the code reading: the scheme scores **22.79 BLEU-4 on 976
clips**. Your documented FP16 failure returns token 0 at every step with logprob −10.169 = −ln(26078),
a uniform distribution over the pruned vocabulary. That failure mode cannot produce a working score.

So what we built is exactly the variant your §1 lists as credible — **INT8 weights + FP32 activations**.
It does not need replacing, and no FP16 engine work is implied by it.

**Action taken.** The label `W8A16` has been renamed to `W8A32` across all docs and code (17
occurrences in 8 files), and `unisign/quant.py`, `RESULTS.md` §L9.1 and `ABLATION-REPORT.md` §2.B now
each carry an explicit precision contract stating that no activation is ever FP16 and why that matters.
The "A16" had referred to scale storage. It cost you a wrong conclusion; it won't recur.

**This does not rescue INT8.** See question 1 — the answer is still "drop it," for your *other* reason.

---

## 1. Is "pruned + INT8, beam 4" still the converged config?

**Split the question three ways. Two of the three change.**

**Pruning — keep.** Your CI restatement is right and we've adopted it: −0.28 [−0.66, +0.07] is **not
established as a loss** (our own run agrees: −0.28 [−0.63, +0.05]). It is what gets the model to 1.03 GB
torch peak-allocated, and on a shared 8 GB board with pose resident that headroom is the point. Free.

**INT8 — drop it from the deployment config and from the frontier plot.** Not for the FP16 reason, but
for your host-bound one, which we can corroborate independently: our own `--w8-runtime int8` mode (int8
resident, dequantised per forward) is **10× slower on the decoder** — 2139 vs 197 ms, because 217 layers
dequantise per token. Two unrelated implementations, neither buys latency. Combined with your finding
that the FP16 decoder step was no faster than FP32 (18 vs 19 ms/token) while the encoder was 3–4×
faster, the conclusion is the same from both sides: **the decoder step is host-bound, so quantising
weights cannot help it.**

INT8's honest claim is therefore **571 → 293 MB on disk for −0.09 BLEU-4 [−0.34, +0.14], P = 0.77** — a
memory result, reported as such. It does not belong on an accuracy-versus-energy axis, because it moves
neither axis. **Do not build INT8 engines.**

**Beam 4 — becomes an axis rather than a fixed choice.** See question 2.

**Converged config going forward: pruned, FP32, with decoding strategy as a measured axis.**

---

## 2. Is beam on TensorRT worth implementing?

**No.** Two reasons, one of which is a correction to the framing.

**The framing correction:** beam is not unavailable on the board, only on the TRT path. The PyTorch path
takes `--num-beams` and runs on the board at your measured 55 ms/token for the pruned model. So the
choice is not "beam or the board," it is "beam or the TRT decoder."

**The substantive reason:** we already have the price of giving up beam, with CIs, on all 976 clips.
These are the numbers your §4 said were still in flight — they are done (`RESULTS.md` §L6.1, pruned
model, Mac CPU, max_new_tokens 100):

| Decoding | BLEU-4 | paired Δ vs beam 4 [95 % CI] | eval wall |
|---|---:|---|---|
| beam 4 | 22.87 | — | 919 s |
| beam 2 | 22.06 | **−0.81 [−1.29, −0.39]** | 529 s (1.7× faster) |
| greedy | 20.88 | **−2.00 [−2.63, −1.41]** | 342 s (2.7× faster) |

Greedy is a real two-point loss, not a rounding error. But that is exactly what makes **PyTorch beam 4**
and **TRT greedy** the two natural endpoints of the frontier plot: one maximises accuracy, the other
minimises energy, and the gap between them is measured rather than assumed. Implementing per-beam cache
reordering would buy an interior point that beam 2 on PyTorch already approximates for free.

Spend the session on the step loop instead — CUDA graphs and a device-resident KV cache, which your
§1 already identifies as the actual lever and which helps *every* configuration rather than one.

---

## 3. What should the frontier plot's LM axis be?

**Proposal: decoder strategy × frame rate, all at pruned FP32.** Those are the two levers that actually
move both axes; everything else we tested moves one axis or neither.

The accuracy side is already measured on 976 clips with CIs, and per your §5 it transfers to the board:

| Frame rate | Δ BLEU-4 vs source rate [95 % CI] | frames processed |
|---|---|---|
| source (30/24, unthinned) | — | 100 % |
| **24 fps** | **+0.02 [−0.38, +0.45]** — free | ~80 % |
| 16 fps | −1.12 [−1.71, −0.48] | ~53 % |
| 12 fps | −2.52 [−3.22, −1.85] | ~40 % |

We also checked whether these two levers interact, since all the frame-rate deltas above were measured
on the quantised model: repeating 16 fps without INT8 and differencing on identical bootstrap resamples
gives an interaction of **+0.21 [−0.14, +0.57]** — no detectable interaction, so the frame-rate and
quantisation deltas can be quoted separately (`unisign/did_ci.py`, `RESULTS.md` §L7.3).

**What we need from the board to finish the plot** — joules per sentence and ms/token, pruned FP32,
each run reporting `cpu0_MHz` per your §3:

1. **Decoder axis at source rate:** greedy, beam 2, beam 4. Three runs. *This is the blocking item.*
2. **Frame-rate axis at beam 4:** 24 fps and 16 fps. Two runs.
3. Optional interior point: greedy at 24 fps, if 1 and 2 suggest the frontier bends there.

Five runs gives a frontier with both levers separated. You offered to wrap the LM scripts externally
with `common/power_logger.py:run_with_power` — yes please, that unblocks it without any new LM artifact.

**Our recommended default operating point is 24 fps**, since it costs nothing in accuracy and cuts
frames by ~20 %.

### 3b. Update: the accuracy half of this plot is now finished

Since writing the above we have measured the **full 3x3 grid** so you are not waiting on us for
anything. Pruned FP32, 976 test clips, cap 64, paired bootstrap with one resample draw scoring every
cell (`unisign/grid_table.py`, `RESULTS.md` L13):

| decoder | source | 24 fps | 16 fps |
|---|---:|---:|---:|
| **beam 4** | **22.87** | **22.80** | 21.54 |
| beam 2 | 22.06 | 21.93 | 21.19 |
| greedy | 20.88 | 20.66 | 19.83 |

Three things follow that change what is worth measuring on the board:

1. **24 fps is free at every decoder setting**, not just beam 4 (−0.07 / −0.13 / −0.22, all inside the
   0.6 noise band). So frame rate and decoder can be chosen independently, and you do not need a
   9-cell board sweep — the five runs in Appendix A are sufficient to span it.
2. **beam 4 @ 24 fps is statistically tied with the source-rate reference** (22.80 vs 22.87) at ~80 % of
   the frames. On accuracy grounds this is the operating point; whether energy agrees is your half.
3. **The levers are additive within noise.** Adding beam 4's frame-rate cost (−1.33) to the source-rate
   greedy cost (−1.99) predicts 19.55 for greedy @ 16 fps; measured is 19.83, +0.28 better than
   additive. Same conclusion as the INT8 x frame-rate check: no interaction we can detect. **You can
   therefore measure the two axes separately and combine them**, which is what Appendix A assumes.

What we still cannot answer without you: greedy is 2.7x faster and costs ~2 BLEU-4; 16 fps cuts ~47 % of
frames and costs 1.33. **Which of those buys more joules per BLEU-4 is entirely an energy question**, and
the accuracy grid above cannot decide it. That is the whole reason Appendix A's five runs are the
blocking item.

---

## 4. Re-run evals to record clip names?

**Not needed for what's already published; yes for anything new.**

Our `bootstrap_ci.py` asserts `A["refs"] == B["refs"]` and aborts on mismatch, so the positional
alignment you were worried about would already have raised rather than silently shifting sentences. The
existing comparisons are safe.

But name-based alignment is strictly better and costs nothing, so everything new will record names. We
will adopt your extended `bootstrap_ci.py` as the single entry point once you push — and the
difference-in-differences logic in our `unisign/did_ci.py` should fold into it rather than sit beside it
as a second script. We'll do that merge; just push first.

---

## 5. Something your §1 and §3 imply together that neither document states

Your §3 measures that GPU load downclocks the CPU (`cpu0_MHz` ×1.165) and slows two CPU-only stages by
an identical ×1.281. Your §1 concludes the LM decoder step is **host-bound**, i.e. CPU-bound.

Put those together: **reducing the pose frame rate should make the language model faster**, by giving
the CPU its clock back. The two stages are not merely sharing a budget, they are trading a resource the
decoder is bottlenecked on.

If that holds, the 24 fps row is **doubly free** — no accuracy cost (measured, above), less pose energy,
*and* a faster LM in the fused pipeline. That is a genuine cross-stage result and a better story for the
report than either track's numbers alone.

**This is an inference from your two findings, not a measurement.** It is testable with the M1 fused
pipeline at 30 / 24 / 16 fps, logging `cpu0_MHz` next to LM ms/token. Worth one run.

---

## 6. Accepted from your side, and what it changes for us

- **Your §5 settles a question we had been arguing rather than measuring.** Atisri's objection was that
  Mac-at-25 W and Jetson-at-15 W results could not be compared. Your 16.23 on the board versus 16.23 on
  the Mac, same clips, two decimals, settles it for BLEU. Our Mac evals transfer; we will keep scoring
  on the Mac and stop caveating it.
- **Your pose front-end gap resets our priority order, against our own interests.** −6.33 BLEU-4
  [+0.48, +13.11] on your poses versus the authors'. Everything the LM track has ablated spans about
  1.5 BLEU-4 in total. Ranked by size, the accuracy problems are now: **pose front-end (≤13) ≫ greedy
  decoding (2.00) > 16 fps un-adapted (1.12)**. Our planned 16 fps Colab adaptation targets the
  *smallest* of the three, so we are holding it until the energy rows in §3 justify it.
- **Headline labelling.** Our 22.79 is measured on the authors' released poses. Until the gap above is
  bounded, it is an upper bound for the deployed system and every report table that quotes it will say
  so explicitly.

## 7. Retracted on our side

- ~~"L7.1: capping encoder input length emulates a lower frame rate"~~ **RETRACTED.** A length cap only
  shortens clips already longer than the cap, so it hits long clips hard and leaves short ones untouched,
  whereas a slower camera thins every clip. OpenASL also has no single source rate — measured over 400
  test clips, 73 % are 30 fps, 21 % are 24 fps, the rest 25/31/32/60. Superseded by L7.2, which derives
  each clip's own rate from its filename timestamps. The 16 fps cost was corrected **1.61 → 1.12**.
- ~~"W8A16"~~ **RENAMED**, see §0. The activations were always FP32; the label said otherwise.
- ~~Presenting the stacked ablation deltas as if they independently added.~~ **RETRACTED.** They
  telescope by construction, because each was measured against the configuration immediately before it.
  That arithmetic is guaranteed and is not evidence of independence (`ABLATION-REPORT.md` §3.1). The
  16 fps × INT8 cell in §3 above is the one interaction we have since tested directly.

## 8. Logistics — a conflict is coming

As of this writing `origin/main` has no commits from the pose track, so your §6 changes are still local
to your machine. Two files will conflict when you push:

- **`unisign/eval_openasl.py`** — you added `"names"` recording; our commit `35edde8` changed the same
  file for per-clip frame-rate emulation (the `fps_ratio_for_clip` call site).
- **`unisign/bootstrap_ci.py`** — you extended it; we added a sibling `unisign/did_ci.py` that duplicates
  part of its resampling.

**Push first and we will merge onto yours**, rather than the other way round — your changes to those two
files are the more invasive ones. We will hold off touching either file until you do.

---

## Appendix A — the five board runs, copy-paste

**Dependency: pull first.** These need `unisign/unisign_infer.py` at or after the commit that adds
`--fps`, which did not exist when you wrote the handoff. Without it there is no way to run the board at
16 fps under the *same* definition our BLEU rows use, and an energy row measured under a different
definition can't be paired with an accuracy row.

`--fps` derives each clip's own source rate from its filename timestamps and keeps
`round(duration x target_fps)` frames — identical to `eval_openasl.py --fps`. Verified on the Bitcoin
clip (300 frames, 9.967 s, so 30.1 fps source): 24 fps -> 239 frames, 16 fps -> 159 frames.

Run `jetson/run.sh whoelse` first — per your own §6, another tenant invalidates any timing or power row.

### Phase A — five runs, one clip, ~10 minutes

All at pruned FP32. Only two things vary: `--num-beams` and `--fps`.

```bash
CLIP=data/openasl_ref_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl
COMMON="--pkl $CLIP --ckpt weights/openasl_pose_only_slt_pruned.pth --mt5 weights/mt5-base \
        --device cuda --dtype fp32 --max-new-tokens 64 --repeat 5 --idle-seconds 5"

# 1-3: decoder axis at source frame rate
for B in 1 2 4; do
  python3 -m unisign.unisign_infer $COMMON --num-beams $B \
    --power-json results/lm_energy_beam${B}_srcfps.json \
    --out        results/lm_beam${B}_srcfps.json
done

# 4-5: frame-rate axis at beam 4
for F in 24 16; do
  python3 -m unisign.unisign_infer $COMMON --num-beams 4 --fps $F \
    --power-json results/lm_energy_beam4_fps${F}.json \
    --out        results/lm_beam4_fps${F}.json
done
```

**Send back:** the ten JSONs, plus the printed `[power]` and `[infer]` lines. The numbers that matter are
`J_per_sentence`, `dynamic_J_per_sentence` (idle-baseline subtracted), `mJ_per_token`, `avg_watts`, and
`aux_avg.cpu0_MHz` — the last one per your §3, so we can see whether the LM is being downclocked while
it runs.

### Phase B — optional, only if Phase A looks sane

One clip gives a repeatable number, not a representative one. If the shape of Phase A is worth
publishing, repeat the three source-rate runs over the 30-clip set you now have and report the mean
J/sentence:

```bash
for P in data/openasl_pose/*.pkl; do
  N=$(basename "$P" .pkl)
  python3 -m unisign.unisign_infer --pkl "$P" \
    --ckpt weights/openasl_pose_only_slt_pruned.pth --mt5 weights/mt5-base \
    --device cuda --dtype fp32 --num-beams 4 --max-new-tokens 64 --repeat 1 --idle-seconds 0 \
    --power-json "results/lm_energy_30clip_${N}.json" --out /dev/null
done
```

(`--idle-seconds 0` because one shared idle baseline is enough across a batch; take it once at the
start with a single `--idle-seconds 10` run.)

### What we expect to see, stated in advance so the run can falsify it

From the Mac (relative only — Apple silicon CPU, not the board): at source rate the per-sentence split
is GCN 71 ms + encoder 65 ms + decoder 190 ms. At 16 fps it becomes 54 + 49 + 188.

**The GCN and encoder shrink with frame rate; the decoder does not** — it is per-token, and the token
count barely moves. That is the same shape as your host-bound finding, reached from the other side.

Two consequences if it holds on the board:

1. **Frame rate is a weaker LM lever than it looks** — it cuts the ~40 % of LM time that scales with
   frames, not the 60 % that scales with tokens. Its real value is still in the pose stage.
2. **Beam width is the stronger LM lever**, because it multiplies the part that dominates. Which is why
   the decoder axis is runs 1-3 and gets measured first.

If the board disagrees with either, that is a finding, not an error — say so and we will re-plan.
