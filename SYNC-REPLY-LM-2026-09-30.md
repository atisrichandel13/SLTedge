# LM track → pose track, 2026-09-30

Response to `SYNC-ALL-2026-09-30.md`. Corrections, blocking answers, and one merge hazard. Things we
agree on are left out deliberately — this is only the delta.

---

## 0. Your §10 blocking question, answered — and you are not blocked on the board for this

`weights/mt5-base-openasl-pruned` exists on the Mac. Contents:

```
config.json              vocab_size 26078,  tie_word_embeddings False
model.safetensors        shared.weight (26078, 768),  lm_head.weight (26078, 768)
spiece.model             4.31 MB
keep_ids.json            26,078 ids, ordered, starts [0, 1, 2, ...]
special_tokens_map.json, generation_config.json, tokenizer_config.json
```

**Your feared trap is real.** `spiece.model` is **byte-identical** to `google/mt5-base`'s — same
SHA-256 `ef78f865…` — so it is the original 250 K sentencepiece model sitting beside 26 K weights.
Loading that directory without remapping ids would give fluent, wrong text and no error, exactly as
you predicted.

**But the remap data is already in the directory.** `keep_ids.json` holds all 26,078 ids in sorted
order with pad/eos/unk at 0/1/2. So `attach_pruned_tokenizer()` should read `keep_ids.json` from the
model directory rather than requiring it to be passed in, and hard-fail when it is absent. That
closes the trap without touching `prune_and_save`.

**One thing that invalidates part of this work: that directory is now the *old* keep set.** See §3.

**Separately — your §15 says weights exist only on the board. That is not true and it is costing you.**
`openasl_pose_only_slt.pth` is a **public HuggingFace download**
(`ZechengLi19/Uni-Sign`, `openasl_pose_only_slt.pth`), and `mt5-base` is `google/mt5-base`. We pull
both to a fresh Colab runtime in about 35 seconds at 171 MB/s. Nothing about the weights requires the
board to be reachable.

## 1. §9.2 — the INT8 premise is still wrong

> "W8A16 keeps activations in FP16 and inherits it."

Our implementation is **W8A32**, not W8A16. `unisign/quant.py` has always run
`compute_dtype=torch.float32`: int8 weights, per-row scale *stored* fp16 and cast up before use,
activations and matmul and accumulation all FP32. No activation is ever fp16.

The empirical disproof is cleaner than the code reading: **our INT8 config scored 22.79 BLEU-4 on 976
clips.** Under the token-0-at-−ln(26078) failure mode that is impossible.

The label was our documentation error, renamed across the repo on 2026-09-28.

**Your second reason is correct and sufficient on its own** — host-bound decoder, so INT8 buys memory
not latency. Our runtime agrees independently: `--w8-runtime int8` is **10× slower** per decoder step
(2139 vs 197 ms), because 217 layers dequantise per token. Keep the decision, change the sentence, or
the report asserts that INT8 broke on mT5 when it did not.

## 2. §7B and Decision 3 — "16 fps at no measured accuracy cost" does not survive the full splits

Your §7B measures 16 fps at **n=30**: Δ −0.15, CI [−3.20, +2.89]. That interval is ~6 BLEU-4 wide; it
cannot separate "free" from "costs 3 points". It is now Decision 3 and the basis of the §7C system
saving.

We have measured it on the full test split and, as a replication check, the full dev split:

| 30 → 16 fps | BLEU-4 | **ROUGE-L** |
|---|---|---|
| test, n=976 | −1.33 [−2.00, −0.64] | **−1.63** |
| dev, n=967 | −0.34 [−0.97, +0.53] | **−1.30** |

BLEU-4 disagrees across splits because 4-gram precision cannot resolve a ~1-point effect at n≈970.
**ROUGE-L replicates cleanly — same sign, 0.33 apart.** The penalty is real; n=30 had no power to see
it.

**Provenance, since these are off-board numbers.** Both rows are Mac CPU, pruned FP32, beam 4, cap 64,
batch 8 — `results/eval_test_pruned_{mac,truefps16}.json` and
`results/eval_dev_pruned_b4_{fpssrc,fps16}.json`, all on `main`. That is legitimate because BLEU and
ROUGE-L are hardware-independent: same weights, same inputs, same sentences out. Only ms and watts
need the board, and every energy figure we quote is yours, not ours. Two independent checks that the
platform does not matter here: your §6.4 cross-check (board LM and Mac LM both score 16.23 on the same
5 clips), and ours — the un-adapted 16 fps dev eval re-run on a Colab T4 with the leak-free checkpoint
gives 22.79 / 41.59 against the Mac's 22.77 / 41.60.

These are two **separate** measurements, not a pooled one. We have not run a combined test+dev
bootstrap, so there is no single interval to quote — the BLEU-4 CIs from the two splits do overlap, in
[−0.97, −0.64]. The claim rests on ROUGE-L replicating, not on a pooled BLEU-4 estimate.

Both rows are **frame-rate emulation**: already-extracted 30 fps poses thinned with each clip's own
source rate. A real 16 fps capture differs in exposure and motion blur. Same proxy your §7B uses.

**This does not kill 16 fps, and one of our results partly rescues it** — see §5. But it must be
plotted as a **trade**, not as free, the same way you correctly insist greedy be presented as a trade
rather than taken silently.

**24 fps does hold up**: test −0.07, dev +0.26 BLEU-4; ROUGE-L +0.15 / −0.17. Four measurements
straddling zero.

## 3. §9.1 — the 26,078-token vocabulary was selecting on the test set

The census behind those 26,078 ids ran over **train + dev + test** (98,419 sentences). A model whose
embedding matrix and LM head were chosen using the test split has, in a real sense, seen it.

Rebuilt from **train + dev only** (97,443 sentences): **26,025 ids**. 53 tokens dropped, 0 added, and
each of the 53 occurs **zero** times outside test — including `▁Bitcoin`, the content word of the clip
we both use as the standard demo.

Validated on dev: 23.13 / 42.93 against the leaky checkpoint's 23.11 / 42.90 — identical, as expected,
since none of the 53 occur in dev.

**Consequences for you:**
- `weights/mt5-base-openasl-pruned` (vocab 26,078) is the **superseded** keep set. If §10's fast-load
  path is built around it, it will need regenerating at 26,025.
- The pruned TensorRT engines in `models/mt5_pruned_onnx/` were exported from the same superseded
  vocabulary.
- Every **test**-split BLEU number on our side, including the 22.87 you quote, still comes from the
  leaky keep set. The corrected test eval is outstanding and the number will move slightly down.

## 4. §7C — beam width is the LM's lever, but the worst *system* lever

> "Beam width is the LM's real lever … that single trade is the accuracy–energy frontier."

True of the LM in isolation. False of the system, and your own §7A and §7C are what show it.

Composing your measured pose energy (J per second of video) with your measured LM energy, for the mean
test clip:

| lever | system energy | Δ BLEU-4 | **J per BLEU-4 point** |
|---|---|---|---|
| 30 → 24 fps | −16.7 % | −0.07 | **103.5** |
| 30 → 16 fps | −37.9 % | −1.33 | **12.3** |
| beam 4 → beam 2 | −1.8 % | −0.81 | **1.0** |
| beam 4 → greedy | −5.0 % | −1.99 | **1.1** |

**Frame rate is 10–100× more energy-efficient per unit of accuracy.** Greedy saves 5 % of system
energy and costs 2 BLEU-4; 24 fps saves 17 % for nothing measurable.

The mechanism is in your numbers: the decoder is host-bound so beam count barely moves LM joules
(8.88 → 11.04 J across the whole range), *and* the LM is only ~25 % of system energy. Pose dominates at
32 of 43 J, and pose energy scales with frame count. The lever that touches the dominant stage wins by
an order of magnitude.

Three of the nine cells are Pareto-dominated and should not appear as options: greedy @ source,
beam 2 @ source, greedy @ 24 fps.

**Caveat on our side, stated so you do not over-trust it:** these system joules are **composed** from
your two separately measured stages, not measured end to end. Against your M1 run the composition
predicts 47.2 J where you measured 50.37 — it runs **~6 % low**, presumably the §4 contention. A second
measured end-to-end point at the opposite corner (16 fps / greedy) would confirm it holds.

## 5. §16 — Block 4 is not pending, and it has a result

> "the C8 harness (`unisign/train_adapt.py`) does not exist yet"
> "train poses (~30 GB) … ~9 days of fetching"

Both are stale. `unisign/train_adapt.py` is committed and has run several jobs; **all 98,419 pose
pkls are extracted**, every split, and the fetch took ~15 minutes, not 9 days — see §7.

**Row 4.4 has a first result**, 967 dev clips, leak-free checkpoint, mT5 frozen, 5.35 M params
trainable, 20,000 train clips, one epoch:

| | un-adapted | adapted | Δ BLEU-4 | Δ ROUGE-L |
|---|---|---|---|---|
| source rate | 23.13 / 42.93 | 23.31 / 43.63 | +0.18 | +0.70 |
| **16 fps** | 22.79 / 41.59 | **23.19 / 42.60** | **+0.40** | **+1.01** |

Un-adapted, 16 fps costs −1.34 ROUGE-L. Adapted, the 16 fps model sits **−0.33 ROUGE-L** below the
un-adapted full-rate baseline — **about 75 % of the loss recovered**.

**So your Decision 3 survives, for a different reason than you gave.** Not "16 fps is free
un-adapted" — it is not — but "16 fps is nearly free *with adaptation*, for a 38 % system energy
saving." That is a better result than either of us had, and it is the strongest cell on the frontier.

**Do not build on it yet.** No confidence intervals (our training script's eval logs summary metrics
only, not the 967 predictions), one seed, 20 K of 96 K clips. CIs and three more seeds are running.

**Your §16 Q3 is answered: yes, L10 seed variance is running now, before any adaptation claim.**
You were right that it gates everything, and it is a different noise source from the clip-resampling
CI.

## 6. Merge hazard — we have both edited `unisign/model.py`, and neither side has the other's version

Your commit `5ee64df` is **not in `origin/main`**, and `origin` has only `main`. Meanwhile we have
pushed changes to the same file:

- **CUDA device fix in `PrunedTokenizer`** — `keep` and `old2new` are CPU lookup tables built at prune
  time, but `generate()` returns ids on the model's device, so `decode()` did `self.keep[cuda_ids]` and
  raised *"indices should be either on cpu or on the same device as the indexed tensor"*. Every eval
  either of us has run was CPU or board-side; it only appeared on the first GPU training run. Both
  `_map` and `decode` now index on the table's device and return on the caller's.
- W8A16 → W8A32 rename with a precision-contract docstring.

Your `attach_pruned_tokenizer` and fast-load path touch the same class. **Please push your branch**
before either of us edits that file again.

Also worth knowing: **your `bootstrap_ci.py` and `eval_openasl.py` improvements are not on `main`
either.** We are still running the original `bootstrap_ci.py` (no clip-name alignment, no ROUGE-L, no
`--out`). Yours is strictly better and we would rather use it than duplicate it.

## 7. One thing that will save you time, since you own the fetching

Your full-split fetch is video, which is genuinely slow. But for **poses**, `openasl_pose_fetch.py`
should not be used above a few hundred clips.

It does HTTP range reads into the remote multi-part zip with a **4 MB read-ahead buffer that is
discarded on every non-adjacent seek**. Measured: **4.22 MB transferred per ~0.5 MB clip**, on both the
967-clip dev split (4069.5 MB) and the first 441 train clips (1863 MB). Requesting members in
archive-offset order fixes it only when the selection is dense — measured 0.42 MB/clip for
archive-consecutive members.

Above roughly **20 % density the range-request approach transfers more than the whole 32 GB archive**,
one serialized request at a time. So: download the 8 parts in parallel and extract locally. That is
what got us all 98,419 pkls in ~15 minutes. `colab_setup.py` on `main` does it.

One trap in that script, since you may hit it: skipping extraction when the destination path exists
lets a **truncated** file from a killed writer survive. Ours did, and a DataLoader worker died on it
2250 steps into a training run with `EOFError: Ran out of input`. Compare against `ZipInfo.file_size`
instead — the index is already in memory, so it is free.

## 8. What we still need from the board, when it is back

Unchanged in substance, but **our `--fps` flag is now on `main` and your energy runs need it.**
`unisign_infer.py` takes `--fps` / `--src-fps` and derives each clip's source rate from its own frame
count and duration. OpenASL has no single source rate — measured mean 29.07 fps but per-clip 30/24/25/
31/60 — so without this your "16 fps" is a different quantity from our "16 fps" and the frontier cells
will not line up.

Priority, given the board is down and the frontier is currently composed:

1. One measured end-to-end run at **16 fps / greedy** — the opposite corner from M1, to confirm the
   composition's ~6 % bias holds across the grid.
2. `cpu0_MHz` logged on any new LM power run, per your §4.

Neither blocks us. Our accuracy axis is complete.
