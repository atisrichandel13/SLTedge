# `colab_probe_scale.py` checked, plus interim Q1a/Q1b numbers

*Pose track → LM track, 2026-10-05, on `ec7371f`. One correction, which is practical rather than
scientific. Everything else I could check, I checked, and it holds.*

---

## 1. What I verified, and it is all correct

I re-derived each load-bearing claim rather than reading it:

| claim in the script | checked against | verdict |
|---|---|---|
| `--limit N` takes the first N, so rungs are nested | `unisign/train_adapt.py:74` — `available_names(..., shuffle=False)` returns `names[:limit]`; called at `:195` without a shuffle flag | **correct**, the ladder is a true scaling curve |
| L15.3's anchor is +1.04 ROUGE-L [+0.25, +1.80] | `results/adapt_ci_dev.json`, `deltas["16 fps: un-adapted -> adapted"]["ROUGE-L"]` = +1.044 [+0.247, +1.802] | **correct to the digit** |
| `CAP = 100` matches `adapt_ci_dev.json` | that file's `point` keys are `*_cap100`; and its `cap 64 -> cap 100` delta is exactly 0.00 on both metrics | **correct**, and the cap turns out not to matter |
| retraining 20,000 here keeps the curve in one environment | `colab_block4.py:38` uses the same `CKPT = /content/pruned_traindev.pth` | **correct** — same base, so a 20,000 rung that misses +1.04 really does indict the environment and not the checkpoint |
| `adapted_full.pth` is the right resume sentinel | `train_adapt.py:309` writes exactly that name at the end of training | **correct** |

The §"WHAT IT DOES NOT TELL US" paragraph is the right one to have written, and it is the thing I
would otherwise have had to say back. Frame-rate scale makes J9 *worth running*; it does not predict
J9's result. Agreed, and I will not quote the curve as if it did.

## 2. One correction: `adapted_full.pth` is the **full** model, so the ladder writes ~2.9 GB to Drive

The docstring says:

> Everything expensive here -- checkpoints and eval JSONs -- is small (train_adapt saves only the
> 5.35 M trainable params), so it goes to Drive

That is true of `last.pt` and false of `adapted_full.pth`. From `unisign/train_adapt.py:304-310`,
`full` is the entire `model.state_dict()`, and the file's own docstring at `:23-24` says so:

```
Outputs in --out-dir: last.pt (trainable weights + optimizer + scheduler + epoch, for --resume),
adapted_full.pth (full state dict + keep_ids, loadable by unisign.model.load_model / eval_openasl),
```

At `results/RESULTS.md:1849` that state dict is **570 MB**. `OUT` is on Drive and `train()` puts a
rung dir under it, so five rungs leave **~2.9 GB of `adapted_full.pth`** plus five `last.pt` on Drive
— on a 15 GB account that is a fifth of it, and 570 MB Drive writes are slow enough to be noticeable
next to a ~15-minute training run.

**Suggested fix, which also makes the resume cheaper rather than more expensive.** Right now `train(n)`
runs before `evaluate(...)`, so deleting the big file after the eval would make a resumed run retrain.
Checking the eval JSON first avoids that:

```python
for n in SIZES:
    ev_path = f"{E}/eval_dev_adapt_fps16_n{n}.json"
    if os.path.exists(ev_path):          # rung already finished; never reload the 570 MB model
        ev = ev_path
    else:
        d = train(n)
        ev = evaluate(f"adapt_fps16_n{n}", f"{d}/adapted_full.pth")
        os.remove(f"{d}/adapted_full.pth")   # last.pt + the L5.4 base reconstruct it (L15.3)
    ...
```

That keeps Drive at the eval JSONs, the CIs and five `last.pt`, which is the set that is actually
expensive to recreate. **Your call** — if Drive headroom is not tight, leaving it alone costs nothing
but space, and I would rather flag it than have the ladder die on rung four.

## 3. Q1a and Q1b — interim numbers at 449 of ~966 dev clips

The fetch is still running. These are **interim** and I will repost the finals; I am sending them now
because both are already decisive against the bars you set.

**Q1a, source height** (your bar: accept unless 720p yield falls below ~60 %):

| source height | dev (n=449, interim) | test (n=931) |
|---:|---:|---:|
| **720** | **345 — 76.8 %** | **710 — 76.3 %** |
| 470 | 12 | — |
| 480 | 5 | 45 combined with 470 |
| 360 | 6 | 8 |
| other tail | 81 | 168 |

76.8 % against 76.3 %. That is not near the 60 % bar, and the match to the eval mixture is closer
than I expected. **Accept and record**, per your Q1a.

**Q1b, native rate** (your bar: within a few points of 22.1 %, record and move on):

| native fps | dev (n=449, interim) | test (n=931) |
|---:|---:|---:|
| 29.97 | 291 | 570 |
| 30.0 | 67 | 142 |
| 23.976 | 54 | 116 |
| 24.0 | 31 | 90 |
| 59.94 | 4 | 7 |
| 25.0 | 1 | 1 |
| 26.565 | 1 | 4 |
| **≤ 24 fps — not thinned at `--fps 24`** | **85 — 18.9 %** | **206 — 22.1 %** |

**18.9 % against 22.1 %, a 3.2 pp gap.** On your stated bar that is "within a few points", so my
reading is record-and-move-on. Two caveats I am not hiding: it is half the set, and the gap is in the
direction that gives dev *less* unthinned data than the eval set rather than more. If the final tally
drifts past ~5 pp I will say so and put it in the J9 writeup as a confound rather than a footnote.

## 4. Nothing of mine is blocking you

`OPEN-QUESTIONS-POSE-2026-10-05.md` is down to **Q3b only** — you answered it (scp, both
normalisations, ~320 MB), so really it is just me executing. Q4 is settled your way: pose engine
first, then the LM, matching pipeline order. Q5 agreed, J11 stays unrun. Q6 is Tushar's call and he
has deferred it to after submission; no answer wanted.

Next from me: final dev histograms and the dev `meta.json` set when the fetch lands, then board
extraction at native rate, once.
