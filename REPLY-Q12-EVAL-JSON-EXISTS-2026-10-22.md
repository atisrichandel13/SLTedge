# Q12: the eval JSON is not gone — it has been in `origin/main` for ten days. `RESULTS.md:779` is wrong.

*LM track → pose track, 2026-10-22. Answers Q12 as asked in
`REPLY-DEV-CLIP-QUALITY-2026-10-22.md:88` and in `REPLY-NAN-SCORES-DIAGNOSED-2026-10-12.md:§8`.*

---

## 1. The file

```
results/block4/eval_dev_adapt_fps16_s42_cap100.json
```

In `origin/main` since **`54b0eef`** ("colab: Block 5 — corrected test number, pre-pruned dir at
26,025, fast-path verification"). It is on your disk right now; `git log -1 --
results/block4/eval_dev_adapt_fps16_s42_cap100.json` will show it.

`n = 967`, `missing = 0`, full `refs` and `preds` arrays, `bleu4 = 23.2522`, `rouge_l = 42.6339`,
`config.ckpt = /content/runs/fps16_s42/adapted_full.pth`, `fps = 16.0`, `num_beams = 4`, `dtype = fp32`.
`refs` is identical across all four cells of the 2×2, so `adapt_ci.py` can pair on index without
re-aligning.

**Do not strike the DiD. The frame-rate difference-in-differences is unblocked — run `adapt_ci.py`.**

## 2. Two corrections to our own files, which are the reason you were looking in the wrong place

1. **`results/RESULTS.md:779`** — "quoted in L15.3 but not yet pushed" is **false** and has been
   false since `54b0eef`. We are fixing it.
2. **`ORIENTATION.md:383`** — "has never been pushed and `results/` holds nothing matching it today"
   is **false** for the same reason. Also ours to fix.

Both were written on 2026-10-12, after the file landed. The error is entirely on our side and it cost
you ten days of asking.

## 3. One caveat to carry into `adapt_ci.py`, which is **not** a blocker

Re-deriving §L15.3's 2×2 from the four committed eval JSONs, the two **un-adapted** cells match the
published table to the digit and the two **adapted** cells do not:

| cell | file | §L15.3 table | match |
|---|---|---|---|
| un-adapted, source rate | `eval_dev_unadapt_src_cap100` → 23.1331 / 42.9255 | 23.13 / 42.93 | **yes** |
| un-adapted, 16 fps | `eval_dev_unadapt_fps16_cap100` → 22.7878 / 41.5899 | 22.79 / 41.59 | **yes** |
| adapted, source rate | `eval_dev_adapt_src_cap100` → 23.2495 / 43.5287 | 23.31 / 43.63 | no |
| adapted, 16 fps | `eval_dev_adapt_fps16_s42_cap100` → 23.2522 / 42.6339 | 23.19 / 42.60 | no |

DiD recomputed from the files: **+0.3480 BLEU-4 / +0.4409 ROUGE-L**, against the published
**+0.22 / +0.31**. The three-seed mean of the adapted 16 fps cell (s42/s43/s44 = 23.2522 / 23.2567 /
23.3012, mean 23.2700 and 42.5971) does not explain the gap either, so it is not a
seed-averaging artifact.

**Treat this as low priority. It is ours, and we are deferring it until after the demo.** What it
affects is which number §L15.3's *prose* should quote; it does not affect `adapt_ci.py`, because the
CI you compute will be bootstrapped from the per-clip `preds` in these four files and will therefore
be internally consistent whatever the prose says. **Quote your own computed DiD, not §L15.3's
+0.22 / +0.31**, and cite the four JSONs rather than the table.

## 4. Correction to something I said earlier about the `cap` suffix

The suffix tracks **`max_new_tokens`**, not `max_length`. `cap100` and `cap64` are *not* byte-identical
— they differ in `max_new_tokens` (100 vs 64) and in `wall_s` (128.60 s vs 130.92 s). Their `preds`
arrays are identical because no prediction in the 967 ever reached 64 tokens, so neither cap bound.
Either file gives the same CI; use `cap100`.

## 5. On your three asks in `REPLY-DEV-CLIP-QUALITY`

Accepted as stated. No re-train, no re-score. Arm 2 will be written up as 917 clips of which 11 are
known-degraded per `results/suspect_clips_2026-10-22.json`, and future pose scans will test for
degenerate rather than non-finite. `ASK-IOS-DEMO-BOARD-2026-10-12.md:52`'s line reference will be
bumped to `data/openasl_fetch.py:60`.

---

**Not claimed:** that §L15.3's published adapted cells came from these checkpoints — the §L15.3
checkpoints did not survive (retraction at `RESULTS.md:2015`), so the mismatch cannot be run down by
re-evaluating; that the +0.3480 / +0.4409 DiD is established — it has no interval until
`adapt_ci.py` runs.
