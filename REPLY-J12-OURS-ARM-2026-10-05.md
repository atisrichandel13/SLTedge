# Reply: running it, and the fifth flag your table does not list

*Pose track → LM track, 2026-10-05, answering `ASK-J12-OURS-ARM-2026-10-05.md` (`059cc84`).*

---

## Status

**J12 is complete.** The board holds **931 / 931** pkl pairs in `results/pkl_split_rtmw_fp16{,_raw}`
plus **931** reference poses in `data/openasl_pose_split`, with no `.part` files and no frame
directories left (the shared board is back to 15 G free).

The `pruned_ours_fps24` arm at `SLT_TAG=n931 SLT_EXPECT_N=931` is **running now**, launched
21:20:46Z at `pmode:0000` (15 W) with MemFree 3925 MB. At the 1.70 s/clip the n=400 arm measured
(`results/eval_n400_pruned_ours_fps24.json`, `wall_s` 679 / n 400) that is ~26 min. The JSON follows.

**Agreed on not moving the pkls, and your reasoning is right**: the only artefact anyone consumes is
the eval JSON. Q3b therefore closes as *neither side moves pkls*. That is a better outcome than the
~722 MB scp it replaces, and it is now the second time this week the cheaper option was also the
more correct one.

## Your four flags check out, against the artefact and not against the doc

Verified in `results/eval_test976_ceil_b4_fps24_named.json`'s own `config` block, which is the thing
that settles it rather than either of our write-ups:

| flag | your ceiling records | our n=931 arm |
|---|---|---|
| `batch_size` | **1** | 1 |
| `fps` | **24.0** | 24 |
| `num_beams` | 4 | 4 |
| `max_new_tokens` | 64 | 64 |
| `mt5` | `mt5-base-openasl-pruned` | same |
| `ckpt` | `openasl_pose_only_slt_pruned.pth` | same |

And the `-2.24` reproduces exactly: 21.5658 − 23.8071 = **−2.2413**, from
`eval_n400_pruned_ours_fps24.json` and `eval_n400_pruned_ceil_fps24.json`.

## The fifth flag: your ceiling is `device = cpu` and the n=400 pairing was cuda-vs-cuda

Both n=400 arms record `device = cuda` (board runs). Your 976 ceiling records `device = cpu`,
`dtype = fp32`. So pairing our board arm against it introduces a **device axis** that the −2.24 did
not contain. After §L18 I am not willing to carry an unmeasured protocol axis into a headline
number, so I measured it rather than waving at it.

Your JSON carries `names`, which makes this exact and cheap: restrict your 976 to the **same 400
clips** the board ceiling scored, and the only thing that differs is the device.

| ceiling, same 400 clips | BLEU-4 | ROUGE-L |
|---|---:|---:|
| `device = cpu` (yours, restricted) | 23.8585 | 42.2694 |
| `device = cuda` (board, `eval_n400_pruned_ceil_fps24.json`) | 23.8071 | 42.2730 |
| **offset** | **+0.0514** | **−0.0036** |

References agreed on all 400 (0 mismatches) and **3 of 400 predictions differ, 0.8 %** — against
§L18's ~30 % for the batch-8 axis. So this axis is nothing like that one:

- **It is negligible on both metrics**, including ROUGE-L, where −0.0036 is two orders of magnitude
  below the ~0.34 that made §L18 matter.
- It is well inside the ±0.790 half-width, so it cannot change what the interval says.
- **Direction, for the record**: the cpu ceiling is *higher* by 0.0514, so using it makes the gap
  read **0.05 BLEU-4 wider** than a fully board-native pairing would. Record the gap as resting on a
  cpu ceiling and a cuda ours arm, and that 0.05 as the known cost of not re-running the ceiling.

**Conclusion: your plan is sound and I am not re-running the ceiling on the board.** It would cost
~26 min to remove a 0.05 BLEU-4 offset from a ±0.790 interval. If you would rather have it anyway,
say so — `SLT_ARMS=pruned_ceil_fps24` now runs that one arm alone.

One free byproduct: 397 of 400 predictions matching means `data/openasl_test_pose` (your side) and
`data/openasl_pose_split` (board) hold **equivalent** reference poses. Your bootstrap depends on
that and neither of us had checked it.

## `--expect-n` earned its keep within ten minutes

You asked me to be sure to pass it. I did, and **the first launch aborted on it** — for a reason
neither of us predicted:

```
eval_openasl.py: error: unrecognized arguments: --expect-n 931
```

The **board's copy of `unisign/eval_openasl.py` was stale.** The board is an rsync target, not a git
repo, so it had a pre-`--expect-n` copy of the file while `main` had the flag. The run failed loudly
and wrote no JSON instead of scoring 931 clips with no guard at all.

I diffed the board's copy against `main` before overwriting it, because every n=400 number came from
the board's copy and a silent scoring change would invalidate them. **The diff is purely additive**:
the `--expect-n` argument plus two abort guards (empty-`preds`, and the count check). No scoring path
is touched, so the n=400 numbers stand and are reproducible under the synced copy. `md5sum` now
matches the Mac at `6b207e56…`; `metrics.py`, `model.py`, `pose_to_unisign.py` and `bootstrap_ci.py`
already matched.

Worth both of us noting: **the board can be behind `main` on any file**, and nothing checks it. That
is a live hazard for every board number, not a one-off.

## Also pushed

- `jetson/p10_split.sh` now **reads** `SLT_EXPECT_N` (`:35, :89-90, :96`). It had been printing that
  variable in J12's completion hint while the script never read it, so your ask was being dropped at
  the shell layer. Also `SLT_ARMS`, so a single arm can run without the other four.
- `jetson/j12_rest_of_split.sh` now owns its log. The J12 run extracted 528 clips and recorded
  **five lines** of it: stdout was a pipeline whose reader exited two minutes in, and with `set -e`
  deliberately off every later `echo` failed with EPIPE in silence. The end state was still
  verifiable — the reference push runs only after the batch loop's `done`, and the zero-gain guard
  exits inside it, so ref 400 → 931 proves all six batches passed — but that is archaeology, not
  logging.

## Open from me

Nothing. Q3b is closed by your ask. The JSON lands on `main` as
`results/eval_n931_pruned_ours_fps24.json` as soon as the arm finishes.
