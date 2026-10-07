# My Q10 ask was one tar short: step 2 needs the TEST poses on Drive as well

*LM track → pose track, 2026-10-07. A correction to `REPLY-Q10-POSE-DELIVERY` §1, found while
building the step-2 runner. Nothing new to extract — the file already exists.*

---

## 1. The ask: add `pkl_split_rtmw_fp16` to the same handoff

`REPLY-Q10-POSE-DELIVERY` asked for `pkl_dev_rtmw_fp16` only, on the reasoning that dev is the
training input. That was right as far as it went and **it missed step 3**: the adapted checkpoints
have to be *scored*, on the test set, with our poses — and `results/pkl_split_rtmw_fp16` is on the
board and your Mac, not on Colab.

**Ask: package and stage `pkl_split_rtmw_fp16` the same way** — `data/package_dev_poses.sh` already
does it with `SLT_PKL`, so nothing new is needed:

```
SLT_PKL=results/pkl_split_rtmw_fp16 SLT_EXPECT_N=931 ./data/package_dev_poses.sh
```

Both tars in one trip, since the upload is a manual browser step either way. Dev ~294 MB plus test
~326 MB is **~620 MB on Drive**, so the headroom question matters more than it did — that check is
still ours and I have not run it yet.

`results/pkl_split_rtmw_fp16.manifest` is already committed (931 bare names, generated from the
n=931 arm's own `names`), so the guard has its reference on both sides.

## 2. Why the eval cannot run on the board instead, which was my first idea

It would avoid the second upload, and it does not work: **the two checkpoint lineages have different
vocabularies.**

| | checkpoint | mt5 | keep set |
|---|---|---|---:|
| board / our Mac | `weights/openasl_pose_only_slt_pruned.pth` | `weights/mt5-base-openasl-pruned` | **26,078** (train+dev+test) |
| Colab (`colab_setup.py`) | `/content/pruned_traindev.pth` | `weights/mt5-base` | **26,025** (train+dev, leak-free) |

Verified: `results/openasl_vocab_keep_ids_traindev.json` holds 26,025 ids. An adapted checkpoint
trained in the Colab lineage cannot be scored against the board's n=931 arms — the delta would
conflate adaptation with a 53-token vocabulary change *and* with the test-set leak, and §L18 already
records the mt5 directory as a decode-protocol axis that changes outputs.

**So all three arms get scored in one lineage on Colab**, and the absolute BLEU-4 will not equal
21.73. That is fine and is stated in the runner's docstring: the deliverable is a delta, the lineage
offset is shared by every arm, and it cancels. **Worth knowing before you see a number that looks
wrong against §2.5k.**

## 3. A bug in the probe that is worth your attention, since it touches §L17's provenance

`colab_probe_scale.py`'s summary block formatted `ev["bleu"]` as a float. `eval_openasl` writes
`bleu` as a **dict** `{bleu1..bleu4}` in every vintage — checked against all five
`results/probe_scale/evals/*.json` and both n=931 arms — so that line raises
`unsupported format string passed to dict.__format__`.

**It fired after every rung had been trained and bootstrapped**, which is why §L17's curve was
reconstructed from the artifacts by hand rather than printed by the script. **The artifacts were
never wrong; only the printer was.** Fixed, and the fixed printer reproduces the published curve
exactly (+0.08 / +0.44 / +0.38 / +0.25 / +0.41). No §L17 number changes.

**No ask here** — flagged because a crash in a summary looks like a failed run, and the next person
to re-run that script should know it was cosmetic.
