# Q8: adapted @ 16 fps does **not** dominate un-adapted @ 24 fps, and the choice Q8 asks about does not exist

*LM track → pose track, 2026-10-05, on `23b4e42`.*

---

## 1. The arithmetic, carried through, leaves it behind on both metrics

Q8 pairs the dev-set recovery with the frontier's energy column but never completes the accuracy side.
Doing so, with `results/frontier.csv` beam-4 rows and `results/adapt_ci_dev.json`:

| | BLEU-4 | ROUGE-L | system J |
|---|---:|---:|---:|
| un-adapted @ 24 fps *(the current recommendation)* | **22.80** | **43.13** | 36.1 |
| un-adapted @ 16 fps | 21.54 | 41.35 | 27.0 |
| 16 fps adaptation recovery (dev, authors' poses) | +0.464 | +1.044 | — |
| **adapted @ 16 fps, projected** | **22.00** | **42.39** | **27.0** |
| **shortfall against un-adapted @ 24 fps** | **−0.80** | **−0.74** | −25 % |

**It is behind on both metrics, on point estimates, before any interval is considered.** The
recommendation does not move.

What it *is*: a cheaper Pareto candidate. 25 % less energy for ~0.8 points. That is a legitimate
addition to the frontier — it would lift the existing 16 fps Pareto point, not displace the 24 fps
one — and it is worth having. It is not a headline change, and the report should not be rewritten
around it.

## 2. On the frontier's primary metric the recovery is **not established**

The frontier's accuracy axis is BLEU-4. The 16 fps adaptation gain in BLEU-4 is
**+0.464 [−0.170, +1.064]** — the interval spans zero. Only the ROUGE-L gain (+1.044 [+0.247, +1.802])
is established. So the quantity Q8 needs is the one we cannot claim, which is the same metric-split
§2.5h keeps producing: frame-rate effects register on ROUGE-L, pose-source effects on BLEU-4.

## 3. That row is a projection, not a measurement, and must not enter the frontier as one

The +0.464 / +1.044 were measured on **967 dev clips**; the 21.54 / 41.35 cells are **976 test clips**.
Adding a dev delta to a test cell is arithmetic across two different sets. The number in the table
above is there to show the conclusion, not to be quoted.

**To put "adapted @ 16 fps" on the frontier it has to be measured**: evaluate the adapted checkpoint
on the test split at `--fps 16`, producing an eval JSON, then bootstrap it against the un-adapted
16 fps cell on shared draws like every other cell. That is Colab/Mac work with no board time, and it
is the only version of this row that can be published.

## 4. The choice Q8 offers does not exist — the pose pkls are rate-independent

> *"I can supply either rate from the same dev poses at no extra board cost."*

There is nothing to supply. `jetson/p13_dev_poses.sh` extracts at the clip's **native** rate and passes
no `--fps`; thinning happens at load time in `common/pose_to_unisign.py: fps_ratio_for_clip`, which
derives the ratio per clip from the pkl's own frame count. **One dev pose set serves every target
rate**, 24 and 16 alike, from the same files. Extract once at native rate, as the script already does,
and the rate is a training flag.

## 5. J9 should train at `--fps 24` regardless, for two reasons that are not about energy

* **Training at 16 fps conflates the two shifts in one checkpoint.** J9 exists to measure adaptation to
  **pose source** (−2.24 BLEU-4). Adding a simultaneous frame-rate shift means neither effect can be
  attributed afterwards — the same failure the control arm was added to prevent, one level up.
* **It breaks the pairing.** The baseline J9 is scored against, `eval_n400_pruned_ours_fps24.json`
  at 21.57, is a 24 fps run. Training and evaluating at 16 fps would need a new un-adapted 16 fps
  baseline on our own test poses, which does not exist.

If "adapted @ 16 fps" is wanted as a frontier row, it is a **separate** evaluation of the same
`--fps 24`-trained checkpoint, or of L15's existing frame-rate-adapted checkpoint — not a change to
J9's training rate.
