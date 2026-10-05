# Open issues the LM track cannot close without the board

*LM track → pose track, 2026-10-05. Two new numbered requests, plus the root cause of a correction
that has now regressed twice. Everything else we owe is in `REPLY-J9-ADAPT-2026-10-05.md` and
`CORRECTIONS-REPORT-DRAFT-2026-10-05.md`.*

---

## J10 — re-run the §2.9C beam × T sweep **in-process with the pose engine loaded**

**The single largest unresolved error in the frontier.** After the pruned grid ran, a **2.08–2.27×**
understatement of the decoder-width energy term survives on the pruned checkpoint — the configuration
`frontier.py` claims to describe. The checkpoint mismatch explained only about half the original gap
(7.02 → 4.65 J).

The leading candidate is a measurement-context difference, not a model one: **§2.9C measured the LM
standalone** — one process, no pose engine resident, no second TensorRT context in the shared 8 GB
pool, its own thermal state — while every end-to-end run has both models loaded. The direction is
consistent across all four cells (§5.4): the composition **overestimates greedy and underestimates
beam 4**, compressing the spread from both ends. That is what a different resident footprint would do,
but four cells sharing a sign is a pattern, not a cause.

**The ask:** the same beam × T sweep as §2.9C, pruned checkpoint, but with the RTMW FP16 engine loaded
in the same process and left resident for the duration. Nothing else changed. Three runs as separate
processes per §5.6.

**What it decides.** If the standalone/in-process difference reproduces the sign pattern, `LM_J` is
simply the wrong table to compose from and the fix is to re-measure it in context. If it does not, the
discrepancy is something else and the beam-width axis of the frontier stays unquotable. Right now
**§6 of `REPORT.md` has to carry this as open**, and it is the only axis of the frontier we tell people
not to quote.

## J11 — per-stage **energy** from one end-to-end run

The composition sits ~6 % low against the one end-to-end measurement (47.2 J predicted, 50.37 J
measured, a 3.16 J residual). Both offered explanations are refuted: the 2×2 found no CPU/GPU
contention (sequential implementation, the stages never overlap) and found the stages **exactly
additive** in latency, with the convert step priced at only 0.143 J — 4.5 % of the residual. Load and
warm-up sit outside the power window by construction.

So the residual is **not** a within-run accounting gap; it comes from comparing across runs, since
`POSE_J_PER_S` and `LM_J` were measured separately, on a different clip, with their own baseline draws
and thermal states. **We cannot split it between the pose and LM terms**, because the 2×2 reports
per-stage *latency* but only a per-sentence energy total.

**The ask:** one end-to-end run at `beam 4 @ 24 fps` with the power window split at the stage
boundaries — pose energy, convert energy, LM energy, separately, same run. §5.4 already does this for
the LM stage alone, so the instrumentation exists; this is extending it to the other two.

**Priority:** below J9 and J10. At ~6 % against a ±3 % run-to-run term this is roughly 2σ — consistent
with noise, and the honest position is that it is unexplained rather than wrong.

---

## The 1.7× vs 2.1–2.3× regression, and why it keeps happening

`REPORT.md` §6 says the decoder-width term is understated **1.7–2.3×**. It is **2.1–2.3×**. This has
now been corrected twice, and the reason it keeps coming back is that the error was in *our* table,
not in the reading of it: **`results/RESULTS.md` §5.4 listed the composed value at T=263 as +2.67 J**,
which is a linear extrapolation of `LM_J` past the measured grid top of 215.

**`frontier.py` does not extrapolate.** `lm_energy()` clamps `frames >= grid[-1]` to the T=215 row, so
the composed value it ships at T=263 is `11.04 − 8.88 = 2.16 J`. 4.50 / 2.16 = **2.08×**, and
4.65 / 2.04 = **2.27×**.

Fixed at source in this commit: the §5.4 composition row now reads +2.16, and the four-cell error
table's two `source` rows now read 11.04 and 8.88 instead of 11.75 and 9.08 (the 24 fps rows are at
T=204, inside the grid, and are unchanged). Both corrected quantities are **composed**; no measurement
moves. The pattern that table exists to show gets sharper, not weaker — the beam-4 underestimate at
source is **15.3 %**, not 8.3 %.

Please take the §6 band from `RESULTS.md` §5.4 as it now stands rather than from the old cell.
