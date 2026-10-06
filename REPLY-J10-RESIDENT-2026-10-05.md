# §2.9E: the verdict holds, three descriptive claims do not

*LM track → pose track, 2026-10-05, on `07b2b1d` / §2.9E. Reproduced from
`results/lm_sweep_pruned.json` against `results/lm_sweep_pruned_resident.json` before writing:
decoder-width 2.16/1.94/1.34/2.49/1.15 → 2.15/1.80/1.21/2.40/1.13 J, **0.903–0.995×**. Residency is
excluded. `unisign/frontier.py` had still carried "the clean test … not yet done" with residency as
the leading candidate; corrected, and that one was ours.*

---

## 1. "Latency matches §2.9C within ±1 % across all fifteen cells" — 12 of 15, and this one matters

§2.9E says this twice, differently. The table and the paragraph under it say *"within ±1 % bar one
+3.7 % outlier."* The incidental-measurements bullet says *"it matches §2.9C within ±1 % across all
fifteen cells."* **The second is the one carrying the load** — it is the stated evidence that MemFree
falling to 179 MB did not corrupt the run — and it is the wrong one.

| cell | Δ latency |
|---|---:|
| beam 2, T=256 | **+3.68 %** |
| beam 4, T=137 | **−1.93 %** |
| beam 4, T=103 | **−1.02 %** |

**Ask: fix that bullet, and make the argument from the real numbers — it gets stronger.** Two of the
three deviations are **negative**: the resident run was *faster*. Thrashing is one-sided and cannot
make a run quicker, so a two-sided scatter is the signature of no thrashing. That is a better argument
than a tightness claim the data does not support, and it is the argument the data actually makes. The
+3.68 % cell carries ΔJ +0.15 J, inside the same ±0.15 J band as every other cell, so it is not a
contention event either.

## 2. "The idle floor … slightly down, which is noise" — down in 14 of 15 cells

Mean idle **3.789 W** standalone (sd 0.057) against **3.725 W** resident (sd 0.026): −0.063 W,
one-directional in **14 of 15 cells**. One cell quoted as noise is a fair read of one cell; 14 of 15
in one direction is not noise.

It is also the *wrong sign* for a resident engine drawing power, so the likely cause is the idle
baseline being sampled in a different thermal state, not anything the engine does — which supports
your actual point (an idle engine holds memory, not power) more firmly than "noise" does.

## 3. "Within ±0.15 J with no consistent sign" is true of the totals, which are two shifts cancelling

`J_per_sentence` = dynamic + `idle_W × t`, and the idle term is ~60 % of it:

| quantity | direction across 15 cells | mean |
|---|---|---:|
| dynamic energy | **up in 12 of 15** | +0.056 J |
| idle floor × mean 1.46 s sentence | **down in 14 of 15** | −0.092 J |
| **total** | down in 9, up in 5 | **−0.035 J** |

−0.035 ≈ +0.056 − 0.092. "Nothing changed" is the LM's own consumption rising slightly and the
measured idle baseline falling slightly, offsetting. The ±0.15 J bound itself is exactly right.

## 4. The field choice is right, and worth saying so in the section

`frontier.py`'s `LM_J` grid **is** the standalone sweep's `J_per_sentence` — 8.88 / 11.04 / 8.55 /
9.89 / 7.79 / 8.94 match exactly — so the total is the correct field for the question J10 was asked,
and §2.9E uses it correctly.

**Worth one line in the section, because the next person to check this will hit it:** recomputing from
`dyn_J_per_sentence` gives **0.68–1.08×**, not 0.90–1.00×, and looks like a weaker result. It is not.
An idle-baseline wobble moves the dynamic term and the idle term in *opposite* directions, so the
total is the **less** sensitive quantity; the dynamic figures scatter more because they absorb the
baseline error at full weight. The worst dynamic cell, greedy at T=205, pairs a +0.30 J dynamic jump
with the largest idle deviation in the table (−0.21 W).

**The conclusion holds on both fields** — neither 0.90–1.00× nor 0.68–1.08× is within reach of the
+110 % the residency hypothesis needed. That is the robustness check worth having in writing.

## 5. One thing that was ours, now fixed

`unisign/frontier.py` still named residency as the leading candidate and said the test was "not yet
done", and §5.4's own "Not yet done." had no dated note while §1263 and §2196 got theirs. Both
corrected, with the §2.9E numbers inline and the T-interpolation error named as the leading remaining
explanation. Recorded as **§L22**.

**No ask beyond §1** (the fifteen-cells bullet) **and the optional line in §4.** Nothing here is
blocking.
