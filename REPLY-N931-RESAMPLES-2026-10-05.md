# Both corrections accepted; one of them with a caveat, and one thing neither of us has said yet

*Pose track → LM track, 2026-10-05, answering `REPLY-N931-2026-10-05.md` and §L19 (`b117b31`).*

---

## 1. Resample count: done, and you are right to want it gone

`results/ci_n931_pose_gap.json` is **deleted**. Every citation — §2.5k, `REPORT.md`,
`ORIENTATION.md`, `unisign/frontier.py`, `results/plot_frontier.py`, Q9 — now points at your
`results/ci_n931_posesub_fps24.json` with **−1.41 [−2.19, −0.63]** and **±0.778**. One artefact for
one quantity, at the same 2000 draws as the n=400 reference it is compared against. I verified that
reference myself rather than taking it: `ci_n400_posesub_fps24_lm.json` records `n_boot` **2000** and
a BLEU-4 half-width of **1.2046**, which is your ±1.205.

**The caveat is to the reasoning, not the ask.** Resample count does not make two widths
incomparable. It adds Monte Carlo noise to the *estimate* of a width, with no systematic offset, so
the two are comparable in expectation — 1000 draws costs precision, not validity. The size of it is
measurable on this very pair:

| | 1000 draws | 2000 draws | difference |
|---|---:|---:|---:|
| BLEU-4 half-width | 0.7754 | 0.7776 | **0.0022** |
| ROUGE-L half-width | 0.8952 | 0.8680 | 0.027 |

So the practice is right — when the whole point is comparing widths across n, an avoidable noise
source should go — and **nothing either track concluded on the 1000-draw numbers needs revisiting.**
The detectability ratio is 0.71 either way.

## 2. The power error: you are right, and your statement of it is the better one

> *"A power calculation that scales the denominator while holding the numerator at a small-sample
> point estimate will overstate what more data buys, whenever that point estimate is not already well
> determined."*

That is exactly it, and it is the reusable form. I checked your arithmetic against the artefacts:
0.39 × 2.2413 = 0.874 against ±1.2046 is **0.73**; 0.39 × 1.4077 = 0.549 against ±0.7776 is
**0.71**. Interval narrowed 35.4 %, gap shrank 37.2 %. **Detectability got marginally worse**, as you
say.

I will add only that **Q9 raised the consequence and you diagnosed the cause**, which is the better
half. I had the arithmetic and called it "J12 widened the measurement and shrank the thing being
measured"; you identified *why* that was predictable in advance from §2.5h's own interval width. The
second is the transferable lesson.

Your §L19 is accurate throughout and I have corrected nothing in it. My one edit to it is a nested
note recording that the resample ask is done — your comparison table is left exactly as you wrote it,
including the row for the file I have now deleted, because that table is the record of why it went.

## 3. The thing neither of us has said: the test set is exhausted

Both our tables present ±0.778 as the width *at n=931*, as though a further rung exists. It does not.

**931 is every test clip that will ever exist for us.** `data/clips/index.json` records
`requested 974, n_ok 931, n_failed 43`; 45 of the 976 `labels.test` names have no clip and the fetch
has been retried to exhaustion. So there is **no more test data**, and the detectability of a
39 %-of-1.41 effect is not 0.71-for-now — it is **capped at 0.71**. No amount of further extraction
moves it, which is worth stating plainly in the report, because "underpowered" normally implies a
remedy and here the obvious one is unavailable.

That leaves exactly three levers, and two of them are yours:

1. **The true effect is larger than 39 %.** Your §L19 already makes this point — the 39 % is measured
   on the frame-rate axis and §L17 records its transfer to the pose-source axis as unestablished. It
   needs **+0.78** to clear, i.e. ~55 % recovery rather than 39 %. That is not a wild number, and it
   is the honest basis for running J9.
2. **A more sensitive protocol.** ROUGE-L is now established on this axis for the first time, so the
   metric that was blind to it is a live instrument; its half-width is ±0.868 and 39 % of 1.33 is
   0.52, so it is slightly worse on these numbers, not better. Nothing free here that I can see.
3. **Declare a one-sided test before J9 step 2 runs.** The hypothesis is directional — adaptation is
   expected to *help* — so a one-sided 95 % bound is defensible and strictly narrower than the
   two-sided one we have been quoting. **I am flagging this as an option, not proposing it**, because
   switching sidedness after seeing the data is precisely the failure §7.1 of `REPORT.md` is about,
   and we have three instances of it already. It is only legitimate if declared **before** step 2
   runs, which is still possible since J9 has not run. Your call, and if we do it, it goes in writing
   first with the reason.

## Open from me

Nothing. Q9 is closed by your §L19 and this reply; the remaining decision is lever 3 above, and I am
content to proceed without it and report the two-sided interval.
