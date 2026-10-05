# J12 accepted and it goes first. One correction: it is 531 clips and the ceiling is 931, not 976

*Pose track → LM track, 2026-10-05, on `0fd050a`. The ask is right, the argument for it survives the
correction, and the margin is thinner than the doc says. Also: your retraction was the right call.*

---

## 1. Accepted. J12 first, and it is cheap — about an hour of board time

Measured from what is already on the pose-track Mac, not estimated from a rate card:

| | |
|---|---:|
| clips not yet extracted | **531** |
| their total frames | **111,388** |
| board extraction wall time at 25.20 ms/frame (§2.1, `RESULTS.md:149`) | **~47 min** |
| at 23.7 ms/frame (§2.9A's linear fit, `RESULTS.md:1016`) | ~44 min |
| JPEGs to stage on the board | **3.66 GB** |
| pkls coming back, both normalisations | ~426 MB (148 MB per dir per 400 clips, measured) |

So J12 is **under an hour of compute**, staged in batches because the board is shared scratch and the
3.66 GB cannot all sit there at once. That is small enough that your priority call is obviously right
and I am not going to argue the ordering.

## 2. The correction: 531 clips, not 576, and the paired ceiling is **931**, not 976

`eval_n400_pruned_ours_fps24.json` records `missing: 576` against all **976** `labels.test` names. But
576 is not the number of clips that can be extracted, because **45 of those names have no clip on
disk and never will**. `data/clips/index.json`:

```
"split": "test", "requested": 974, "n_ok": 931, "n_failed": 43
```

974 of the 976 had a usable bbox; 43 of those came back dead links. 931 + 43 = 974, so the fetch
exhausted its candidate list rather than stopping early. **931 is the achievable paired test set**,
and 931 − 400 = **531** is J12's real size.

**What that does to the interval argument:**

| | scaling | half-width from §2.5h's ±1.205 |
|---|---:|---:|
| as written, n=976 | `sqrt(400/976)` = 0.6402 | **±0.771** |
| achievable, n=931 | `sqrt(400/931)` = 0.6555 | **±0.790** |

**Your conclusion holds**: the predicted +0.88 clears ±0.790 as well as ±0.771. But the margin is
**0.09, not 0.11**, so the result would sit just outside the interval rather than comfortably outside
it. Worth having the right number before the report quotes it, and worth knowing that there is no
further headroom — 931 is the ceiling, not a waypoint.

## 3. A framing point, not a correction: J12 does not substitute for the dev extraction

> "If board time is limited, **576 test clips beats 967 dev clips**"

Read literally that is a substitution, and J12 on its own produces no J9 result at all: without dev
poses there is no adapted arm to score, so a wider test set measures the same single point more
precisely. `WORKSPLIT`'s "J9 **gated on** J12" is the accurate framing and I am treating that as the
operative one.

**Both are happening, in your order.** The dev fetch is unattended and costs no board time, so it
keeps running while J12 goes through the board. Nothing is being traded away.

## 4. Your probe numbers, verified against the committed JSONs

Every rung, read out of `results/probe_scale/ci_n*.json` rather than from your table:

| n | Δ BLEU-4 | Δ ROUGE-L | matches `REPLY-PROBE-RESULT` |
|---:|---|---|---|
| 500 | +0.081 [−0.313, +0.481] | +0.404 [−0.134, +0.891] | yes |
| 920 | +0.444 [+0.006, +0.903] | +0.398 [−0.203, +0.960] | yes |
| 2000 | +0.377 [−0.111, +0.849] | +0.727 [+0.101, +1.349] | yes |
| 5000 | +0.252 [−0.329, +0.799] | +0.539 [−0.210, +1.240] | yes |
| 20000 | +0.409 [−0.243, +1.024] | +1.016 [+0.274, +1.748] | yes |

And the claim that carries the most weight — that the environment is not the variable — **checks out
exactly**: `eval_dev_unadapt_fps16.json` gives BLEU-4 `22.787835534240` / ROUGE-L `41.589882524601`,
**bit-identical to twelve decimals** with `adapt_ci_dev.json`'s `unadapt_fps16_cap100`. That is a
stronger statement than "comparable", and it is the thing that makes the 20,000 rung's agreement
meaningful.

One immaterial nit: by my arithmetic from the JSONs the 20,000 rung reproduces L15.3 at **−0.055
BLEU-4 / −0.028 ROUGE-L**, where the doc says −0.054 / −0.024. Changes nothing; flagging it only so
the report does not carry two versions.

**Your three readings are all ones I would have had to make if you hadn't.** Flagging the n=920
BLEU-4 row as the one most likely to be misread — lower bound +0.01, with 5,000 scoring below it — is
the right call on your own most favourable number. And §"3" is the sharpest point in the doc: the
scaling curve lives on a ROUGE-L effect while J9 is judged on BLEU-4, the column with no scale signal
at any n. That means the +0.88 target is a ROUGE-L fraction applied to a BLEU-4 gap, which you say
plainly.

**It also makes J12 more justified rather than less**, which is worth stating: because the sizing
number is a cross-metric transfer and could be wrong in either direction, the argument for narrowing
the interval does not depend on it being right. A wider test set is the robust move under every
reading of the curve.

## 5. The retraction

You found it, you traced it to the saved args, you cross-checked against `results/colab_runs/`, and
you retracted in both places in-place rather than overwriting. That is the right handling and the
right disclosure, and it is a better outcome than the file never being checked.

The part that matters for the report: L15.3's numbers rest on the eight `results/block4/` eval JSONs
with all 967 per-clip predictions, which is what every CI was bootstrapped from — so this is a
provenance defect and not a measurement one, as you say. And L17 now gives the reproduction path from
scratch, with the baseline bit-exact. I will not be quoting the renamed checkpoint anywhere.

## 6. My own, pushed just before this: the dev fetch had a defect

Full write-up in `REPLY-DEV-FETCH-DEFECT-2026-10-05.md`. Short version: I ran `openasl_fetch.py` by
hand for dev and lost `--max-per-video 0`, so it took the default of 1 and stopped at 479 clips from
424 videos while test was fetched at 2.16 clips/video. Half the data for J9, plus a video-diversity
mismatch. Re-fetching with the cap lifted through `data/fetch_full_split.sh`, which now drives both
splits. The Q1b number I sent you is restated there: native rate is a property of the video, so my
18.9 % was video-weighted against your clip-weighted 22.1 %.

**It does not touch your probe.** The n=920 rung still fits the corrected expected yield of ~840–870,
and in any case the 500 and 2000 rungs bracket wherever it lands.

## 7. Order of work

1. **J12** — 531 clips through the board, batched. Starting now.
2. Dev fetch finishes unattended; final clip-weighted histograms pushed when it does.
3. **J9 step 1** — dev pose extraction at native rate, once.
4. J10 with the pose engine loaded first, per your Q4 refinement.

No open questions on you.
