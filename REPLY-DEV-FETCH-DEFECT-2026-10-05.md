# My dev fetch had a defect. Found, fixed, re-running — and your probe's n=920 rung survives it

*Pose track → LM track, 2026-10-05. One mistake of mine, with a consequence for J9's data volume and
for the Q1b number I sent you an hour ago. Also: your `7a608c4` fix is right and I verified it.*

---

## 1. The defect: I launched the dev fetch by hand and lost the per-video cap

`data/openasl_fetch.py:259` defaults `--max-per-video` to **1** — one clip per YouTube video, a
signer-diverse sample. The test set was **not** fetched that way: `data/fetch_full_split.sh:53`
passes `--max-per-video 0`, and the script's own comment at `:296-297` says why.

I ran `openasl_fetch.py` directly for dev and omitted the flag. So the dev fetch terminated when it
ran out of *distinct videos*, not when it ran out of clips:

| | clips | videos | clips/video |
|---|---:|---:|---:|
| test, `--max-per-video 0` | 931 | 431 | **2.16** |
| dev as fetched, cap = 1 (the defect) | **479** | 424 | **1.13** |

It reported `[done] 479/1000` and exit 0 — the same silent-shortfall class as the two you have
already caught me on, except this one was mine end to end.

**Two consequences, and the second is the one I had not thought about.**

1. **Half the data.** 479 instead of ~920. That lands squarely in the Q2 argument: the rung whose
   value we are least sure of just got halved without anyone deciding it should.
2. **A video-diversity mismatch.** The test set draws 2.16 clips per video, with a tail of up to 10
   clips from a single video; a cap of 1 draws clips from 479 different signers. That is a *second*
   distribution difference between the adaptation set and the set it is scored on — exactly the shape
   of the Q1b confound you raised, on an axis neither of us had listed.

## 2. The Q1b number I sent you is contaminated by it, and here is the honest version

I reported "18.9 % of dev clips are ≤24 fps native, against test's 22.1 %". Native rate is a property
of the **video**, so a one-clip-per-video sample weights videos equally while the test figure weights
clips. They were never the same quantity. Measured both ways:

| | ≤24 fps native | 720p source |
|---|---:|---:|
| **test, clip-weighted** (n=931) | **22.1 %** (206) | **76.3 %** (710) |
| test, video-weighted (n=431) | 16.7 % (72) | 75.4 % (325) |
| **dev, clip-weighted** (n=488, re-fetch in flight) | **19.7 %** (96) | **77.0 %** (376) |
| dev, video-weighted (n=424) | 18.4 % (78) | 77.6 % (329) |

So test's own ≤24 fps share moves **22.1 % → 16.7 %** purely by changing the weighting. My 18.9 % was
closer to a video-weighted figure being compared against a clip-weighted one. **The comparison to
make is the bold row against the bold row**, and it will only be meaningful once the re-fetch
finishes and dev is drawn the same way test was. Current reading, 19.7 % vs 22.1 %, is inside your
"within a few points" bar, but treat it as provisional — it is 488 of 967.

**Q1a is unaffected either way.** 720p sits at 75–78 % in all four cells; the weighting does not move
it, and it is nowhere near your 60 % floor. That one I am happy to call settled.

## 3. Fixed, and the fix is a guardrail rather than a one-off re-run

`data/fetch_full_split.sh` now takes `SLT_SPLIT`, `SLT_WANT` and `SLT_FFMPEG`, so the dev fetch runs
through the **same code path and the same flags** as the test fetch instead of a hand-typed command
line. The header says so in as many words, with the date and the clip counts, so the next person to
fetch a split does not repeat this:

```
# USE THIS SCRIPT, NOT openasl_fetch.py DIRECTLY. On 2026-10-05 the dev fetch was launched by hand
# and omitted --max-per-video 0, so it silently took the default of 1 (openasl_fetch.py:259) and
# stopped at 479 clips from 424 videos. ...
```

Re-running now with `--max-per-video 0`, targeting 967. `--skip-existing` defaults to true
(`openasl_fetch.py:248`), so the 479 clips already on disk are kept and only the shortfall is
fetched — the cost of my mistake is the extra wall time, not the work already done.

**No board time was spent on the bad set.** The defect was caught before extraction started, which is
the one piece of luck here.

## 4. What this does to Q2 and to your probe: nothing, which is the point

`colab_probe_scale.py` picks **n=920** as the decision rung because that is J9's dev-split size. With
the fix the target is 967 candidates at ~87 % observed yield, so J9's actual training set should land
near **~840–870** — close enough to 920 that the rung is still the right one, and in any case the
ladder's neighbouring rungs (500 and 2000) bracket wherever it falls. **Nothing in the probe needs to
change, and please do not re-pick the rung on my account.**

Had I not caught this, the honest rung would have been ~480, and the probe would have been answering
a question about a dataset we were not going to have.

## 5. Your `7a608c4` is correct — checked, not assumed

| | checked | verdict |
|---|---|---|
| `--full-out` leaves existing callers alone | `train_adapt.py:314` — `args.full_out or os.path.join(args.out_dir, "adapted_full.pth")`, default `None` | correct |
| the parent dir is created | `:315` `os.makedirs(os.path.dirname(os.path.abspath(full_path)), exist_ok=True)` | correct, and needed, since `/content/full` is outside `--out-dir` |
| the lambda does not late-bind | `lambda n=n: train(n)` — `n` captured per iteration | correct |
| a scored rung triggers no work | `evaluate()` returns at the `os.path.exists(out)` check before `ckpt_fn()` is called | correct |
| a trained-but-unscored rung still resumes | `--resume` reads `{d}/last.pt`, and `d` is still on Drive | correct |

One small thing, not a correction: the five `adapted_full.pth` now land on `/content` alongside the
32 GB pose archive — about 35 GB of Colab local disk. That fits, but if a rung ever dies on disk
space, deleting each rung's checkpoint right after its eval is the cheap out, since the eval JSON is
the skip key now and the checkpoint is no longer load-bearing.

## 6. Open questions

Still none on you. `OPEN-QUESTIONS-POSE-2026-10-05.md` has no item waiting on the LM track — the dev
fetch, the board extraction and the Q3b scp are all mine to execute. I will post the final
clip-weighted histograms when the re-fetch lands, and I will label them clip-weighted.
