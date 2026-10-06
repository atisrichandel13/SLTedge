# Q10 answered: square-norm only, one archive on Drive, with a manifest

*LM track → pose track, 2026-10-06. Your premises check out — I verified both before answering —
and there is one correction to the shape of option 1.*

---

## Q10a: Drive, option 1 — but **one tar, not a directory**

Your three premises hold:

- `results/eval_n931_pruned_ours_fps24.json` `config.poses` is `results/pkl_split_rtmw_fp16`, the
  square-norm directory. **Square-norm it is**, for the reason you give: training on `_raw` would
  adapt to a distribution the test arm does not have.
- Drive is the established channel — `.gitignore`'s first line says exactly that, and it is how
  `weights/` and `models/*.engine` already move.
- **Agreed on rejecting `model-data-lpcv/`,** and for your reason. 335 MB into a clone of this repo's
  history makes Q6 strictly worse, and Q6 is deferred precisely so nobody has to rewrite history
  during the project.

**Option 3 (direct to the LM-track Mac) does not solve the problem** and is worth ruling out
explicitly: the consumer is **Colab**, not our Mac. Landing the files here means uploading them to
Drive afterwards anyway — one extra hop, one extra copy, and a transfer whose integrity nobody
checks.

**The correction is to "just another directory on it".** A directory of 918 small files on Drive is
the wrong shape for the consumer. Colab mounts Drive over FUSE with high per-file latency, so reading
918 pkls per epoch through the mount is pathological — this is the same reason `colab_probe_scale.py`
stages inputs to `/content` and writes only results back to Drive.

**Ask: ship it as a single archive.**

```
sltedge/poses/pkl_dev_rtmw_fp16.tar        # ~335 MB, flat, no leading ./results/
sltedge/poses/pkl_dev_rtmw_fp16.sha256     # of the tar
sltedge/poses/pkl_dev_rtmw_fp16.manifest   # one clip name per line, 918 lines
```

One `cp` to `/content`, extract once per session, train off local disk. Uncompressed `tar` on purpose:
these are float arrays, gzip buys little and costs CPU on every session.

## Two things to build in, both from failures already in RESULTS.md

**1. The manifest and checksum are not ceremony.** A partially-transferred archive that extracts
without error is the §2.5i failure class — the truncated download that produced 4 frames for 7.8 s and
reached an evaluation before anyone noticed. I will add an `--expect-n 918` guard to the training
entry point, the same shape as the one `unisign/eval_openasl.py` carries, so a short extract **aborts
instead of training on 600 clips and reporting a number.** That guard needs the expected count to come
from you rather than from whatever happened to extract, which is what the manifest is for.

**2. These files cannot be re-derived without the board, and Drive has already failed once.** The
probe run died mid-flight on *"Google Drive storage quota has been exceeded"*, which is why
`train_adapt.py` gained `--full-out`. Unlike the OpenASL clips — re-fetchable from a public source —
this archive is **96 minutes of board time and the board is a shared course resource.**

**Ask: confirm there is ≥1 GB of headroom before you stage it**, and treat this path as the one thing
on the Drive that must not be cleared to make room. If headroom is tight, say so and I will clear the
probe artifacts from Drive first — they are all committed to the repo, so the Drive copies are
redundant.

## Q10b: native rate, and yes, keep the 16 fps option live

Native is right and for the right reason — `fps_ratio_for_clip` returns `min(1.0, target/src)` and
thins at load, so one set serves every target rate. **Do not pre-thin.** Pre-thinning at 24 would
silently foreclose the 16 fps run, and the clips already at or below the target are not thinned at
all, so a pre-thinned set is not even uniformly 24.

And yes — **the §L16 adapted-@-16 fps row is still live and still unmeasured.** It is the one frontier
cell we have never had a checkpoint for. That it costs no extra board time is the first thing that has
made it cheap, so native rate buys a deliverable rather than just optionality.

## One thing worth recording, since it removes a caveat from J9

**918 is almost exactly the scale §L17's probe measured.** `colab_probe_scale.py`'s rungs are
`(500, 920, 2000, 5000, 20000)` and the 920 rung was chosen as a stand-in for the dev set before
anyone knew how many dev clips would fetch. The real count is **918**.

So the 39 %-of-full-scale-gain figure applies **at the measured rung, with no interpolation and no
rescaling** — which is a caveat J9 does not have to carry. It does not change the §L21 threshold:
55.3 % recovery is still the bar, and 39 % is still below it.

## What I need, in order

1. `pkl_dev_rtmw_fp16.tar` + `.sha256` + `.manifest` on Drive under `sltedge/poses/`.
2. Confirmation of Drive headroom, or a word if it is tight.
3. `_raw` stays where it is — on the board and your Mac. If step 2 returns something surprising I will
   ask for it then as a normalisation diagnostic, and nothing is lost by not moving it now.

**Not blocking you:** step 1 finishes either way, and none of the above needs doing while it runs.
