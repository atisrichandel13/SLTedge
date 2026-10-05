# Dev fetch finished: 918 clips, and it matches the test set on all three axes

*Pose track → LM track, 2026-10-05. Q1a and Q1b closed on like-for-like numbers. J9 step 0 is done.*

---

## The numbers, clip-weighted on both sides

The fetch converged: attempt 2 ended at 918, attempt 3 added **zero**, so the remaining 49 are
permanent dead links and I stopped it there rather than letting it re-walk. 918 of 967 is a **94.9 %
yield**, against the test split's 95.6 % (931 of 974) — the same ballpark, which is what you would
want from the same fetcher against the same source.

| | test (`data/clips`) | dev (`data/clips_dev`) | your bar |
|---|---:|---:|---|
| clips | 931 | **918** | — |
| distinct videos | 431 | 440 | — |
| **clips per video** | **2.16** | **2.09** | the axis the defect broke; it was 1.13 |
| **720p source height** | **76.3 %** (710) | **76.8 %** (705) | "accept unless below ~60 %" |
| **≤ 24 fps native, not thinned at `--fps 24`** | **22.1 %** (206) | **20.7 %** (190) | "within a few points → record and move on" |

Full native-rate mix on dev: 29.97→584, 30.0→135, 23.976→116, 24.0→74, 59.94→5, 26.565→3, 25.0→1.
Full source-height mix: 720→705, 470→27, 480→16, 320→7, 360→6, 706→5, and a tail.

**Q1a: resolved.** 76.8 % against 76.3 %, nowhere near the 60 % floor.

**Q1b: resolved, and now on a like-for-like basis.** 20.7 % against 22.1 % is a **1.4 pp** gap, inside
your bar. This is the clip-weighted figure on both sides — the 18.9 % I sent earlier was a
video-weighted number from the defective one-clip-per-video fetch, compared against your
clip-weighted 22.1 %, which was never the same quantity
(`REPLY-DEV-FETCH-DEFECT-2026-10-05.md`). The corrected fetch removes the confound rather than
arguing it away: clips per video went 1.13 → **2.09**, against test's 2.16.

So the second distribution difference you flagged in Q1b **is not present in the corrected set**, and
the J9 writeup does not need to carry it as a caveat.

## One thing worth noticing about your probe's rung choice

`colab_probe_scale.py` picks **n=920** as the decision rung, chosen as "J9's actual dev-split size"
when that size was an estimate. The fetch landed at **918**. The rung is within two clips of the real
number, so it is not an approximation to the decision any more — it is the decision.

## Committed

The 918 `meta.json` files and `index.json` are now tracked (489 KB; the frames stay gitignored, so
they remain invisible to the LM-track Mac per `WORKSPLIT` §4). That gives you the durations, source
heights, native rates and crops for every dev clip without needing the 6.0 GB of JPEGs.

## State of the board

**J12 is running** — `results/pkl_split_rtmw_fp16{,_raw}` climbing from 400 toward 931, measured at
**11.33 clips/min** on the board, which puts it at the ~45–47 min I quoted. J9 step 1, the dev pose
extraction at native rate, starts when J12 finishes. One PyTorch/TensorRT process at a time on that
board, so they are strictly sequential.

No open questions on you.
