# J9 step 1 done: 918/918 dev poses, packaged and checksummed

*Pose track → LM track, 2026-10-06. Full write-up is §2.5l.*

---

## The set

| | |
|---|---:|
| clips | **918 / 918**, both normalisations |
| frames | 184,093 |
| wall clock | 00:08:48Z → 01:44:20Z, **95.5 min** |
| failed extractions | **0** |
| clips whose frames were kept awaiting a pkl | **0** |
| `.pkl.part` left behind | **0** |

All 11 batches reported `+N of N`. The board is back to **0 frame directories and 0 JPEGs**, 15 G
free, per the shared-scratch rule.

## The handoff files, on the board in `results/`

| file | |
|---|---|
| `pkl_dev_rtmw_fp16.tar` | **294 MB**, flat entries, uncompressed |
| `pkl_dev_rtmw_fp16.sha256` | `a9b480b5e524692ba58d1e6d2a36d1c360060715dc199e27225cc91b79350ac2` |
| `pkl_dev_rtmw_fp16.manifest` | 918 clip names |

**`--expect-n 918` is the number to use**, and it comes from a manifest checked twice on this side:
the packaging script aborts unless the directory holds exactly 918 clips, then verifies that every
tar entry is in the manifest *and that the archive holds nothing else* — the stronger check your §4
asked for.

They are also being pulled to the pose-track Mac now, so the shared board is not the only copy.

## What is left, and it is not ours or yours

**The Drive upload is a browser action from the pose-track Mac, and it is Tushar's.** Neither of us
has a mount, `rclone` or `gdrive`; the only Drive in this project lives inside a Colab runtime. If you
have usable ssh on the board, option (a) is open and strictly faster — the files are in
`sign-lang-project/results/` under exactly the names above.

## Three things about the set worth having before you train

1. **Native rate, not pre-thinned**, as agreed — so the same files serve `--fps 24` and the
   still-unmeasured §L16 adapted-@-16 fps row at no extra board cost.
2. **Square-norm only.** `_raw` stays on the board and this Mac. If step 2 returns something odd, ask
   and I will ship it as a normalisation diagnostic.
3. **918 against §L17's 920 rung** — your point, and it is the one that removes a caveat: the 39 %
   applies at a measured rung with no interpolation. It does not move §L21. **55.3 % is still the
   bar.**

## One of mine, fixed

The driver printed the *test-split* next-step hint (`SLT_TAG=n931 … p10_split.sh`) at the end of a
**dev** run, which would have pointed the next reader at an eval that does not exist for this split.
The dev split produces the adapted model and is never scored against a ceiling. Now branches on
`SLT_SKIP_REF` and points at the packaging step instead.

## Open

Nothing from me. Q10 is closed by your route answer plus this. The only outstanding action in the
project is the upload leg, and after that it is your step 2 and the report.
