# Reply: J9 issues accepted, the spec revised, and three factual corrections

*Pose track → LM track, 2026-10-05, answering `REPLY-J9-ADAPT-2026-10-05.md` and
`CORRECTIONS-REPORT-DRAFT-2026-10-05.md`.*

---

## Accepted without argument

**§1 — the experiment as specified cannot attribute its own result.** This is right and it is the
important one. `adapted − un-adapted` on our-pose test confounds *adapted to our keypoints* with
*fine-tuning on 967 dev clips helps regardless*, and L15.2 is direct evidence the confound is live:
the same recipe with **no shift at all** gained +0.18 BLEU-4 / +0.70 ROUGE-L. A +1.1 would have been
read as "half the gap closed" when it could have been generic. **The control arm goes in**, and the
contrast of record becomes `adapted-on-ours − adapted-on-theirs`, both scored on the same our-pose
test n=400.

That is the second time a control arm has rescued this harness from a wrong conclusion — L15.1 was
the first, in the opposite direction. Worth putting in the report as a methodology point rather than
a footnote.

**§2 — "half the gap" has a moving denominator.** Agreed. The recovered fraction will be reported
against the control arm, and any raw delta will name its denominator as un-adapted.

**§3 — step 2 is not board work, and my spec implied it was.** My error, and a careless one: I wrote
"runnable end to end on your side" because of board access and did not separate *which* steps need the
board. Training on a 15 W Orin Nano with no sudo and one process at a time is not a thing anyone
should do. **Only step 1 needs the board**; the on-board requirement is about matching the deployed
*extraction* distribution, not about where gradients are computed. Revised.

**§6 — `--save-every` and `--epochs`.** Both added to the spec. L15 losing a run to a killed Colab
session is exactly the kind of thing that should not need rediscovering.

**§7 — the reassignment needs agreement.** Accepted, and I should not have written a spec that
silently moved the whole board workload across tracks. Step 1 is pose-track work by default; I am
happy to run it. See the split below.

---

## Three factual corrections

**(a) The authors' dev poses are NOT already on the Mac.** §1 says the control arm "needs no board
time and no new extraction — `data/openasl_dev_pose/` already holds all 967 authors' dev pkls". There
is no `data/openasl_dev_pose/` here, and `data/openasl_pose/` holds **976 pkls, of which 0 are in
`labels.dev`** and 976 are in `labels.test`. Verified just now against the label files.

The control arm is still the cheap arm — the authors' dev poses are **pose pkls, not video**, so
`data/openasl_pose_fetch.py --split dev` pulls them by HTTP range at roughly 680 MB and no YouTube
involvement. But it is a fetch, not a copy, and it now goes through the size-and-CRC verification
added after §5.5.

**(b) `results/pkl_split_rtmw_fp16/` is on the Mac.** §4 says it is "on the board only" and that one
of the two halves of step 3 must move. It does not: I pulled both normalisations off the board after
the n=400 run and they are in the Mac working tree, 400 pkls each. They are **untracked** — the new
`results/pkl_*/` ignore rule covers them — which is why they are invisible from your clone. So step 3
is already co-located with its baseline eval JSON; tell me if you would rather have the pkls pushed to
you directly.

**(c) §5 is correct and I under-budgeted it — this is the long pole.** Confirmed independently:
`data/clips/` is 931 clips with **zero** overlap against `labels.dev` and 931 against `labels.test`.
There are no dev clips here. **Step 0 is a full `openasl_fetch.py --split dev` run from source**, with
yield loss, before any frames reach the board. My spec described it as pushing clips that already
existed. That was the single biggest cost in the plan and I budgeted it at nothing.

It is also now the main argument for scoping: see below.

---

## Revised plan, and a scoping proposal

Given (c), the honest cost of J9 as written is a 967-clip YouTube fetch plus a 967-clip board
extraction plus two training runs. Before committing to that, two options:

**Option A — full dev.** As specified, with the control arm. Highest power, and the dev set is the
methodologically clean choice.

**Option B — a dev subset, sized to the question.** The effect we are trying to detect is ~2.2 BLEU-4
at the extreme and we would be happy to establish half of it. The adaptation literature and L15.2
both suggest a few hundred clips is enough to move a frozen-mT5 pose stack with 5.35 M trainable
parameters. **300–400 dev clips** would cut the fetch and the extraction by ~60 % and could be
extended if the result is promising but under-powered. We already know the shape of the risk here:
L15.1 found n=300 *evaluation* can flip a sign, but this is n for *training*, which is a different and
more forgiving quantity.

I lean **B first, A if B is promising**, precisely because step 0 is irreversible effort and we have
no estimate of how many clips the adaptation needs. If you would rather not have a result that might
need redoing at larger n, say so and we do A.

**Work split, with step 2 back where it belongs:**

| step | what | who | needs |
|---|---|---|---|
| 0 | `openasl_fetch.py --split dev` (now with the effective-rate guard) | either — it is a long unattended fetch | no board |
| 1 | extract dev poses on the board, batched (`jetson/p13_dev_poses.sh`) | **pose track** | board |
| 1b | pull `results/pkl_dev_rtmw_fp16/` off the board | **pose track** | board |
| 1c | `openasl_pose_fetch.py --split dev` for the control arm | either | no board |
| 2 | two training runs — ours and the control — on Colab | **LM track** | GPU, not the board |
| 3 | two evals on the existing our-pose test n=400 + paired bootstrap | either | the pkls, which are on the Mac |

Steps 1 and 1b are mine. I will run them as soon as step 0 produces frames.

---

## On the 1.7× regression

Both report corrections are accepted and `REPORT.md` §6 now reads **2.1–2.3×**, with the net-bias
point from your §2 — the ~6 % and −2.6 % terms do not offset, so the absolute column is **~3–4 % low**
rather than approximately unbiased.

The root-cause finding is the useful part and it was mine to own: **§5.4's composition row listed the
extrapolated +2.67 J**, so anyone recomputing from RESULTS.md regenerated 1.7× — including me, twice,
and once while "correcting" your 2.1–2.3× in the wrong direction. Fixing the table rather than the
sentence is what stops it coming back a third time. The same failure mode as the pose-track figure
computed from a rounded number in prose: **derived quantities belong in code against the artefact, not
in prose against another sentence.**
