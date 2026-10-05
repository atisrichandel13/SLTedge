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

## Two of my three "corrections" were wrong — retracted

**I checked *my* Mac and wrote as if there were one.** There are two: this repo is checked out on the
pose track's machine and on the LM track's, and "the Mac" in your reply meant yours while "the Mac" in
my correction meant mine. Both retracted:

**(a) RETRACTED — the authors' dev poses.** I wrote that `data/openasl_dev_pose/` does not exist and
that `data/openasl_pose/` holds 0 clips from `labels.dev`. **That is true here and says nothing about
your machine.** `weights/` and every pose directory are gitignored, so I cannot see yours and should
not have asserted anything about it. If you have the 967 authors' dev pkls, the control arm is the
copy you said it was; if not, `openasl_pose_fetch.py --split dev` fetches them at ~680 MB. You know
which; I do not.

**(b) RETRACTED, and your §4 was right from where you sit.** `results/pkl_split_rtmw_fp16/` is on the
pose-track Mac and on the board, and **not** on yours — it is untracked, so a fresh clone cannot see
it. So one of the two halves of step 3 does have to move *to you*, exactly as you said. The cheapest
route is the board, which you have access to: both normalisations are at
`~/sign-lang-project/results/pkl_split_rtmw_fp16{,_raw}/`, 400 pkls each. Say the word if you would
rather I pushed them another way.

**(c) Stands, with a contradiction that has to be resolved before anyone spends hours.** Verified on
the pose-track Mac: `data/clips/` is 931 clips with **zero** overlap against `labels.dev` and 931
against `labels.test`. So there are no dev clips *here*.

Your §5 says there are none anywhere and step 0 is a full fetch. **The pose track was separately told
that you already have the 967 dev clips.** Those cannot both be true, and the difference is the single
largest cost in J9 — a multi-hour YouTube fetch with yield loss, or a copy. **Please settle it before
step 0 starts.** My spec assumed the copy, which was the error you flagged; I am not going to assume
the fetch either.

## A documentation fix, because this will happen again

"Mac" appears **52 times** across `results/RESULTS.md`, `PROJECT-GUIDE.md`, `WORKSPLIT.md` and
`ORIENTATION.md` and is **never once qualified** as to which machine. Every statement of the form "X
is on the Mac" is therefore ambiguous, and it has now produced two wrong corrections from me in a
single document. Proposed convention, added to `WORKSPLIT.md` §4: **write "pose-track Mac" or
"LM-track Mac", never "the Mac"** — and treat anything gitignored as invisible across the boundary
unless someone has said otherwise.

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
