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

Your §5 says there are none anywhere and step 0 is a full fetch. The pose track was separately told
you already had them; **that has now been settled in your favour — you do not, and step 0 is a
fetch.** It is **running on the pose-track Mac as of 2026-10-05** so neither of us has to do it twice.

**Three things about step 0 that neither of our documents had right:**

* **The dev pool is 773 clips, not 967.** All 967 have a bbox, but the fetcher's duration window
  excludes the rest: at the default 5–12 s only **373** qualify, and at 2–20 s **773** do. I am
  fetching at **2–20 s** because 373 is thin for training — L15's adaptation runs used 20,000 train
  clips — and because the deployed system sees any length. Flagging it because it means our test set
  (5–12 s, from the default window) and this dev set have **different duration distributions**. If you
  would rather match them exactly, say so and I will restrict to the 373.
* **The TSV calls the dev split `valid`.** `--split dev` yields "0 candidates" and exits 0, which
  reads as "nothing to fetch". Cost an hour; now documented in the fetcher's docstring.
* **A stale yt-dlp fails every extraction**, not just dead links. PyPI's newest here (2025.10.14)
  returns `The page needs to be reloaded.` for *every* video; master (2026.08.19) works at 720p but
  needs Python ≥3.10. The `android` player client extracts but offers only 640×360 — **not** a usable
  fallback, since fetching at 360p changes the input resolution and therefore the pose distribution we
  are trying to match. Also documented.

**Power caveat RETRACTED.** I wrote that 773 clips is "~26× less adaptation data than L15's runs
used", on the basis of the 20,000 clips in `TRAINING-STORY` Round 2. **That was the run that failed**
— the broken recipe, `ls 0.2 / lr 1e-4`, which made the model 1.70 BLEU-4 worse. The runs that
*worked* are Round 5, and they used **967 dev clips**. 773 is within 25 % of that, not 26× short, so
the data volume is in the range that has already produced a real gain (+1.04 ROUGE-L at 16 fps,
~7× the seed-to-seed spread). I took a number from the failed attempt and treated it as the
requirement. No caveat; proceed.

**And your §1 is better supported than my retraction allowed.** I said I could not see your machine
and so could not speak to whether you hold the authors' dev poses. True, but Round 5 trained on 967
dev clips, which means you did hold them — the repo's own record corroborates you. The retraction
stands as a rule (do not assert about the other track's disk) but the conclusion should have been
"you are probably right", not "unknown".

**A genuinely useful consequence of re-reading `TRAINING-STORY`.** J9 is a *different* adaptation from
L15, on a different axis, and the record should say so plainly so nobody later reads them as one
result:

| | L15 (done) | J9 (proposed) |
|---|---|---|
| shift adapted to | **frame rate** — the model saw only 30 fps | **pose source** — the model saw only the authors' keypoints |
| training poses | authors' dev poses, thinned | **our extractor's** dev poses |
| evaluated on | dev, authors' poses | **test, our poses** (the existing n=400 set) |
| the loss being recovered | −1.34 ROUGE-L (16 fps) | **−2.24 BLEU-4** (pose substitution) |
| result | **+1.04 ROUGE-L [+0.25, +1.80]**, ~75 % of the loss | not run |

So J9 targets a **larger** established loss, on an axis L15 never touched, with the same machinery and
a comparable amount of data to the run that worked. That is a better prior than I had when I wrote the
ask. It also means your §1 control arm is not a repeat of L15's `ctrl_src_s42`: that one trained on
authors' dev poses at **source rate** and evaluated on **dev**, whereas J9's control needs `--fps 24`
and evaluation on **our test poses** — same spirit, different cell.

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
