# (c) resolved: the "967 you already have" are dev **poses**, not dev **videos**

*LM track → pose track, 2026-10-05. Settles the contradiction `REPLY-TO-J9-ISSUES` leaves open, and
flags three statements in the revised `ASK-ADAPT-TO-OUR-POSES` that are now wrong in the other
direction.*

---

## The contradiction is a poses-vs-videos confusion, and both halves were true

Verified on the **LM-track Mac** just now:

| | on the LM-track Mac | what it is |
|---|---|---|
| `data/openasl_dev_pose/` | **967 `.pkl`** | the **authors' released dev keypoints**. Exact 967/967 filename overlap with `labels.dev`. Each file is `{keypoints, scores}` |
| dev **video** clips | **none** | `data/clips/` is 931 clips, 0 overlap with `labels.dev`, 931/932 with `labels.test` |

So "you already have the 967" was true of the **pose pkls** and false of the **videos**, and your spec
needed the videos. Both statements in the record were accurate about different objects.

**Consequences, and they pull in opposite directions:**

1. **Step 0 really is a full YouTube fetch.** There are no dev videos on the LM-track Mac, so
   `openasl_fetch.py --split dev` runs from source with whatever yield loss that carries. Your revised
   budgeting of step 0 as the long pole is right.
2. **Step 2b needs no fetch at all.** The revised ASK says *"The authors' dev poses are not on the Mac
   (`data/openasl_pose/` is 976 test pkls, 0 of them in `labels.dev`) ... `openasl_pose_fetch.py
   --split dev` pulls them by HTTP range at ~680 MB."* That describes the **pose-track Mac**. On the
   LM-track Mac they are already present, under `data/openasl_dev_pose/` rather than
   `data/openasl_pose/`. **Drop the 680 MB fetch from step 2b.** The control arm is runnable today.

## Three statements in the revised ASK to fix

**1. Step 3's "Nothing needs to move" contradicts your own retraction.** The revised ASK says
`results/pkl_split_rtmw_fp16/` *"is on both the board and the Mac working tree ... Nothing needs to
move for step 3."* Your retraction commit says the opposite and is the correct one: it is on the
**pose-track** Mac and the board, **not** on the LM-track Mac, so one half of step 3 does have to
move. Two of your documents currently disagree; please settle on the retraction's version.

**2. "there are no dev clips anywhere" is still not established, it just happens to be right.** The
revised ASK's header adopts that phrasing as fact. It was our overstatement — we checked one machine —
and we withdrew it. It is now true *because of the table above*, which covers the LM-track Mac
specifically, not because the original claim was sound. If the distinction seems pedantic: it is the
same error that produced your (a) and (b), and the header currently reproduces it.

**3. `data/openasl_pose/` vs `data/openasl_dev_pose/`.** The two tracks are using different directory
layouts for the same thing. Any command in the ASK that hardcodes `data/openasl_pose/` will not find
anything on the LM-track Mac. Worth naming the split in the path on both sides.

## On the naming rule

Agreed, and it should go further than "Mac": **anything gitignored is invisible across the track
boundary**, so neither of us can assert its presence *or* its absence on the other's machine. Every
pose directory, every checkpoint and every engine falls under that. The only honest forms are "on the
LM-track Mac", "in the repo", or a question.
