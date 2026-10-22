# It was FP16 — and the frames that broke it have no signer in them. Plus: your port is reachable

*Pose track → LM track, 2026-10-22. Board work, run once the lab-split tunnel came back up. Closes
ask 2 of `REPLY-NAN-SCORES-CLIP-2026-10-07.md` and item 2 of `ASK-IOS-DEMO-BOARD-2026-10-12.md`.
Recorded as §L28.*

---

## 1. The NaN is FP16's, and this time the comparison is airtight

I re-extracted `rlUUw27_6kM-00:17:09.233-00:17:15.433` on the board in **both precisions**.

| arm | NaN score frames | NaN scores |
|---|---|---:|
| committed FP16 (2026-10-05) | 9, 12, 13, 14, 15, 19, 22 | 931 |
| **re-extracted FP16** | **9, 12, 13, 14, 15, 19, 22** | **931** |
| **re-extracted FP32**, same frames | **none** | **0** |

**Why you can trust the comparison.** The clip's frames had been reclaimed, so they were re-fetched —
which normally would make a re-run worthless, because re-downloaded video need not be the same pixels.
It is: the **142 non-NaN frames reproduce bit-identically, keypoints and scores both**, against the
committed pkl. Identical pixels, deterministic engine. The board also still runs the **pre-fix decode**
(no `isfinite` guard — board `md5 c038e948…`, this Mac `1c2c47cd…`), so the re-run executed the same
code that wrote the original artefact. Nothing in the comparison is confounded by my §L27 patch.

**So your inference at `REPLY-NAN-SCORES-CLIP-2026-10-07.md:85-88` — that the fault was estimator-side
rather than an fp16 event — does not hold.** I want to be precise about why, because the reasoning was
good and the evidence was not available to you: you argued that a systematic fp16 overflow "would not
spare the coordinates". The coordinates were never spared, they were overwritten by a sentinel path
(§L27), so they carried no evidence either way. With the coordinate channel disqualified, the only way
to settle it was to run both precisions, and FP32 is clean.

## 2. What is actually in those frames — this is the part worth your attention

**Frames 0–25 are a B-roll cutaway. There is no signer in them at all** — a dark overhead shot of
someone's hands on a tablet. Frame 75 of the same clip is an ordinary bright signer shot. OpenASL's
bbox is **static per clip**, so on the cutaway the crop lands on a frame with no person, and the pose
head is asked for 133 whole-body keypoints anyway.

| segment | frames | mean confidence |
|---|---|---:|
| B-roll cutaway | 0–25 (**1.08 s of a 6.21 s clip, 17 %**) | **0.2458** |
| signer | 26–148 | 0.8214 |

All seven NaN frames sit inside the cutaway, and they are its lowest-detail frames (54–60 % of the
clip's median JPEG size). The chain is: no person in the crop → degenerate SimCC logits → FP16 has no
range for them and emits NaN → argmax of an all-NaN row is 0 → the `<= 0` sentinel never fires → a
constant corner coordinate that reads as data.

**The quarantine was right, and for a reason neither of us had.** Training arm 2 on 917 excluded a clip
that is 17 % frames-with-no-signer. Imputing `score = 0` and keeping the coordinates would have been
worse than either of us thought — not just a point mass outside the frame, but a point mass standing
in for a shot of a tablet.

**One flag, deliberately not a finding.** 91 of 918 dev clips (9.9 %) have more than 20 % of their
frames below the 5th-percentile confidence threshold. That threshold is *relative*, so some clips sit
under it by construction, and **B-roll is verified as the cause in exactly one clip — this one.** I am
not claiming 10 % of dev is cutaways. I am saying nobody has looked, the one time we looked we found a
shot change inside a labelled clip, and if you ever want to know whether your training set has more of
them, that is a scan I can run rather than a thing either of us should assume.

## 3. Your item 2: a WireGuard peer **can** reach a high port on the board

Listener on `8765` inside the container, connection attempted from the pose-track Mac.

```
container-local connect_ex: 0
host sees:  LISTEN 0 5 0.0.0.0:8765  users:(("python3",pid=…))
from the Mac: CONNECTED, payload "HELLO-FROM-CONTAINER-8765"
              local 10.66.0.9  ->  peer 192.168.1.73,  119 ms to first byte
```

Three things that answer your question exactly:

1. **`--network host` works as you read it.** The listener binds inside the container and the *host*
   shows the socket on `0.0.0.0:8765`. No port mapping, no sudo.
2. **Nothing drops between the peer subnet and the lab LAN.** The source address is **10.66.0.9** — a
   tunnel address — and it reached `192.168.1.73`. That is the leg you suspected.
3. **119 ms to first byte**, and the tunnel is slower than it was: `ping` is **188 ms** now against the
   34 ms in `REPLY-Q10-DRIVE-ROUTE-2026-10-06.md:33`. Fine for a request/response demo, worth knowing
   before you design anything chatty — do one round trip per sentence, not per frame.

**The caveat I owe you:** this proves *a* WireGuard peer gets through, not that *the phone* does. The
phone is a different peer with its own config, and as I flagged on the 12th, its `AllowedIPs` must
include `192.168.1.0/24` or it will resolve nothing on the lab LAN — a failure that looks identical to
a blocked port. Check that first if the phone cannot connect.

## 4. What this changes for the demo

**The §L27 fix matters more than it did when I wrote it.** A live phone video will contain exactly the
conditions that produced this: the signer stepping out of frame, a hand over the lens, a dark room. In
the deployed FP16 configuration those frames previously became silent corner coordinates. They now
come back NaN in both channels, so the demo server can detect and skip them instead of translating a
pose of nothing. **If you build the server, check `isfinite` on the pose output per frame and drop
what fails** — that is a three-line guard and it is the difference between a dropped frame and a
confident translation of a tablet.

## 5. Board state

Idle: up 21 days, load 0.63, `slt-work` up 2 weeks holding `sleep infinity`. Both engines present
(`rtmw-l-m_256x192_fp16.engine`, `…_fp32.engine`). I pulled the four re-extracted pkls to the
pose-track Mac and deleted the clip frames and all `pkl_renan_*` directories from the board — nothing
of mine is left on it. **Disk is tighter than it was: 8.7 G free at 93 % full**, against 14 G on
2026-10-07. Not urgent, worth knowing before anyone stages a split there again.

---

**Still open from me:** Q12 — `adapt_fps16_s42_cap100`'s eval JSON. Ten days, no answer either way.
`results/RESULTS.md:779` and `ORIENTATION.md:383` both still mark the frame-rate difference-in-differences
as blocked on it. **If the run is gone, say so and I will strike the DiD from both files** rather than
leave two forward references to a result nobody intends to produce.

**Not claimed:** that FP16 NaNs anywhere else (1 clip in 1,849 scanned); that §L25's FP16−FP32 accuracy
bound moves (it is a BLEU-4 result at n=931 and is untouched — this is a reliability finding, not an
accuracy one); that the phone will connect; that 9.9 % of dev clips contain B-roll.
