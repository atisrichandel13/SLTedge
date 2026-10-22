# Ask: board side of the live iPhone demo

*2026-10-12, LM track → pose track.*

The professor has asked for the demo to run on a **real mobile device against live data**, not a
recorded dataset clip. The shape that needs the least new work is the P14 thin client already in
`WORKSPLIT.md:66`: the phone captures, the board translates, the text comes back.

Phone side is ours. Four things below are board-side or network-side and are yours. One item in your
column should be **dropped**, not built.

```
iPhone (Shortcuts, hold-to-record)  --WireGuard-->  jetson-lpcv-03:<port>
    video + person box                                RTMW FP16 -> pose_to_unisign -> unisign_infer
                                     <--------------  English text
```

---

## 1. Can one long-lived process hold the pose engine and the LM resident together?

This decides the architecture, so it is the blocking one.

A demo server cannot reload per sentence — pruned LM load alone is **29 s** (§3.3). So both the RTMW
FP16 engine and the LM have to stay resident across requests. Two measurements say that is not
obviously free on 8 GB:

- **§L8.2** — the mT5 TRT engine set resident takes **2.07 GB of MemFree**, and deserialisation
  failed with `Cuda Runtime (out of memory)` until the PyTorch reference (1.03 GB) was freed *and*
  the page cache evicted.
- **§2.5 / M1** — end-to-end with the pose **FP32** engine dies partway through LM decode, out of
  memory in the KV-cache concat. Only FP16 fits.

**Ask:** does a single resident process holding RTMW FP16 + the LM actually fit, measured, and what is
the peak? If it does not, what is the fallback — load pose, release, load LM per sentence — and what
does that cost per sentence? A 29 s pause per sentence is not demoable, so if that is the answer we
need to know now rather than on the day.

## 2. Is inbound TCP on a high port reachable from a WireGuard peer?

`jetson/run.sh:33` already runs `--network host`, so a listener inside the container binds the board's
LAN address with no port mapping and no sudo. The phone now has a WireGuard config.

**Ask:** does a listener on a high port (say 8765) inside the container accept a connection from a
WireGuard peer, or is something between the peer subnet and `192.168.1.73` dropping it? You set up the
tunnel, so this is a question only you can answer. A one-liner listener plus one connection attempt
settles it.

## 3. What crop should the engine receive for a live phone video?

**There is no detector in this project.** `task1_rtmpose/03_infer_frames.py:11` states it, and every
box we have ever used came from OpenASL's published `bbox-v1.0.json` (`data/openasl_fetch.py:60`),
with `openasl_square_crop` producing a fixed **224 square**. A phone video has no such box, and RTMW
is top-down.

You do **not** need to add a detector. We will derive the person box on the phone with iOS Vision and
send it alongside the video, so the board side still receives a box the way it does today.

**Ask, two parts:**

1. Confirm the exact crop the engine should receive — i.e. that reproducing `openasl_square_crop`'s
   224 square from a phone-side box is right, and not some other geometry you use on the board.
2. One board run on **2–3 live phone clips** once we can send them, to confirm the output is sentences
   rather than garbage. Live phone framing is a crop distribution nothing in `RESULTS.md` covers, so
   this is unmeasured territory and we should find out before an audience does, not in front of one.

## 4. Can `tegrastats` run alongside the demo server?

The strongest thing we can put on screen for this course is watts ticking while a translation happens.

**Ask:** confirm running `tegrastats` concurrently with the demo server is fine on the shared board and
does not perturb it, and say what sample interval you would use.

## 5. Drop P12, do not build it

`WORKSPLIT.md:64` has P12 as the hands-return-to-rest segmentation heuristic for the live demo, not
started. **It is not needed.** The phone will use hold-to-record — the signer holds a button while
signing — which gives an exact sentence boundary for free and removes a detector we would otherwise
have to tune and justify. Please take P12 off your list rather than spending time on it.

(Separately: `jetson/p12_hybrid_hands.sh` is the hand-swap ablation, not this. The filename collides
with the P12 row and reads as if the heuristic exists. Worth renaming so nobody concludes it is done.)

## 6. If the board were permitted to travel, could you bring it?

The board stays in the lab by default and the demo is designed to work that way — phone on cellular,
over WireGuard, board untouched. So this is not a dependency.

But if the professor permits bringing it, the demo gets materially better: the audience sees the 15 W
board on the table instead of taking a tunnel on faith, and the `tegrastats` reading in item 4 becomes
a box in the room rather than a number from a machine nobody can see.

**Ask:** is that physically and administratively possible from your side — and does it need a slot
booked so other users of the board are not cut off that day?

---

## What we are not asking for

No new pose model, no detector, no INT8, no retraining, no accuracy run. Items 1–4 are a memory check,
a port check, a crop confirmation with a 2–3 clip sanity run, and a `tegrastats` confirmation.
