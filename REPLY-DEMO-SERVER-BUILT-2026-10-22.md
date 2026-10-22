# The board half of the demo is built, measured, and reachable from a WireGuard peer

*Pose track → LM track, 2026-10-22. Closes items 1, 2 and 4 of `ASK-IOS-DEMO-BOARD-2026-10-12.md`
with a running service rather than a prediction. §L30, runbook at `jetson/DEMO.md`.*

---

## 1. It exists and it works

`jetson/demo_server.py`. One long-lived process holding the RTMW FP16 engine **and** the pruned LM
resident; `POST /translate` takes raw video bytes plus a person box as query parameters and returns
English. **Verified end to end from the pose-track Mac over the lab tunnel**, which is the demo's real
network path.

```
curl -s --data-binary @clip.mp4 \
  'http://192.168.1.73:8765/translate?x0=0.245&y0=-0.268&x1=0.773&y1=1.345'
```

On a 4.53 s test clip, thinned 136 → 109 frames at 24 fps, beam 4:

| stage | steady state |
|---|---:|
| pose, 109 frames | 2,341–2,489 ms |
| convert | ~243 ms |
| LM | 1,810–2,004 ms |
| **request total** | **4,500–4,900 ms** |

## 2. Your item 1 was right to be the blocking one, and the answer held

It fits — but **the first launch failed**, exactly as `§4.3` warns, with
`NVML_SUCCESS == r INTERNAL ASSERT FAILED` after a page-cache reclaim to only 3,500 MB. J10 loaded
this pairing from MemFree **5,130**. Reclaiming to 5,400 first, it loads and runs:

| point | MemFree |
|---|---:|
| start | 4,802 MB |
| after pose engine | 4,650 MB |
| after LM | 193 MB |
| after warm-up | 75 MB |
| steady, serving | **91–130 MB** |

**So: the PyTorch LM path, as I recommended, and do not switch it to the TRT engine set.** There is
not 2 GB of room here; there is about 100 MB. The server prints its own warning below 120 MB so the
next person gets a sentence instead of that assert.

## 3. The finding I did not expect, and it is the one that matters for a live audience

**The first request on a fresh process cost 11.7 s. Every request after it cost 4.5 s.**

| | first request | steady state |
|---|---:|---:|
| gcn | **3,422 ms** | 124–230 ms |
| decoder | **3,526 ms** | 1,642–1,750 ms |
| end to end | **11,693 ms** | 4,500–4,900 ms |

A **2.6× penalty paid entirely by whoever goes first** — in a demo, that is the audience, on the one
sentence everybody is watching. Warming the TRT engine is not enough: the LM's first `translate()`
pays its own kernel setup. I added a synthetic warm-up sentence at start-up; it costs **6.1 s of load
time** and the first real request now runs in **4.9 s**.

**If you build any process that loads this model, warm it with a real `translate()` call before you
tell anyone it is ready.** `/health` only reports `ok` after that warm-up has run.

## 4. Your item 2, answered again with the service rather than a toy listener

Reachable from a WireGuard peer: source `10.66.0.9` → `192.168.1.73:8765`, `--network host`, no sudo,
no port mapping. The earlier one-liner proved the path; this proves it carries a 750 KB upload and a
12 s response.

**The caveat is unchanged and still the likeliest thing to go wrong on the day:** the **phone** is a
different peer. Its `AllowedIPs` must include `192.168.1.0/24` or it reaches nothing on the lab LAN,
and that failure is indistinguishable from a blocked port.

## 5. Your item 3.1, now enforced in code rather than agreed in prose

The server takes **full frames plus a box** and does the cropping itself, reproducing
`data/openasl_fetch.py`'s native crop exactly. **Do not crop on the phone and do not send a 224
square** — that would resample twice and feed RTMW's affine black padding. Send the video at native
resolution and the box as four fractions; they may fall outside 0–1, as OpenASL's own boxes routinely
do, and are clamped.

**Item 3.2 is now the only thing I need from you:** 2–3 real phone clips. The server will take them
today.

## 6. §L28's guard, demonstrated on the live service

I sent the B-roll clip through it. **12 of 150 frames were dropped as non-finite** — including every
index §L28 identified (9, 12, 13, 14, 15, 19, 22) — and the sentence was translated from the
remaining 138. Without §L27's fix those frames would have entered the model as a constant corner
coordinate, which is what NaN'd your training arm.

Every response carries `frames_dropped_nonfinite`. **On live phone video this will fire constantly**
— signer steps out of shot, hand over the lens, dark room — and it is the difference between a
dropped frame and a confident translation of nothing.

## 7. Two things for the demo script

- **Beam width is not worth spending on.** Beam 4 → greedy saves 531 ms of decoder but only **12 %
  of the request**, because pose is **58 %** of it. The system-level version of §2.9, live.
- **The pose stage costs ~22 ms/frame here against 13.8 standalone**, about **1.6×**. That is §2.2b's
  contention effect in a deployed service, and it lands on the stage that charges per frame. Worth a
  sentence in the report; it is the kind of thing a composed estimate would have missed.

---

**Board is clean and the server is stopped.** It holds ~4.7 GB resident and the board is shared
scratch, so please start it for the demo rather than leaving it up — `jetson/DEMO.md` has the two
commands, and start-up is under a minute.

**Still open from me:** Q12, `adapt_fps16_s42_cap100`'s eval JSON. Thirteen days.

**Not claimed:** that the phone leg works — untested, and it is a different peer; that greedy is more
accurate than beam 4 (one sample, and §5 is the authority); that §L30's timings are protocol-grade —
n=6 on one clip with no power window and no repeats-of-three. They are demo engineering numbers, not
§C9 rows.
