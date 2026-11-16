# Presentation proposal — not the deck

*2026-10-12. A plan to agree on before any slide gets made. Assumes ~12 min + Q&A; the timings scale.*

## The one structural decision: open with the demo, don't close with it

Standard instinct is to save the live demo for the finale. Do the opposite.

- If the network fails, you find out in minute two with ten minutes left to recover, instead of
  ending the talk on a failure.
- It reframes everything after it. Once the audience has seen it work, the rest of the talk is "here
  is how we made that fit in 15 W" rather than a promise being built toward.
- The professor asked for the real-device proof. Leading with it answers him before he has to ask.

## Proposed arc

| # | Section | min | What carries it |
|---|---|---:|---|
| 1 | **Live demo** | 2 | Sign a sentence into the phone, English appears. One line: everything after this is how it fits in 15 W. |
| 2 | **Pipeline** | 1.5 | One diagram. The pose stage never passes pixels on — only 69 of 133 keypoints reach the decoder. |
| 3 | **The baseline reproduces** | 1.5 | Four-row table bracketing 22.5–22.7 BLEU-4, plus the prefix trap: scored raw, the authors' own predictions give **58.7** BLEU-4. A 36-point error from a text prefix. |
| 4 | **The structural finding** | 3 | The opposite-cost-shapes table. Pose charges per frame, the LM per sentence; pose is ~75 % of system energy. So **frame rate is the lever and beam width is not**: 103.5 J per BLEU-4 point for 30→24 fps against 1.0 J for beam 4→2. |
| 5 | **Compression, accuracy axis on every row** | 2.5 | FP16 pose; vocabulary pruning 587.7 M → 243.6 M params for −12.1 % system energy; frame rate. Each with its interval, not just its point estimate. |
| 6 | **What we got wrong** | 1.5 | The FP16 sign flip (−0.13 at n=30, +0.148 at n=931, both inside each other's intervals) and the adaptation null whose interval *excludes* the effect that motivated it. |
| 7 | **The frontier** | 1 | `results/frontier.png`. The plot the whole project is built around. |

Section 4 is the intellectual contribution and should get the most airtime. "Calling beam width the
frontier was true of the language model alone and false of the system" is the sentence to land.

Section 6 is not a confession, it is the strongest differentiator in the talk. Most course projects
present point estimates. Showing a sign that flipped under more data, and a null stated as a null,
is what separates measurement from storytelling. Keep it.

## Demo mechanics

- **Hold-to-record** on the phone. The signer holds a button while signing, which gives an exact
  sentence boundary and removes a heuristic we would otherwise have to tune and defend.
- **Rehearse one sentence** until it is reliable. Offer a second, audience-suggested one only if the
  first lands and time allows.
- **Pre-recorded backup video**, which we owe as a C11 deliverable anyway. If the network misbehaves,
  play it and say plainly that the network failed and this is the same pipeline recorded earlier.
  Doubling the deliverable as the fallback means the demo carries no schedule risk.
- **If the board is in the room**, a second screen with `tegrastats` — watts ticking while a
  translation runs. That is the single most on-point thing we can show for this course.

## Three honesty rules for the deck

These are not optional; the report is built on them and the slides must not undercut it.

1. **Every number on a slide traces to a file in `results/`.** No round numbers invented for a slide.
2. **The demo's latency is not quoted as a measurement.** It runs pose during capture, which is the
   streaming overlap the report explicitly flags as *unmeasured* for contention. Show it, don't
   number it.
3. **The phone is not the measured platform.** iOS gives no per-rail power reading. Say once, out
   loud: the phone is the camera and the screen, the 15 W Jetson is where the numbers come from.
   Said plainly, that reads as rigor. Left unsaid, it looks like a gap.

## Split

| Who | Sections |
|---|---|
| Atisri | 1 (drives the phone), 3, 5 (pruning), 6, 7 |
| Tushar | 2, 4 (pose cost shape), 5 (FP16 pose), `tegrastats` screen |

Section 4 is the hinge and depends on both tracks' numbers, so it needs one rehearsal together
rather than two separately prepared halves.

## What has to be true first

| Needed | Owner | State |
|---|---|---|
| Board-side demo server, port reachable from a WireGuard peer | pose track | asked, `ASK-IOS-DEMO-BOARD-2026-10-12.md` |
| Phone sends video + person box (iOS Shortcuts) | LM track | not started, ~half a day |
| 2–3 live phone clips sanity-checked on the board | pose track | asked, item 3 |
| Backup demo video | both | outstanding C11 item |
| Rehearsal of section 4 together | both | not scheduled |

## Open question before slides

**How many minutes do we get, and is there a required format?** The arc above is sized for ~12 min.
At 8 min, sections 3 and 7 merge and section 6 drops to one line. At 20 min, section 4 expands into
the per-stage energy breakdown rather than everything growing evenly.
