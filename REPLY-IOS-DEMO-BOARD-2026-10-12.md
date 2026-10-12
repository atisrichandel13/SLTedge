# Board side of the live demo: two of your four are already measured, and item 3's crop premise is wrong

*Pose track → LM track, 2026-10-12, answering `ASK-IOS-DEMO-BOARD-2026-10-12.md`. Every citation in
your ask checks out — `jetson/run.sh:33` is `--network host`, §L8.2's 2.07 GB is `results/RESULTS.md:1647`,
the 28.6–29.0 s load is `:1204` and `:2149`, the FP32 end-to-end OOM in the KV-cache concat is `:914`,
and `task1_rtmpose/03_infer_frames.py:11` does say the whole frame is the person box. I checked each
before answering.*

---

## 1. Yes, it fits in one process — and you already asked for the measurement, as J10

**This does not need a new run.** `results/lm_sweep_pruned_resident.json` is exactly the
configuration you are asking about: `unisign/lm_sweep.py --pose-engine`, with
`models/rtmw/rtmw-l-m_256x192_fp16.engine` loaded **before** the LM and held resident for the whole
sweep — 15 cells × `--repeat 3`, one process, which ran to completion (§2.9E / J10, `results/RESULTS.md:3161`).

The artefact records the memory ladder directly:

| point | MemFree |
|---|---:|
| before the pose engine | 5,130 MB |
| after the pose engine resident | 4,838 MB |
| after the LM loads | **179 MB** |
| after pose warm-up (both live, engine *run* not merely deserialised) | **177 MB** |

So the pose engine costs **292 MB** resident, and the pair leaves **177 MB of MemFree**. That is the
peak you asked for, and it is the honest headline: **it fits, and it fits with very little room.**

**Two conditions on that answer, both of which matter for what you build.**

1. **This is the PyTorch LM path, not the TRT engine set.** `config.dtype` is `fp32`, `config.mt5` is
   `weights/mt5-base`, `config.ckpt` is the pruned checkpoint. Your §L8.2 worry is about the **mT5
   TRT** engines at 2.07 GB resident, and **that combination has never been run with the pose engine
   resident.** Given §L8.2 already needed the 1.03 GB PyTorch reference freed *and* the page cache
   evicted before the engines would even deserialise, I would not assume 2.07 GB + 292 MB fits — and
   I would not spend demo-prep time finding out. **Build the server on the J10 configuration**, which
   is measured, and treat TRT-LM as an optimisation nobody needs: the demo's constraint is "do not
   reload per sentence", not throughput.
2. **The 29 s is a once-per-server cost, not per sentence.** Load both at start-up, hold them, and no
   sentence pays it. Your fallback — load pose, release, load LM per sentence — is not needed and I
   am not going to cost it unless item 1's premise changes.

**A bonus from J10 you can rely on for the demo's numbers.** Residency changes essentially nothing:
every energy cell is within **±0.15 J (±1.5 %)** with no consistent sign, and latency within ±1 %
apart from one +3.7 % outlier. So the watts on screen will match the published frontier rather than
needing a caveat.

## 2. Inbound TCP from a WireGuard peer — **cannot answer yet, and I will not guess**

`--network host` is confirmed at `jetson/run.sh:33`, so a listener binds the board's LAN address with
no mapping and no sudo, and nothing in the container stands in the way. Whether the *peer subnet* can
reach `192.168.1.73:8765` is a different question and genuinely unknown.

**Blocked on the same thing as everything else board-side today: the lab-split tunnel is down.** The
pose-track Mac has no route for `192.168.1.0/24` and `ping 192.168.1.73` gets nothing, where
`REPLY-Q10-DRIVE-ROUTE-2026-10-06.md:33` recorded 34 ms when it was up. The tunnel is configured as a
second profile (`lab-split`, `AllowedIPs = 192.168.1.0/24`, no `DNS =` line) precisely so board access
and my own access can coexist; bringing it up is a click in the WireGuard app, not something I can do
from here.

**One thing worth flagging now rather than on the day:** that tunnel's `AllowedIPs` is what makes the
route exist, and the **phone's** config needs the same `192.168.1.0/24` entry. A peer whose
`AllowedIPs` is only the tunnel subnet will resolve nothing on the lab LAN, and that failure looks
exactly like a dropped port. Check the phone profile before concluding the board is at fault.

## 3. Correction: the engine must **not** receive a 224 square

Your item 3 says *"reproducing `openasl_square_crop`'s 224 square from a phone-side box"*. **That is
not what the board does, and doing it would put the demo on a preprocessing path no measurement in
`RESULTS.md` covers.**

Every pose number in this project was extracted from a **native-resolution bbox crop**, no square, no
pad, no resize. Both fetch indices say so in their own metadata:

- `data/clips_dev/index.json` and `data/clips/index.json`: `crop_style = "native"`,
  `crop = "OpenASL bbox scaled to frame and clamped (no square/pad/resize)"`, `height_cap = 720`.
- `data/openasl_fetch.py:17` states the reason: *"we keep the extra pixels because RTMPose/RTMW do
  their own affine"*.

**`openasl_square_crop` is not in the pose path as a crop at all.** It produces a 224 square only
under `--crop-style openasl`, which neither the dev nor the test set used. Where it *does* appear is
`task1_rtmpose/09_batch_clips.py:46` (`square_fn`), as a **coordinate** transform applied to keypoints
after extraction — the §2.5d square-norm that puts them in the authors' frame. Pixels and coordinates,
two different jobs, one helper.

**What the engine actually gets** (`task1_rtmpose/rtmpose_utils.py:76-91`): the native crop, with
`bbox = None` so the whole crop is the box, then `center/scale` from a **1.25 padding**, aspect-fixed
to the engine's input, then one `warpAffine` to **256 × 192** — `models/rtmw/preproc.json`, the RTMW
engine. Note that too: the deployed engine is **RTMW at 256 × 192**, not the RTMPose-x 288 × 384 in
`models/preproc.json`.

**So the instruction for the phone side is simpler than what you planned.** Send the video at native
resolution (cap the height at 720 to match the corpus) plus the person box, and let the board crop to
the box and hand the crop to the engine. **Do not resize to 224 and do not black-pad on the phone** —
that would resample twice (224, then the affine to 256 × 192) and feed the engine padding the affine
would happily include. You were already planning to send the box alongside the video, which is right;
only the 224 square is wrong.

**3.2 accepted.** 2–3 live phone clips through the board once you can send them, and I agree with your
reason for wanting it: live phone framing is a crop distribution nothing in `RESULTS.md` covers, and
the 1.25 padding plus the aspect fix behave differently on a box that is not a signer-centred OpenASL
box. **Queued behind the tunnel**, with the one-clip re-extraction from
`REPLY-NAN-SCORES-DIAGNOSED-2026-10-12.md:§5`.

## 4. `tegrastats` alongside the demo: yes, and it has been doing exactly that all along

**Already answered by the record rather than needing a confirmation run.** Every J-series power
figure was sampled concurrently with inference in the same process: `common/power_logger.py` starts
`tegrastats --interval <ms>` as a subprocess and reads it while the work runs.

- **Interval: 100 ms.** That is what every published run used — `lm_sweep_pruned_resident.json`
  records `power_interval_ms: 100` — so use it for the demo and the on-screen number is directly
  comparable to the frontier.
- **No sudo**, and `power_sudo: False` is recorded in the same artefact. `common/power_logger.py:25-27`
  notes some images need root for the INA3221 rails; ours does not, and there is a `_SysfsBackend`
  fallback reading `in*_input × curr*_input` from `/sys` if the binary is ever unavailable.
- **Does it perturb the measurement? Measurably no, and J10 is the evidence** — power sampling was
  active in both arms of that ±1.5 % agreement.

One caveat for the screen rather than the protocol: `tegrastats` reports `VDD_IN` as a running average
since start as well as an instantaneous value, and `parse_tegrastats_line` takes the instantaneous one
(`common/power_logger.py:19-23`). If you put a live number on screen, read it the same way or it will
lag what the audience is watching.

## 5. P12 dropped, and the filename collision fixed

**Accepted, and done in this commit.** `WORKSPLIT.md`'s P12 row is now marked dropped with your
reasoning recorded: hold-to-record gives an exact sentence boundary, so the heuristic has nothing left
to infer, and it removes a component that would have needed tuning and justifying on a crop
distribution we have never measured.

**You were right about the filename, and it was worse than cosmetic** — `jetson/p12_hybrid_hands.sh`
is the §2.5j hand-swap ablation and had nothing to do with P12. Renamed to
`jetson/hand_group_ablation.sh`, with the old name recorded at its `results/RESULTS.md` reference so
the artefact trail still resolves.

## 6. Bringing the board — **Tushar's call, not mine to make**

Physically it is a 15 W board with a barrel-jack supply, so there is no technical obstacle. But it is
**shared lab equipment under someone else's administration**, and both halves of your question —
whether it may leave the lab, and whether a day needs booking so other users are not cut off — are
permission questions about a shared resource, not facts I can look up. **Flagged to Tushar; answer to
follow from him rather than from me.**

What I will say is that the demo should not be *designed* around it. Your tunnel-based shape works
with the board in the lab, and that is the version that cannot be derailed by an answer arriving late.

---

## State of this ask

| item | status |
|---|---|
| 1 — resident footprint | **Answered from J10. Fits; 177 MB MemFree; 292 MB for the pose engine. Build on the PyTorch LM path, not TRT-LM** |
| 2 — inbound TCP from a peer | **Blocked on the lab-split tunnel.** Check the phone's `AllowedIPs` too |
| 3.1 — crop geometry | **Corrected: native bbox crop, not a 224 square.** Engine is RTMW 256 × 192 |
| 3.2 — 2–3 live clips | **Accepted, queued behind the tunnel** |
| 4 — `tegrastats` | **Answered from the record. 100 ms, no sudo, measured not to perturb** |
| 5 — drop P12 | **Done**, plus the filename collision fixed |
| 6 — travel | **Referred to Tushar** |

**Not claimed:** that TRT-LM plus the pose engine fits (untested, and §L8.2 is a reason for doubt);
that a listener on 8765 is reachable from the phone (unmeasured); that live phone framing produces
sentences rather than garbage (that is what 3.2 is for); that 177 MB of MemFree is comfortable — it is
not, and a demo server doing anything else in that process should re-check it.
