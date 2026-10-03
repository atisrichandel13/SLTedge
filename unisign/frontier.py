"""
Accuracy-energy frontier for the deployed pipeline.

IMPORTANT - what kind of number each column is:

  * ACCURACY  : measured by us, n=976 OpenASL test clips, pruned FP32 checkpoint,
                max_new_tokens 64, batch 8. See results/RESULTS.md L13.
  * POSE ENERGY: measured by the pose track on the board (15 W, mode 0, INA3221 VDD_IN,
                RTMW-l-m FP16, 3 repeats). HANDOFF-LM-TRACK-2026-09-26 s0d.
  * LM ENERGY : measured by the pose track on the board, same conditions, as a function of
                beam width and encoder length T. HANDOFF s0b.
  * SYSTEM J  : *** COMPOSED ***, not measured end-to-end. pose_J_per_s * clip_seconds + LM_J.
                Validated against the single measured end-to-end run (M1): composition
                predicts 47.2 J where 50.37 J was measured, i.e. the composition runs ~6% LOW.
                Every composed cell inherits that. Do not quote a composed cell as a measured one.

TWO BIASES IN THE ABSOLUTE COLUMN, IN OPPOSITE DIRECTIONS. Neither affects the RELATIVE
comparisons across cells -- every cell shares the same clip and the same rates -- and the
relative ordering is the deliverable. But the absolute joules are not a typical clip's.

  1. ~6% LOW (optimistic). The composition misses the 3.16 J between 47.2 predicted and 50.37
     measured. Two explanations have been offered here and BOTH are now refuted by measurement:

       * CPU/GPU contention (the pose track's s3). Their M1 run measured NO contention
         (25.10 ms/frame end-to-end vs 25.03 standalone) because the implementation is sequential
         -- all frames, then the LM -- so the stages never overlap.
       * Unmodelled within-run stages: the convert step, the idle floor during the 64 s model
         load, or the warm-up sentence. Refuted by the 2x2 (RESULTS.md 5.3, 2026-10-02), which
         found the stages EXACTLY ADDITIVE: pose + convert + LM = total to within 0.04 ms in every
         cell. Priced at M1's own 5.56 W average, the 25.7 ms convert step is 0.143 J, i.e. 4.5%
         of the 3.16 J residual. Load and warm-up sit outside the power window by construction
         (RESULTS.md 5.1).

     So the residual is NOT a within-run accounting gap. It comes from the comparison ACROSS runs:
     POSE_J_PER_S and LM_J were measured in separate runs, on a different clip from M1's, each
     with its own baseline draw and thermal state. Chasing it further needs per-stage ENERGY from
     one end-to-end run -- the 2x2 reports per-stage LATENCY but only a per-sentence total, so it
     cannot split the residual between the pose and LM terms. That is an open ask to the board.

  2. ~7% HIGH (pessimistic) for a median clip. POSE_J_PER_S was measured on one clip whose crop
     is at the 83rd percentile of crop area across the 931-clip split (9k to 770k px, median
     395k). Per-frame pose cost is ~13.97 ms fixed (TRT runs at a fixed 256x192 input) plus
     ~11.13 ms that scales with crop area, so a median clip is ~7% cheaper per frame and a p10
     clip ~27% cheaper. Meanwhile CLIP_S below is the MEAN TEST CLIP duration, not that clip's --
     so the absolute column already mixes a duration from our split with an energy rate from
     their clip. Fixing it needs pose energy over clips spanning the crop range (their C9 work).

Two interpolations are flagged in the output:
  * beam 2 was not measured at T=137 or T=68, so it is placed at the same fraction of the
    beam1->beam4 interval that it occupies at T=215 (0.639).
  * LM energy at the exact frame counts our accuracy cells use (221/183/122) is linearly
    interpolated within the measured T grid (215/137/68).
"""

# --- measured: pose stage, J per second of video (HANDOFF s0d) ---
POSE_J_PER_S = {"source": 4.25, 24: 3.36, 16: 2.27}

# --- measured: LM stage, J per sentence, by (beams, frames used) (HANDOFF s0b) ---
LM_J = {(1, 215): 8.88, (2, 215): 10.26, (4, 215): 11.04,
        (1, 137): 8.55,                  (4, 137): 9.89,
        (1, 68):  7.79,                  (4, 68):  8.94}

# --- measured by us: accuracy, n=976 test (RESULTS.md L13) ---
ACC = {(4, "source"): (22.87, 42.98), (4, 24): (22.80, 43.13), (4, 16): (21.54, 41.35),
       (2, "source"): (22.06, 42.22), (2, 24): (21.93, 42.29), (2, 16): (21.19, 40.89),
       (1, "source"): (20.88, 41.56), (1, 24): (20.66, 41.18), (1, 16): (19.83, 39.53)}

CLIP_S = 7.61        # mean test clip duration, measured over the 976 clips
SRC_FPS = 29.07      # mean per-clip source rate, measured over the 976 clips

# beam 2 sits this far along the beam1->beam4 energy interval at the one T where all three
# were measured; assumed constant across T.
B2_FRAC = (LM_J[(2, 215)] - LM_J[(1, 215)]) / (LM_J[(4, 215)] - LM_J[(1, 215)])


def frames_for(fps):
    return round(CLIP_S * (SRC_FPS if fps == "source" else fps))


def lm_energy(beams, frames):
    """Linear interpolation in T within the measured grid. Returns (joules, interpolated?)."""
    grid = sorted({t for (_, t) in LM_J})           # 68, 137, 215
    interp = False

    def at(t):
        nonlocal interp
        if (beams, t) in LM_J:
            return LM_J[(beams, t)]
        interp = True                                # beam 2 off-grid in T
        lo, hi = LM_J[(1, t)], LM_J[(4, t)]
        return lo + B2_FRAC * (hi - lo)

    if frames <= grid[0]:
        return at(grid[0]), interp
    if frames >= grid[-1]:
        return at(grid[-1]), interp
    for a, b in zip(grid, grid[1:]):
        if a <= frames <= b:
            interp = True
            f = (frames - a) / (b - a)
            return at(a) + f * (at(b) - at(a)), interp
    raise AssertionError


def build_rows(acc=None):
    """The nine cells with composed system energy. Pure; no printing.

    `acc` maps (beams, fps) -> (bleu4, rouge_l); it defaults to this module's transcribed ACC,
    but plot_frontier.py passes values read straight out of the eval JSONs instead.
    """
    rows = []
    for (beams, fps), (bleu, rouge) in (acc or ACC).items():
        fr = frames_for(fps)
        lm, itp = lm_energy(beams, fr)
        pose = POSE_J_PER_S[fps] * CLIP_S
        rows.append({"beams": beams, "fps": fps, "frames": fr, "bleu": bleu, "rouge": rouge,
                     "pose_J": pose, "lm_J": lm, "sys_J": pose + lm, "interp": itp})
    mark_pareto(rows)
    return rows


def mark_pareto(rows):
    """Pareto: maximise BLEU-4, minimise system joules. Mutates rows in place."""
    for r in rows:
        r["pareto"] = not any(o["bleu"] >= r["bleu"] and o["sys_J"] <= r["sys_J"]
                              and (o["bleu"] > r["bleu"] or o["sys_J"] < r["sys_J"]) for o in rows)
    return rows


NAME = {4: "beam 4", 2: "beam 2", 1: "greedy"}


def main():
    rows = build_rows()
    name = NAME
    print(f"\nComposed accuracy-energy frontier, mean test clip ({CLIP_S} s, {SRC_FPS} fps source)")
    print("System J is COMPOSED from two separately measured stages, not measured end-to-end.")
    print("Absolute J carries two opposing biases (see docstring): ~6% low from the unmodelled")
    print("convert/load/warm-up, ~7% high because the pose energy rate came from an 83rd-percentile")
    print("crop. RELATIVE comparisons across cells are unaffected. * = interpolated LM cell.\n")
    print(f"{'decoder':<9}{'fps':>7}{'frames':>8}{'BLEU-4':>8}{'ROUGE-L':>9}"
          f"{'pose J':>8}{'LM J':>7}{'sys J':>8}  {'vs best':>8}  frontier")
    best = max(r["sys_J"] for r in rows)
    for r in sorted(rows, key=lambda x: -x["sys_J"]):
        print(f"{name[r['beams']]:<9}{str(r['fps']):>7}{r['frames']:>8}{r['bleu']:>8.2f}{r['rouge']:>9.2f}"
              f"{r['pose_J']:>8.1f}{r['lm_J']:>6.1f}{'*' if r['interp'] else ' '}{r['sys_J']:>8.1f}"
              f"{100*(r['sys_J']/best-1):>+8.0f}%  {'PARETO' if r['pareto'] else '-- dominated'}")

    print("\nFrontier, cheapest first:")
    for r in sorted([x for x in rows if x["pareto"]], key=lambda x: x["sys_J"]):
        print(f"  {r['sys_J']:6.1f} J   BLEU-4 {r['bleu']:5.2f}   ROUGE-L {r['rouge']:5.2f}   "
              f"{name[r['beams']]} @ {r['fps']} fps")

    ref = next(r for r in rows if r["beams"] == 4 and r["fps"] == "source")
    print("\nThe two levers, priced from the reference cell (beam 4 @ source):")
    for beams, fps, label in [(4, 24, "frame rate 30 -> 24"), (4, 16, "frame rate 30 -> 16"),
                              (2, "source", "beam 4 -> beam 2"), (1, "source", "beam 4 -> greedy")]:
        r = next(x for x in rows if x["beams"] == beams and x["fps"] == fps)
        dj, db, dr = r["sys_J"] - ref["sys_J"], r["bleu"] - ref["bleu"], r["rouge"] - ref["rouge"]
        print(f"  {label:<22} {dj:+6.1f} J ({100*dj/ref['sys_J']:+5.1f}%)   "
              f"BLEU-4 {db:+5.2f}   ROUGE-L {dr:+5.2f}   -> {abs(dj)/max(abs(db),1e-9):5.1f} J per BLEU-4 point")


if __name__ == "__main__":
    main()
