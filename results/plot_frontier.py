#!/usr/bin/env python3
"""Milestone M4 / C10: the accuracy-energy frontier plot and its CSV.

    python results/plot_frontier.py

Writes results/frontier.png and results/frontier.csv.

PROVENANCE OF EVERY NUMBER ON THE PLOT -- nothing here is transcribed by hand:

  * ACCURACY (y axis, and the error bars) comes from results/frontier_accuracy.json, which
    unisign/grid_table.py writes by scoring the nine eval JSONs over all 976 OpenASL test clips.
    Regenerate with:  python -m unisign.grid_table -n 1000 --out results/frontier_accuracy.json
    The error bars are a PAIRED bootstrap against the beam-4/source reference cell under ONE set
    of resample draws shared by every cell, so the bars are comparable with each other and not
    only with the reference. They are intervals on the DELTA from the reference, re-centred on
    each cell's own BLEU-4; the reference cell therefore has no bar by construction.

  * ENERGY (x axis) is imported from unisign.frontier, which composes two stages the pose track
    measured on the board at 15 W (mode 0, INA3221 VDD_IN): pose J/s x clip seconds + LM J.

THREE THINGS THIS PLOT IS NOT, ALL ANNOTATED ON THE FIGURE ITSELF:

  1. System joules are COMPOSED, not measured end-to-end. Against the one measured end-to-end run
     the composition predicts 47.2 J where 50.37 J was measured (~6% low), and the pose rate came
     from a clip at the 83rd percentile of crop area (~7% high for a median clip). The two biases
     oppose. RELATIVE ordering across cells -- which is the deliverable -- is unaffected, because
     every cell shares the same clip and the same rates.
  2. Reduced frame rates are EMULATED by thinning already-extracted 30 fps keypoints. Exact for
     the pose model (extraction is per-frame independent), but a real 16 fps camera would differ
     in exposure and motion blur.
  3. Accuracy was scored with the keep set built over train+dev+test. The test-only leak is worth
     -0.03 BLEU-4 [-0.11, +0.01] and applies to all nine cells alike, so it cannot reorder them.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unisign.frontier import CLIP_S, NAME, SRC_FPS, build_rows  # noqa: E402

R = os.path.join(ROOT, "results")
ACC_JSON = os.path.join(R, "frontier_accuracy.json")
PNG = os.path.join(R, "frontier.png")
CSV = os.path.join(R, "frontier.csv")

if not os.path.exists(ACC_JSON):
    sys.exit(f"missing {ACC_JSON}\n"
             f"run: python -m unisign.grid_table -n 1000 --out {ACC_JSON}")

surf = json.load(open(ACC_JSON))
cells = surf["cells"]

# (beams, fps) -> (bleu, rouge), keyed the way unisign.frontier wants it. fps stays the string
# "source" for the unthinned rate and becomes an int otherwise.
acc, ci = {}, {}
for v in cells.values():
    fps = v["fps"] if v["fps"] == "source" else int(v["fps"])
    key = (v["beams"], fps)
    acc[key] = (v["bleu4"], v["rouge_l"])
    ci[key] = v["ci"]

rows = build_rows(acc)
missing = [k for k in acc if k not in {(r["beams"], r["fps"]) for r in rows}]
if missing:
    sys.exit(f"energy model has no rate for {missing}")

# ---------------------------------------------------------------------------- CSV
with open(CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["decoder", "beams", "fps", "frames_used", "bleu4", "rouge_l",
                "delta_bleu4_vs_ref", "ci_lo", "ci_hi",
                "pose_J", "lm_J", "system_J", "lm_J_interpolated", "on_pareto_frontier"])
    for r in sorted(rows, key=lambda x: x["sys_J"]):
        k = (r["beams"], r["fps"])
        c = ci.get(k)
        d = cells[f"{r['beams']},{r['fps']}"]["delta_bleu4"]
        w.writerow([NAME[r["beams"]], r["beams"], r["fps"], r["frames"],
                    f"{r['bleu']:.2f}", f"{r['rouge']:.2f}",
                    "reference" if d is None else f"{d:+.2f}",
                    "" if not c else f"{c[0]:+.2f}", "" if not c else f"{c[1]:+.2f}",
                    f"{r['pose_J']:.1f}", f"{r['lm_J']:.1f}", f"{r['sys_J']:.1f}",
                    int(r["interp"]), int(r["pareto"])])
print(f"wrote {CSV}")

# ---------------------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(9.2, 7.1))

# one colour per frame rate, one marker per decoder: the two levers read independently
COL = {"source": "#1f4e79", 24: "#2e8b57", 16: "#c0504d"}
MRK = {4: "o", 2: "s", 1: "^"}
FPS_LABEL = {"source": f"source (~{SRC_FPS:.0f} fps)", 24: "24 fps (emulated)",
             16: "16 fps (emulated)"}

# Pareto staircase first, so it sits behind the points.
par = sorted([r for r in rows if r["pareto"]], key=lambda x: x["sys_J"])
ax.step([r["sys_J"] for r in par], [r["bleu"] for r in par], where="post",
        color="0.45", lw=1.4, ls="--", zorder=1)

for r in rows:
    k = (r["beams"], r["fps"])
    c = ci.get(k)
    # The CI is on the DELTA from the reference, and this cell's score already equals
    # ref + delta. So the half-widths around this cell's point are (d - lo) and (hi - d),
    # which places the bar at [ref + lo, ref + hi] -- the interval the bootstrap actually
    # estimated. Re-centring the raw [lo, hi] on the cell's score instead would put the whole
    # bar below the point, which is what the first version of this did.
    d = cells[f"{r['beams']},{r['fps']}"]["delta_bleu4"]
    err = None if not c else [[d - c[0]], [c[1] - d]]
    ax.errorbar(r["sys_J"], r["bleu"], yerr=err, fmt=MRK[r["beams"]], ms=9.5,
                color=COL[r["fps"]], mec="white", mew=1.1, ecolor=COL[r["fps"]],
                elinewidth=1.2, capsize=3.5, alpha=0.95, zorder=3)
    ax.annotate(f"{NAME[r['beams']]}\n{r['sys_J']:.1f} J  {r['bleu']:.2f}",
                (r["sys_J"], r["bleu"]), textcoords="offset points",
                xytext=(9, -14 if r["beams"] == 1 else 5), fontsize=7.3, color="0.25")

ax.set_xlabel("System energy per sentence (J)  —  composed: pose J/s × clip s + LM J, "
              "measured at 15 W", fontsize=9.5)
ax.set_ylabel("BLEU-4, 976 OpenASL test clips", fontsize=9.5)
ax.set_title("Accuracy–energy frontier: frame rate vs decoder search on Jetson Orin Nano @ 15 W",
             fontsize=11.5, pad=12)
ax.grid(alpha=0.25, lw=0.6)
ax.set_axisbelow(True)

# Give the rightmost annotation room so it does not run off the axes.
x0, x1 = ax.get_xlim()
ax.set_xlim(x0 - 0.02 * (x1 - x0), x1 + 0.13 * (x1 - x0))

# Two legend columns that each mean one thing: colour = frame rate, marker = decoder.
# matplotlib fills ncol=2 column-major, so pad the shorter column to keep the split clean.
blank = Line2D([], [], ls="", label="")
handles = [Line2D([], [], marker="o", ls="", color=COL[f], mec="white", ms=9, label=FPS_LABEL[f])
           for f in ("source", 24, 16)]
handles += [Line2D([], [], ls="--", color="0.45", lw=1.4, label="Pareto frontier")]
handles += [Line2D([], [], marker=MRK[b], ls="", color="0.35", mec="white", ms=9, label=NAME[b])
            for b in (4, 2, 1)]
handles += [blank]
ax.legend(handles=handles, fontsize=8.2, loc="lower right", framealpha=0.95, ncol=2,
          handletextpad=0.6, columnspacing=1.6)

paras = [
    f"Accuracy: n={surf['n_clips']} OpenASL test clips, pruned FP32, max_new_tokens {surf['cap']}. "
    f"Mean test clip {CLIP_S:.2f} s at {SRC_FPS:.2f} fps source. Error bars: paired bootstrap, "
    f"{surf['n_boot']} resamples with draws shared by every cell, on the BLEU-4 delta vs beam 4 @ "
    f"source; the reference cell has no bar by construction.",
    "System J is COMPOSED from two separately measured board stages (pose J/s x clip s + LM J), "
    "NOT measured end-to-end: ~6% low against the one end-to-end run, ~7% high for a median-crop "
    "clip. The two biases oppose, and neither reorders the cells.",
    "Reduced frame rates are EMULATED by thinning 30 fps keypoints -- exact for the pose model, "
    "but not equivalent to a real low-rate camera. All cells carry a -0.03 BLEU-4 test-set "
    "vocabulary leak, uniform across cells and so unable to reorder them.",
    "Accuracy is scored on the AUTHORS' released keypoints. The deployed system uses our own RTMW "
    "FP16 extractor; substituting it is worth -0.35 BLEU-4 [-4.23, +3.17] at n=30, so the deployed "
    "system's ABSOLUTE accuracy carries roughly +-4 BLEU-4. Every cell shares the same poses, so "
    "the relative ordering -- the deliverable -- is unaffected.",
]
# Wrap to the figure width in characters, or the caveats get clipped off the right edge.
import textwrap  # noqa: E402
width = int(fig.get_figwidth() * 100 / 6.0)
foot = "\n".join(textwrap.fill(p, width) for p in paras)
fig.text(0.010, 0.010, foot, fontsize=6.3, color="0.35", va="bottom", linespacing=1.4)
# four caveat paragraphs need real room, or they ride up over the x-axis label
fig.tight_layout(rect=(0, 0.165, 1, 1))
fig.savefig(PNG, dpi=200)
print(f"wrote {PNG}")

# ---------------------------------------------------------------------------- what the plot says
ref = next(r for r in rows if r["beams"] == 4 and r["fps"] == "source")
cheap = min(rows, key=lambda r: r["sys_J"])
knee = next(r for r in par if r["fps"] == 24 and r["beams"] == 4)
print(f"\nfrontier: {len(par)} of {len(rows)} cells are Pareto-optimal")
print(f"  reference   beam 4 @ source : {ref['sys_J']:5.1f} J  BLEU-4 {ref['bleu']:.2f}")
print(f"  knee        beam 4 @ 24 fps : {knee['sys_J']:5.1f} J  BLEU-4 {knee['bleu']:.2f}  "
      f"({100*(knee['sys_J']/ref['sys_J']-1):+.0f}% J for {knee['bleu']-ref['bleu']:+.2f} BLEU-4)")
print(f"  cheapest    {NAME[cheap['beams']]:6s} @ {str(cheap['fps']):6s}: {cheap['sys_J']:5.1f} J  "
      f"BLEU-4 {cheap['bleu']:.2f}  ({100*(cheap['sys_J']/ref['sys_J']-1):+.0f}% J for "
      f"{cheap['bleu']-ref['bleu']:+.2f} BLEU-4)")
