#!/usr/bin/env python3
"""Paired bootstrap CIs for the adaptation 2x2 (guide row 4.4), BLEU-4 and ROUGE-L.

    python -m unisign.adapt_ci --evals results/block4 -n 1000

Why this exists rather than reusing bootstrap_ci.py: row 4.4 needs more than pairwise deltas. The
question "does adaptation help MORE under frame-rate shift than without it" is a difference of two
deltas, and subtracting two independently-bootstrapped CIs is both wrong and less sensitive than
resampling the difference directly. Every comparison here is scored on ONE set of resample draws, so
the deltas and the difference-in-differences are mutually comparable (same approach as
unisign/grid_table.py and unisign/did_ci.py).

ROUGE-L is reported alongside BLEU-4 and is the metric to lead with for effects below ~1 point. At
n<=1000, BLEU-4 is 4-gram precision and cannot resolve them: it disagreed between the test and dev
splits on the 16 fps cost (-1.33 vs -0.34) while ROUGE-L replicated (-1.63 vs -1.30), and at n=300 it
flipped the sign of that effect outright. See results/RESULTS.md L15 and the L13 section.

Alignment is positional. These eval JSONs carry no "names" field, so the script asserts that every
file's `refs` list is identical before comparing -- a config missing one clip would otherwise shift
every later sentence against its partner with no error raised.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from unisign.external_metrics import sacrebleu  # noqa: E402


def bleu4(refs, hyps):
    return sacrebleu.corpus_bleu(sys_stream=hyps, ref_streams=[refs], tokenize="13a").scores[3]


def rougeL(refs, hyps):
    from rouge import Rouge
    # Same call as unisign.metrics.translation_performance, so numbers match the eval JSONs.
    return Rouge().get_scores(hyps, refs, avg=True)["rouge-l"]["f"] * 100


def q(x):
    return np.percentile(x, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", default="results/block4", help="dir of eval_dev_*.json")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rouge", action="store_true", help="BLEU-4 only (much faster)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    runs = {}
    for f in sorted(glob.glob(os.path.join(a.evals, "eval_dev_*.json"))):
        tag = os.path.basename(f)[len("eval_dev_"):-len(".json")]
        runs[tag] = json.load(open(f))
    if not runs:
        sys.exit(f"no eval_dev_*.json in {a.evals}")

    refs = None
    for tag, d in runs.items():
        if refs is None:
            refs = d["refs"]
        elif d["refs"] != refs:
            sys.exit(f"{tag}: refs differ from the others -- positional alignment is unsafe")
    N = len(refs)
    print(f"{len(runs)} evals, N={N} clips, {a.n} resamples\n")

    metrics = [("BLEU-4", bleu4)] + ([] if a.no_rouge else [("ROUGE-L", rougeL)])

    # Point estimates first: cheap, and they are what goes in the tables.
    print(f"{'run':28s} {'BLEU-4':>8s} {'ROUGE-L':>8s}")
    for tag, d in runs.items():
        print(f"{tag:28s} {d['bleu']['bleu4']:8.2f} {d['rouge_l']:8.2f}")

    # One draw scores every run, so all deltas below are mutually comparable.
    rng = np.random.default_rng(a.seed)
    draws = {name: {tag: [] for tag in runs} for name, _ in metrics}
    for i in range(a.n):
        idx = rng.integers(0, N, N)
        r = [refs[j] for j in idx]
        for name, fn in metrics:
            for tag, d in runs.items():
                draws[name][tag].append(fn(r, [d["preds"][j] for j in idx]))
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{a.n}", flush=True)

    out = {"n_clips": N, "n_resamples": a.n, "point": {}, "deltas": {}, "did": {}}
    for tag, d in runs.items():
        out["point"][tag] = {"bleu4": d["bleu"]["bleu4"], "rouge_l": d["rouge_l"]}

    def delta(name, b, aa):
        """Paired delta b - a on the shared draws."""
        v = np.array(draws[name][b]) - np.array(draws[name][aa])
        full = runs[b]["bleu"]["bleu4"] - runs[aa]["bleu"]["bleu4"] if name == "BLEU-4" \
            else runs[b]["rouge_l"] - runs[aa]["rouge_l"]
        lo, hi = q(v)
        return full, lo, hi, float(np.mean(v < 0)), v

    pairs = [("unadapt_src_cap100", "unadapt_fps16_cap100", "un-adapted: source -> 16 fps"),
             ("unadapt_src_cap100", "adapt_src_cap100", "source rate: un-adapted -> adapted"),
             ("unadapt_fps16_cap100", "adapt_fps16_s42_cap100", "16 fps: un-adapted -> adapted"),
             ("adapt_fps16_s42_cap64", "adapt_fps16_s42_cap100", "adapted 16 fps: cap 64 -> cap 100")]

    print("\nPaired deltas (95% CI, 'P(<0)' = fraction of resamples where the delta is negative)")
    for aa, b, label in pairs:
        if aa not in runs or b not in runs:
            print(f"  {label:42s} -- waiting on {', '.join(t for t in (aa, b) if t not in runs)}")
            continue
        print(f"  {label}")
        for name, _ in metrics:
            full, lo, hi, p, _ = delta(name, b, aa)
            print(f"      {name:8s} {full:+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]  P(<0)={p:.3f}")
            out["deltas"].setdefault(label, {})[name] = {"delta": full, "lo": lo, "hi": hi, "p_lt0": p}

    # Difference-in-differences: does adaptation help MORE at 16 fps than at source rate?
    need = ["unadapt_src_cap100", "adapt_src_cap100", "unadapt_fps16_cap100", "adapt_fps16_s42_cap100"]
    if all(t in runs for t in need):
        print("\nDifference-in-differences: (adapted-unadapted @16fps) - (adapted-unadapted @source)")
        print("  positive = adaptation buys more under frame-rate shift than it does in general")
        for name, _ in metrics:
            g_src = np.array(draws[name]["adapt_src_cap100"]) - np.array(draws[name]["unadapt_src_cap100"])
            g_16 = np.array(draws[name]["adapt_fps16_s42_cap100"]) - np.array(draws[name]["unadapt_fps16_cap100"])
            d = g_16 - g_src
            lo, hi = q(d)
            print(f"      {name:8s} {np.mean(d):+6.2f}  [{lo:+6.2f}, {hi:+6.2f}]  P(>0)={np.mean(d>0):.3f}")
            out["did"][name] = {"mean": float(np.mean(d)), "lo": lo, "hi": hi,
                                "p_gt0": float(np.mean(d > 0))}
    else:
        print("\nDiD: waiting on " + ", ".join(t for t in need if t not in runs))

    # Seed spread. NOT a CI: it bundles seed effects with GPU non-determinism, which is non-zero
    # here -- three runs at the identical seed 43 gave final losses 0.3765 / 0.3742 / 0.3788.
    seeds = {t: runs[t] for t in runs if t.startswith("adapt_fps16_s") and t.endswith("cap100")}
    if len(seeds) > 1:
        print(f"\nSeed spread across {len(seeds)} runs at 16 fps (range, not a CI):")
        for name, key in (("BLEU-4", lambda d: d["bleu"]["bleu4"]), ("ROUGE-L", lambda d: d["rouge_l"])):
            v = [key(d) for d in seeds.values()]
            print(f"      {name:8s} {min(v):.2f} .. {max(v):.2f}   spread {max(v)-min(v):.2f}   "
                  f"mean {np.mean(v):.2f}")
            out.setdefault("seed_spread", {})[name] = {"min": min(v), "max": max(v),
                                                       "mean": float(np.mean(v)), "runs": list(seeds)}

    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
