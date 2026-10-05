#!/usr/bin/env python3
"""Paired bootstrap CI on BLEU-4 / ROUGE-L between two eval JSONs. Interface L12.

    python -m unisign.bootstrap_ci results/eval_test_released_mac.json results/eval_test_pruned_mac.json -n 1000
    python -m unisign.bootstrap_ci results/eval_rtmw_fp32.json results/eval_authors_30clip.json -n 5000

Corpus BLEU is not a mean over sentences, so there is no per-sentence variance to t-test: resample
whole clips with replacement, recompute corpus BLEU on each resample for BOTH configs from the same
clip indices, and report the percentile interval of the paired delta. Pairing matters because the two
configs share the clips, so per-clip difficulty cancels.

Alignment: if both files carry "names" (eval_openasl.py writes them as of 2026-09-26) the comparison
is made on the intersection, sorted by name. Older files without "names" fall back to requiring
identical refs lists, which is positional and only safe when both evals scored exactly the same clips.

BLEU-4 here is sacrebleu 13a, which is bit-identical to what unisign/metrics.py reports, so the CI and
the point estimate in the eval JSON are the same metric.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from unisign.external_metrics import sacrebleu  # noqa: E402


def bleu4(refs, hyps):
    return sacrebleu.corpus_bleu(sys_stream=hyps, ref_streams=[refs], tokenize="13a").scores[3]


def rouge_l(refs, hyps):
    """Same call as unisign/metrics.py: Rouge().get_scores(hyp, ref, avg=True)['rouge-l']['f'] x 100."""
    from rouge import Rouge  # lazy: the `rouge` pip package, not needed for --no-rouge
    return Rouge().get_scores(hyps, refs, avg=True)["rouge-l"]["f"] * 100


def align(a_path, b_path):
    """-> (names_or_None, [(ref, pred_a, pred_b), ...])"""
    A, B = json.load(open(a_path)), json.load(open(b_path))
    if "names" in A and "names" in B:
        ma = {n: (r, p) for n, r, p in zip(A["names"], A["refs"], A["preds"])}
        mb = {n: (r, p) for n, r, p in zip(B["names"], B["refs"], B["preds"])}
        names = sorted(set(ma) & set(mb))
        if not names:
            raise SystemExit("no clips in common between the two evals")
        odd = set(ma) ^ set(mb)
        if odd:
            print(f"[ci] {len(odd)} clip(s) present in only one eval, dropped from the comparison")
        for n in names:
            if ma[n][0] != mb[n][0]:
                raise SystemExit(f"clip {n} has different reference text in the two files")
        return names, [(ma[n][0], ma[n][1], mb[n][1]) for n in names]
    # legacy path: no names recorded, so alignment is positional
    if A["refs"] != B["refs"]:
        raise SystemExit("neither file records 'names' and their refs lists differ, so the two evals "
                         "cannot be aligned. Re-run the evals so clip names are recorded.")
    print("[ci] no 'names' field: aligning positionally (pre-2026-09-26 eval output)")
    return None, list(zip(A["refs"], A["preds"], B["preds"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-rouge", action="store_true", help="BLEU-4 only (skips the rouge dependency)")
    ap.add_argument("--out", default=None, help="write the numbers as JSON too")
    ap.add_argument("--one-sided", action="store_true",
                    help="ALSO report the one-sided 95%% bound and margin from the SAME draws. "
                         "Reporting only; the two-sided interval stays the headline. A one-sided "
                         "bound is only legitimate if the direction was declared BEFORE the data "
                         "was seen -- see REPORT.md 7.1, which catalogues three instances of "
                         "reading a threshold after the fact.")
    args = ap.parse_args()

    names, rows = align(args.a, args.b)
    N = len(rows)
    print(f"[ci] {N} paired clips, {args.n} resamples")
    if N < 15:
        # C(2N-1, N) distinct multisets: only 126 at N=5. The interval cannot be finer than that
        # grid, so a narrow interval at this N is not evidence of precision.
        print(f"[ci] WARNING N={N} is small; the resample grid is coarse and a narrow interval here "
              f"is not evidence of precision")

    metrics = [("bleu4", bleu4)] + ([] if args.no_rouge else [("rouge_l", rouge_l)])
    rng = np.random.default_rng(args.seed)
    idxs = [rng.integers(0, N, N) for _ in range(args.n)]

    out = {"a": args.a, "b": args.b, "n_clips": N, "n_boot": args.n, "aligned_by": "names" if names else "position"}
    for tag, fn in metrics:
        refs = [r for r, _, _ in rows]
        pa = fn(refs, [x for _, x, _ in rows])
        pb = fn(refs, [x for _, _, x in rows])
        d = np.array([fn([rows[i][0] for i in ix], [rows[i][1] for i in ix]) -
                      fn([rows[i][0] for i in ix], [rows[i][2] for i in ix]) for ix in idxs])
        d = -d  # delta is b - a
        ci = (float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
        # One-sided margin, from the same draws. Reported as a MARGIN (distance from the point
        # estimate to the 5th percentile) rather than a bound, because that is the quantity a power
        # question compares an expected effect against. Do NOT compute this as 0.839 x half-width:
        # that normal approximation assumes symmetry and the resample distribution is not symmetric
        # here -- it understates the margin, and so overstates what going one-sided buys.
        one = None
        if args.one_sided:
            lo1 = float(np.percentile(d, 5.0))
            one = {"bound_5pct": lo1, "margin": abs((pb - pa) - lo1),
                   "two_sided_margin": abs((pb - pa) - ci[0])}
        # the degenerate case must not be labelled like a weak one: if every resample gives exactly
        # 0, the two configs emitted identical text on every clip, which is a strong statement
        identical = bool(pb == pa and np.all(d == 0))
        straddles = not (ci[0] > 0 or ci[1] < 0)
        verdict = ("identical text on all clips -- no difference to bound" if identical else
                   "NOT ESTABLISHED (CI includes 0)" if straddles else "sign established")
        out[tag] = {"a": pa, "b": pb, "delta": pb - pa, "ci": ci,
                    "identical_on_these_clips": identical,
                    "straddles_zero": straddles and not identical,
                    "p_wrong_sign": float(np.mean(np.sign(d) != np.sign(pb - pa))) if pb != pa else 0.0}
        if one:
            out[tag]["one_sided"] = one
        print(f"  {tag:8s} {pa:6.2f} -> {pb:6.2f}   delta {pb - pa:+6.2f}  "
              f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  {verdict}")
        if one:
            print(f"  {'':8s} one-sided 95% bound {one['bound_5pct']:+.4f}  margin "
                  f"{one['margin']:.4f} against two-sided {one['two_sided_margin']:.4f}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print("[ci] wrote", args.out)


if __name__ == "__main__":
    main()
