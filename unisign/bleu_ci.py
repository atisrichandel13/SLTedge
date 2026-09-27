#!/usr/bin/env python3
"""Paired bootstrap CI for the BLEU/ROUGE gap between two pose front-ends.

    python unisign/bleu_ci.py --a results/eval_rtmw_fp32.json --b results/eval_rtmw_fp16.json
    python unisign/bleu_ci.py --a results/eval_authors_5clip.json --b results/eval_rtmw_fp32.json --n-boot 10000

Why this exists: at n=5 clips a BLEU-4 difference of several points is a coin flip, and we published
one (the retracted 5/5 FP16 claim, RESULTS.md 2.5b). Corpus BLEU is not a mean over sentences, so a
per-sentence t-test does not apply -- resample whole clips with replacement, recompute corpus BLEU on
each resample for BOTH configs from the same clip indices, and report the percentile interval of the
paired delta. Pairing matters: the configs share the clips, so the shared per-clip difficulty cancels.

Both eval JSONs must carry "names" (added 2026-09-26); the comparison is made on the intersection, in
a fixed order, so a config missing a clip cannot silently shift the alignment.
"""
import argparse
import contextlib
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def load(path):
    d = json.load(open(path))
    if "names" not in d:
        raise SystemExit(f"{path} has no 'names' field -- it predates the paired-analysis change; "
                         f"re-run the eval so clips can be aligned by name rather than by position")
    return {n: (r, p) for n, r, p in zip(d["names"], d["refs"], d["preds"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline eval JSON")
    ap.add_argument("--b", required=True, help="the config being compared against it")
    ap.add_argument("--n-boot", type=int, default=5000)
    ap.add_argument("--alpha", type=float, default=0.05, help="0.05 -> 95 % interval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from unisign.metrics import translation_performance

    A, B = load(args.a), load(args.b)
    names = sorted(set(A) & set(B))
    if not names:
        raise SystemExit("no clips in common")
    only = (set(A) ^ set(B))
    if only:
        print(f"[ci] {len(only)} clip(s) in only one of the two evals, dropped: "
              f"{sorted(only)[:3]}{' ...' if len(only) > 3 else ''}")
    print(f"[ci] {len(names)} paired clips, {args.n_boot} resamples")
    if len(names) < 15:
        # C(2n-1, n) distinct multisets: 126 at n=5. The interval cannot be finer than that grid,
        # so a "significant" result at this n is still resting on a handful of sentences.
        print(f"[ci] WARNING n={len(names)} is small; the resample grid is coarse and a narrow "
              f"interval here is not evidence of precision")

    def score(idx, D):
        refs = [D[names[i]][0] for i in idx]
        preds = [D[names[i]][1] for i in idx]
        # translation_performance prints its own dict; thousands of resamples would bury the result
        with contextlib.redirect_stdout(io.StringIO()):
            bleu, rouge = translation_performance(refs, preds)
        return bleu["bleu4"], rouge

    full = np.arange(len(names))
    a_b4, a_rl = score(full, A)
    b_b4, b_rl = score(full, B)

    rng = np.random.default_rng(args.seed)
    d_b4, d_rl = [], []
    for _ in range(args.n_boot):
        idx = rng.integers(0, len(names), len(names))
        x4, xl = score(idx, A)
        y4, yl = score(idx, B)
        d_b4.append(y4 - x4); d_rl.append(yl - xl)
    d_b4, d_rl = np.array(d_b4), np.array(d_rl)

    lo, hi = 100 * args.alpha / 2, 100 * (1 - args.alpha / 2)
    out = {"a": args.a, "b": args.b, "n_clips": len(names), "n_boot": args.n_boot}
    for tag, pa, pb, d in (("bleu4", a_b4, b_b4, d_b4), ("rouge_l", a_rl, b_rl, d_rl)):
        ci = (float(np.percentile(d, lo)), float(np.percentile(d, hi)))
        # a paired bootstrap CI that straddles 0 means the sign of this delta is not established.
        # The degenerate case is different and must not be labelled the same way: if every resample
        # gives exactly 0, the two configs produced identical text on every clip, which is a strong
        # statement about these clips rather than a weak one.
        identical = bool(pb == pa and np.all(d == 0))
        straddles = not (ci[0] > 0 or ci[1] < 0)
        out[tag] = {"a": pa, "b": pb, "delta": pb - pa, "ci": ci, "identical_on_these_clips": identical,
                    "straddles_zero": straddles and not identical,
                    "p_wrong_sign": float(np.mean(np.sign(d) != np.sign(pb - pa)))}
        verdict = ("identical text on all clips -- no difference to bound" if identical else
                   "NOT ESTABLISHED (CI includes 0)" if straddles else "sign established")
        print(f"  {tag:8s} {pa:6.2f} -> {pb:6.2f}   delta {pb - pa:+6.2f}  "
              f"95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]  {verdict}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print("[ci] wrote", args.out)


if __name__ == "__main__":
    main()
