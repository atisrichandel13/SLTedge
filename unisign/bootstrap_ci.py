#!/usr/bin/env python3
"""Paired bootstrap CI on BLEU-4 / ROUGE-L between two eval JSONs (same refs). Interface L12.
    python -m unisign.bootstrap_ci results/eval_test_released_mac.json results/eval_test_pruned_mac.json -n 1000
"""
import argparse, json, sys, os
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from unisign.external_metrics import sacrebleu

def bleu4(refs, hyps):
    return sacrebleu.corpus_bleu(sys_stream=hyps, ref_streams=[refs], tokenize="13a").scores[3]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
    A, B = json.load(open(args.a)), json.load(open(args.b))
    assert A["refs"] == B["refs"]; refs = A["refs"]; N = len(refs)
    rng = np.random.default_rng(args.seed); da, db, dd = [], [], []
    for _ in range(args.n):
        idx = rng.integers(0, N, N)
        r = [refs[i] for i in idx]; ba = bleu4(r, [A["preds"][i] for i in idx]); bb = bleu4(r, [B["preds"][i] for i in idx])
        da.append(ba); db.append(bb); dd.append(bb - ba)
    q = lambda x: np.percentile(x, [2.5, 50, 97.5])
    full_a, full_b = bleu4(refs, A["preds"]), bleu4(refs, B["preds"])
    print(f"A {os.path.basename(args.a)}: BLEU-4 {full_a:.2f}  95% CI [{q(da)[0]:.2f}, {q(da)[2]:.2f}]")
    print(f"B {os.path.basename(args.b)}: BLEU-4 {full_b:.2f}  95% CI [{q(db)[0]:.2f}, {q(db)[2]:.2f}]")
    lo, med, hi = q(dd)
    print(f"paired delta B-A: {full_b-full_a:+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  P(B<A) = {np.mean(np.array(dd)<0):.3f}  (n={args.n} resamples, N={N})")

if __name__ == "__main__":
    main()
