#!/usr/bin/env python3
"""Difference-in-differences bootstrap: does the frame-rate cost depend on INT8?

DiD = (fps16 - fps30 | INT8) - (fps16 - fps30 | no INT8), on the same resampled clips each draw.
A CI straddling 0 means the two knobs do not interact -> their deltas may be quoted independently.
"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from unisign.external_metrics import sacrebleu

def bleu4(refs, hyps):
    return sacrebleu.corpus_bleu(sys_stream=hyps, ref_streams=[refs], tokenize="13a").scores[3]

R = os.path.join(ROOT, "results") + "/"
F = {"fp30": "eval_test_pruned_mac.json", "fp16": "eval_test_pruned_truefps16.json",
     "q30":  "eval_test_pruned_w8_mac.json", "q16": "eval_test_w8_truefps16.json"}
D = {k: json.load(open(R + v)) for k, v in F.items()}
refs = D["fp30"]["refs"]
for k, d in D.items():
    assert d["refs"] == refs, k
N = len(refs)
rng = np.random.default_rng(0)
did, d_fp, d_q = [], [], []
for _ in range(1000):
    idx = rng.integers(0, N, N)
    r = [refs[i] for i in idx]
    b = {k: bleu4(r, [D[k]["preds"][i] for i in idx]) for k in F}
    a = b["fp16"] - b["fp30"]   # fps cost without INT8
    c = b["q16"] - b["q30"]     # fps cost with INT8
    d_fp.append(a); d_q.append(c); did.append(c - a)
q = lambda x: np.percentile(x, [2.5, 97.5])
full = {k: bleu4(refs, D[k]["preds"]) for k in F}
a0 = full["fp16"] - full["fp30"]; c0 = full["q16"] - full["q30"]
print(f"16 fps cost, no INT8 : {a0:+.2f}  95% CI [{q(d_fp)[0]:+.2f}, {q(d_fp)[1]:+.2f}]")
print(f"16 fps cost, with INT8: {c0:+.2f}  95% CI [{q(d_q)[0]:+.2f}, {q(d_q)[1]:+.2f}]")
print(f"interaction (DiD)     : {c0-a0:+.2f}  95% CI [{q(did)[0]:+.2f}, {q(did)[1]:+.2f}]  "
      f"P(DiD<0) = {np.mean(np.array(did) < 0):.3f}  (n=1000, N={N})")
