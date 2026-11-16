#!/usr/bin/env python3
"""Beam search over the ONNX translator graphs, written to transformers 4.57's _beam_search rules,
and checked token-for-token against PoseOnlyUniSign.translate(num_beams=4) on real clips.

    python3 task3_mt5_onnx/05_beam_onnx.py --pkls results/pkl_30clip_rtmw_fp32 --beams 4

This is the reference the phone's Swift decoder (ios/SLTEdge/Translator.swift) is ported from, so
every rule is spelled out with the transformers line it reproduces:

  * K = max(2, 1 + n_eos) * beams = 2 * beams candidates per step, top-K over beams x vocab of
    (running score + log_softmax)                                   _get_top_k_continuations
  * the next running beams are the best `beams` candidates that did NOT emit EOS (EOS ones get -1e9)
                                                                    _get_running_beams_for_next_iteration
  * an EOS candidate may enter the finished set only if its rank is < beams (top_num_beam_mask);
    its score is sum_logprob / generated_len ** length_penalty, generated_len counting the EOS
                                                                    _update_finished_beams
  * the heuristic stop: once every finished slot is filled and best_running / cur_generated_len <=
    worst finished, stop (early_stopping=False, length_penalty=1.0, the mT5 defaults)
                                                                    _check_early_stop_heuristic
  * max_length = max_new_tokens + 1 (the decoder start token); at that length every candidate
    hits the stopping criteria
"""
import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
NEG = -1.0e9


def log_softmax(x):
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


class OnnxTranslator:
    def __init__(self, d, threads=4):
        so = ort.SessionOptions(); so.intra_op_num_threads = threads
        mk = lambda f: ort.InferenceSession(os.path.join(d, f), so, providers=["CPUExecutionProvider"])
        self.front, self.enc = mk("front.onnx"), mk("encoder.onnx")
        self.init, self.step = mk("decoder_init.onnx"), mk("decoder_step.onnx")
        self.init_out = [o.name for o in self.init.get_outputs()]
        self.step_out = [o.name for o in self.step.get_outputs()]

    def encode(self, src):
        emb = self.front.run(None, {n: src[n].numpy().astype(np.float32) for n in ("body", "left", "right", "face_all")})[0]
        mask = np.ones((1, emb.shape[1]), np.int64)
        h = self.enc.run(None, {"inputs_embeds": emb, "attention_mask": mask})[0]
        return h, mask

    def beam(self, h, mask, beams=4, max_new_tokens=64, eos=1, start=0, length_penalty=1.0):
        max_length = max_new_tokens + 1
        K = max(2, 1 + 1) * beams
        o = dict(zip(self.init_out, self.init.run(None, {"input_ids": np.array([[start]], np.int64),
                                                         "encoder_hidden_states": h, "encoder_attention_mask": mask})))
        # Expand the batch-1 first step to `beams` identical rows; only row 0 is live (scores -1e9).
        past = {n.replace("present.", "past_key_values."): np.repeat(v, beams, axis=0)
                for n, v in o.items() if n.startswith("present.")}
        maskb = np.repeat(mask, beams, axis=0)
        logits = np.repeat(o["logits"][:, -1, :], beams, axis=0)              # [beams, V]
        V = logits.shape[-1]

        run_seq = [[] for _ in range(beams)]
        run_score = np.full(beams, NEG, np.float32); run_score[0] = 0.0
        fin_seq = [[] for _ in range(beams)]
        fin_score = np.full(beams, NEG, np.float32)
        fin_done = np.zeros(beams, bool)
        heur_unsat = True
        cur_len = 1                                                            # tokens so far incl. start
        while True:
            lp = log_softmax(logits.astype(np.float32)) + run_score[:, None]
            flat = lp.reshape(-1)
            top = np.argsort(-flat, kind="stable")[:K]
            top_lp = flat[top]; top_beam = top // V; top_tok = top % V
            top_seq = [run_seq[b] + [int(t)] for b, t in zip(top_beam, top_tok)]
            hits = (top_tok == eos) | (cur_len + 1 >= max_length)

            # e. next running beams: best `beams` non-finished candidates
            r_lp = top_lp + hits.astype(np.float32) * NEG
            nxt = np.argsort(-r_lp, kind="stable")[:beams]
            new_run_seq = [top_seq[i] for i in nxt]; new_run_score = r_lp[nxt]; parent = top_beam[nxt]

            # f. finished beams
            just = hits & (np.arange(K) < beams)
            cand = top_lp / ((cur_len + 1 - 1) ** length_penalty)
            if not heur_unsat:
                cand = cand + NEG
            cand = cand + (~just).astype(np.float32) * NEG
            m_seq = fin_seq + top_seq; m_score = np.concatenate([fin_score, cand]); m_done = np.concatenate([fin_done, just])
            keep = np.argsort(-m_score, kind="stable")[:beams]
            fin_seq = [m_seq[i] for i in keep]; fin_score = m_score[keep]; fin_done = m_done[keep]

            run_seq, run_score = new_run_seq, new_run_score.astype(np.float32)
            cur_len += 1
            worst = np.where(fin_done, fin_score.min(), NEG)
            best_possible = run_score[0] / ((cur_len - 1) ** length_penalty)
            heur_unsat = heur_unsat and bool(np.any(best_possible > worst))
            if not heur_unsat or bool(np.all(hits)):
                break

            # reorder the decoder self-attention cache to the surviving parents; encoder K/V are identical rows
            for n in list(past):
                if ".decoder." in n:
                    past[n] = past[n][parent]
            feed = {"input_ids": np.array([[s[-1]] for s in run_seq], np.int64), "encoder_attention_mask": maskb, **past}
            r = dict(zip(self.step_out, self.step.run(None, feed)))
            for n, v in r.items():
                if n.startswith("present."):
                    past[n.replace("present.", "past_key_values.")] = v
            logits = r["logits"][:, -1, :]
        best = fin_seq[0]
        return [t for t in best]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx-dir", default="models/mt5_pruned_onnx")
    ap.add_argument("--ckpt", default="weights/openasl_pose_only_slt_pruned.pth")
    ap.add_argument("--mt5", default="weights/mt5-base")
    ap.add_argument("--pkls", default="results/pkl_30clip_rtmw_fp32")
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import torch
    from common.pose_to_unisign import collate, load_pkl, to_model_inputs
    from unisign.model import load_model
    m = load_model(args.ckpt, args.mt5, device="cpu", dtype=torch.float32)
    tr = OnnxTranslator(args.onnx_dir)
    files = sorted(glob.glob(os.path.join(args.pkls, "*.pkl")))[:args.limit]
    rows, n_same = [], 0
    for f in files:
        kps, scs, _ = load_pkl(f)
        inputs, _ = to_model_inputs(kps, scs, 256)
        src = collate([inputs], ["x"])
        ref = m.translate(src, max_new_tokens=64, num_beams=args.beams)
        h, mask = tr.encode(src)
        got = tr.beam(h, mask, beams=args.beams)
        ref_t = [t for t in ref["tokens"] if t != 0]
        got_t = [t for t in got if t != 0]
        same = ref_t == got_t
        n_same += same
        rows.append({"clip": os.path.basename(f)[:-4], "same": same, "ref": ref["text"],
                     "ref_tokens": ref_t, "onnx_tokens": got_t})
        print(f"{'OK ' if same else 'DIFF'} {os.path.basename(f)[:40]:40s} {ref['text'][:70]}")
        if not same:
            print(f"      pytorch {ref_t}\n      onnx    {got_t}")
    print(f"token-identical: {n_same}/{len(files)} (beams={args.beams})")
    if args.out:
        json.dump({"beams": args.beams, "n": len(files), "token_identical": n_same, "rows": rows}, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
