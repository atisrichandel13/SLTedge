#!/usr/bin/env python3
"""Which pose extractor produced Uni-Sign's released OpenASL poses?

Compares, for one clip, the authors' released pkl against candidate extractors run on the same
frames. All sources are brought to Uni-Sign's pkl convention: xy normalised by frame [W, H],
133 COCO-WholeBody keypoints, one person per frame.

    python task1_rtmpose/08_compare_extractors.py \
        --ref data/openasl_ref_pose/Ads-4j06eJY-00:07:37.233-00:07:47.200.pkl \
        --jetson results/rtmpose_trt_fp32.json --crop-wh 666 720 \
        --pkl rtmlib_lightweight=results/rtmlib_lightweight_Ads-4j06eJY.pkl \
        --pkl rtmlib_performance=results/rtmlib_performance_Ads-4j06eJY.pkl

Reports per keypoint group the mean normalised distance (x frame width in px for scale) between
each candidate and the reference, over frames/keypoints where both have conf >= --thr, plus a
score-distribution comparison. The closest candidate is the extractor the checkpoint saw in training.
"""
import argparse
import json
import pickle

import numpy as np

GROUPS = {"body9": [0] + list(range(3, 11)), "lhand": list(range(91, 112)),
          "rhand": list(range(112, 133)), "face18": list(range(23, 40, 2)) + list(range(83, 91)) + [53],
          "all133": list(range(133))}


def load_pkl(path):
    d = pickle.load(open(path, "rb"))
    k = np.stack([np.asarray(x)[0] for x in d["keypoints"]]).astype(np.float32)  # T,133,2
    s = np.stack([np.asarray(x)[0] for x in d["scores"]]).astype(np.float32)     # T,133
    return k, s


def load_jetson(path, wh):
    d = json.load(open(path))
    rows = sorted(d["results"], key=lambda r: r["frame"])
    k = np.asarray([r["keypoints"] for r in rows], dtype=np.float32).reshape(len(rows), 133, 2)
    s = np.asarray([r["scores"] for r in rows], dtype=np.float32).reshape(len(rows), 133)
    return k / np.asarray(wh, dtype=np.float32)[None, None], s


def align(a, b):
    """Resample the longer sequence to the shorter by nearest index (handles fps mismatch)."""
    ta, tb = len(a[0]), len(b[0])
    if ta == tb:
        return a, b
    if ta > tb:
        idx = np.round(np.linspace(0, ta - 1, tb)).astype(int)
        return (a[0][idx], a[1][idx]), b
    idx = np.round(np.linspace(0, tb - 1, ta)).astype(int)
    return a, (b[0][idx], b[1][idx])


def compare(ref, cand, thr, width_px):
    (rk, rs), (ck, cs) = align(ref, cand)
    out = {}
    for g, idx in GROUPS.items():
        m = (rs[:, idx] >= thr) & (cs[:, idx] >= thr)
        if m.sum() == 0:
            out[g] = None
            continue
        d = np.linalg.norm(rk[:, idx] - ck[:, idx], axis=-1)[m]
        out[g] = {"mean_px": float(d.mean() * width_px), "median_px": float(np.median(d) * width_px),
                  "p95_px": float(np.percentile(d, 95) * width_px), "n": int(m.sum()),
                  "frac_both_conf": float(m.mean())}
    out["score_mean_ref_vs_cand"] = [float(rs.mean()), float(cs.mean())]
    out["score_corr"] = float(np.corrcoef(rs.ravel(), cs.ravel())[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="authors' released pkl for the clip")
    ap.add_argument("--jetson", default=None, help="03_infer_frames.py JSON (pixel coords)")
    ap.add_argument("--crop-wh", type=int, nargs=2, default=[666, 720])
    ap.add_argument("--pkl", action="append", default=[], help="name=path of a candidate pkl")
    ap.add_argument("--thr", type=float, default=0.3)
    args = ap.parse_args()

    ref = load_pkl(args.ref)
    print(f"[ref] frames={len(ref[0])} score_mean={ref[1].mean():.3f} "
          f"hands={ref[1][:, 91:133].mean():.3f} face={ref[1][:, 23:91].mean():.3f}")
    cands = {}
    if args.jetson:
        cands["jetson_rtmpose-x_384x288"] = load_jetson(args.jetson, args.crop_wh)
    for spec in args.pkl:
        name, path = spec.split("=", 1)
        cands[name] = load_pkl(path)
    report = {}
    for name, c in cands.items():
        r = compare(ref, c, args.thr, args.crop_wh[0])
        report[name] = r
        print(f"\n== {name}: frames={len(c[0])} score_mean={c[1].mean():.3f}")
        for g in GROUPS:
            v = r[g]
            if v:
                print(f"   {g:8s} mean {v['mean_px']:6.2f} px  median {v['median_px']:6.2f}  "
                      f"p95 {v['p95_px']:6.2f}  (n={v['n']}, both-conf {v['frac_both_conf']:.2f})")
        print(f"   score corr {r['score_corr']:.3f}  mean ref/cand {r['score_mean_ref_vs_cand']}")
    ranked = sorted(((r["all133"]["mean_px"], n) for n, r in report.items() if r["all133"]), key=lambda x: x[0])
    print("\nclosest to the released poses:", ranked[0][1] if ranked else "n/a")
    json.dump(report, open("results/extractor_comparison.json", "w"), indent=2)


if __name__ == "__main__":
    main()
