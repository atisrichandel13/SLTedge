#!/usr/bin/env python3
"""Keypoint agreement between two pose engines, per body-part group. Task P4 (guide 2.4).

    python task1_rtmpose/07_kpt_agreement.py --ref results/rtmw_trt_fp32.json \
        --test results/rtmw_trt_fp16.json --out results/kpt_agreement_rtmw_fp16.json

Both inputs are keypoint dumps from task1_rtmpose/03_infer_frames.py (same frames, same
preproc, different engine). The FP32 engine is the reference; "agreement" is the fraction
of keypoints the quantised engine places within 1 / 2 / 5 px of it.

Why this exists. 06_compare_trt.py gates on the *worst* pixel error over 20 frames, and a
simcc head decodes by argmax: a flat heatmap on an out-of-frame hand can flip a whole bin
and blow the gate while every joint that matters is exact. Both FP16 engines failed that
gate; RTMW's failure was one bin of quantisation, RTMPose-x's was a real overflow. A
distribution over thousands of keypoints separates those two cases, and per-group numbers
say whether the disagreement lands where the downstream model looks.

Distance is Euclidean (PCK convention), unlike 06_compare_trt.py's per-axis max, so numbers
here are <= sqrt(2) x that script's for the same keypoint.

Three things are reported per group:
  * within_px       - the agreement metric proper, over keypoints the reference is confident in
  * px_percentiles  - where the tail actually sits (p99.9 and max are the argmax flips)
  * conf_flip       - keypoints that cross the score threshold in one engine but not the other.
                      Uni-Sign zeroes any joint scoring under 0.3 (common/pose_to_unisign.py
                      THR), so a flip changes the tensor the LM sees even at 0 px of motion.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Uni-Sign's pose-only path keeps 69 of the 133 keypoints; the rest never reach the ST-GCN.
# These four slices are common/pose_to_unisign.py's BODY/LHAND/RHAND/FACE_IDX, mirrored because
# that module imports torch and this one is pure numpy (it runs wherever the dumps land). When
# torch *is* installed the real lists are imported and checked against these, so they cannot drift.
BODY_IDX = [0] + list(range(3, 11))
LHAND_IDX = list(range(91, 112))
RHAND_IDX = list(range(112, 133))
FACE_IDX = list(range(23, 23 + 17))[::2] + list(range(83, 83 + 8)) + [53]
try:
    from common import pose_to_unisign as _p2u  # noqa: E402
    assert (_p2u.BODY_IDX, _p2u.LHAND_IDX, _p2u.RHAND_IDX, _p2u.FACE_IDX) == \
        (BODY_IDX, LHAND_IDX, RHAND_IDX, FACE_IDX), \
        "keypoint groups here have drifted from common/pose_to_unisign.py"
except ImportError:  # no torch in this interpreter
    pass

# COCO-WholeBody 133-keypoint layout.
GROUPS = {
    "body": list(range(0, 17)),
    "feet": list(range(17, 23)),
    "face": list(range(23, 91)),
    "left_hand": list(range(91, 112)),
    "right_hand": list(range(112, 133)),
    "hands": list(range(91, 133)),
    # what Uni-Sign's pose-only path actually consumes (69 of 133); the rest is discarded
    # before the ST-GCN, so disagreement there cannot reach the translation.
    "unisign_used": sorted(set(BODY_IDX + LHAND_IDX + RHAND_IDX + FACE_IDX)),
    "all": list(range(0, 133)),
}


def load_one(path):
    """-> frame names, keypoints (N,K,2), scores (N,K), summary."""
    d = json.load(open(path))
    res = d["results"]
    names = [r["frame"] for r in res]
    kpts = np.asarray([r["keypoints"] for r in res], dtype=np.float64)
    scores = np.asarray([r["scores"] for r in res], dtype=np.float64)
    if kpts.ndim == 4:  # (N,1,K,2) if a single-instance axis survived
        kpts, scores = kpts[:, 0], scores[:, 0]
    return names, kpts, scores, d.get("summary", {})


def clip_key(path):
    """Clip id from a dump named <config>__<vid>.json, else the whole stem.

    jetson/p4_clips.sh writes one dump per (engine, clip), so the reference and the test set are
    paired on the clip rather than on argument order - a missing clip on one side is then an error
    instead of a silent off-by-one that would compare two different signers."""
    stem = os.path.basename(path)[:-len(".json")] if path.endswith(".json") else os.path.basename(path)
    return stem.split("__", 1)[1] if "__" in stem else stem


def load(items):
    """Concatenate several per-clip dumps, given (clip id, path) pairs.

    Frame names are qualified by the clip id, so the same f_0001.jpg in five clips stays five
    distinct keypoints when the dumps are concatenated."""
    names, kpts, scores, per_clip, summary = [], [], [], [], {}
    for key, path in items:
        n, k, s, summ = load_one(path)
        names += [f"{key}/{x}" if key else x for x in n]
        kpts.append(k); scores.append(s)
        per_clip.append({"clip": key, "source": path, "n_frames": len(n)})
        summary = summ  # only used for the latency line; last one wins when several are given
    return names, np.concatenate(kpts), np.concatenate(scores), summary, per_clip


def stats(dist, conf, thresholds):
    """dist (N,K) px, conf (N,K) bool mask of reference-confident keypoints."""
    d = dist[conf]
    out = {"n_keypoints": int(dist.size), "n_confident": int(conf.sum()),
           "frac_confident": float(conf.mean())}
    if d.size == 0:  # a group can be entirely unconfident (feet, when the signer is framed chest-up)
        out["within_px"] = None
        out["px_percentiles"] = None
        return out
    out["within_px"] = {f"{t:g}px": float((d <= t).mean()) for t in thresholds}
    out["px_percentiles"] = {f"p{q}": float(np.percentile(d, q)) for q in (50, 90, 99, 99.9)}
    out["px_percentiles"]["max"] = float(d.max())
    out["mean_px"] = float(d.mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, nargs="+",
                    help="FP32 keypoint dump(s) from 03_infer_frames.py; several are concatenated")
    ap.add_argument("--test", required=True, nargs="+",
                    help="the quantised engine's dump(s) over the same frames")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-score", type=float, default=0.3,
                    help="judge px error only where the reference scores at least this; matches "
                         "common/pose_to_unisign.py THR, the threshold Uni-Sign gates joints on")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 2.0, 5.0])
    ap.add_argument("--label", default=None, help="config name for the report (default: --test stem)")
    args = ap.parse_args()

    if len(args.ref) == 1 and len(args.test) == 1:
        # one dump each: the two files are the pair, whatever they are called
        ref_by_clip = {"": args.ref[0]}; test_by_clip = {"": args.test[0]}
    else:
        ref_by_clip = {clip_key(p): p for p in args.ref}
        test_by_clip = {clip_key(p): p for p in args.test}
    if len(ref_by_clip) != len(args.ref) or len(test_by_clip) != len(args.test):
        raise SystemExit("two dumps on the same side map to the same clip id")
    if set(ref_by_clip) != set(test_by_clip):
        raise SystemExit("--ref and --test cover different clips: "
                         f"ref-only {sorted(set(ref_by_clip) - set(test_by_clip))}, "
                         f"test-only {sorted(set(test_by_clip) - set(ref_by_clip))}")
    clips = sorted(ref_by_clip)
    rn, rk, rs, rsum, ref_clips = load([(c, ref_by_clip[c]) for c in clips])
    tn, tk, ts, tsum, test_clips = load([(c, test_by_clip[c]) for c in clips])
    if rn != tn:
        common = sorted(set(rn) & set(tn))
        if not common:
            raise SystemExit("--ref and --test share no frame names")
        print(f"[warn] frame lists differ; comparing the {len(common)} shared frames "
              f"(ref {len(rn)}, test {len(tn)})")
        ri = [rn.index(f) for f in common]; ti = [tn.index(f) for f in common]
        rk, rs, tk, ts, rn = rk[ri], rs[ri], tk[ti], ts[ti], common
    if rk.shape != tk.shape:
        raise SystemExit(f"shape mismatch: ref {rk.shape} vs test {tk.shape}")

    dist = np.linalg.norm(tk - rk, axis=-1)          # (N,K) px in the source frame
    conf = rs >= args.min_score
    rep = {
        "label": args.label or clip_key(args.test[0]),
        "ref": args.ref, "test": args.test, "n_clips": len(clips),
        "n_frames": int(rk.shape[0]), "n_keypoints_per_frame": int(rk.shape[1]),
        "clips": ref_clips,
        "min_score": args.min_score, "distance": "euclidean_px_source_frame",
        "thresholds_px": args.thresholds,
        "groups": {g: stats(dist[:, idx], conf[:, idx], args.thresholds)
                   for g, idx in GROUPS.items()},
    }

    # Score-threshold crossings: same joint, one engine keeps it, the other drops it.
    flip = conf != (ts >= args.min_score)
    rep["conf_flip"] = {
        "n": int(flip.sum()), "frac": float(flip.mean()),
        "ref_conf_frac": float(conf.mean()), "test_conf_frac": float((ts >= args.min_score).mean()),
        "per_group_frac": {g: float(flip[:, idx].mean()) for g, idx in GROUPS.items()},
        "score_mean_abs_diff": float(np.abs(ts - rs).mean()),
        "score_max_abs_diff": float(np.abs(ts - rs).max()),
    }

    # Worst confident keypoint, and which group owns it.
    masked = np.where(conf, dist, -1.0)
    fi, ki = np.unravel_index(int(np.argmax(masked)), masked.shape)
    rep["worst_confident"] = {
        "px": float(dist[fi, ki]), "frame": rn[fi], "keypoint": int(ki),
        "group": next(g for g in ("body", "feet", "face", "left_hand", "right_hand")
                      if ki in GROUPS[g]),
        "ref_score": float(rs[fi, ki]), "test_score": float(ts[fi, ki]),
        "ref_xy": rk[fi, ki].round(2).tolist(), "test_xy": tk[fi, ki].round(2).tolist(),
    }
    # Per-frame worst error, so a single bad frame is visible rather than averaged away.
    per_frame = np.where(conf.any(axis=1), masked.max(axis=1), np.nan)
    rep["per_frame_worst_px"] = {
        "mean": float(np.nanmean(per_frame)), "max": float(np.nanmax(per_frame)),
        "frac_frames_all_within_5px": float(np.nanmean(per_frame <= 5.0)),
    }
    if rsum.get("trt_ms") and tsum.get("trt_ms"):
        rep["trt_ms_mean"] = {"ref": rsum["trt_ms"]["mean"], "test": tsum["trt_ms"]["mean"]}

    order = ["body", "hands", "left_hand", "right_hand", "face", "feet", "unisign_used", "all"]
    w = max(len(g) for g in order)
    head = "  ".join(f"<={t:g}px" for t in args.thresholds)
    refname = os.path.basename(args.ref[0]) + (f" +{len(args.ref) - 1} more" if len(args.ref) > 1 else "")
    print(f"\n{rep['label']}   {rep['n_frames']} frames over {len(clips)} clip(s), reference {refname}")
    print(f"{'group':<{w}}  {'n conf':>8}  {head}  {'p50':>6} {'p90':>6} {'p99.9':>7} {'max':>7}")
    for g in order:
        s = rep["groups"][g]
        if s["within_px"] is None:
            print(f"{g:<{w}}  {s['n_confident']:>8}  (no confident keypoints)")
            continue
        fr = "  ".join(f"{s['within_px'][f'{t:g}px'] * 100:6.2f}%" for t in args.thresholds)
        p = s["px_percentiles"]
        print(f"{g:<{w}}  {s['n_confident']:>8}  {fr}  {p['p50']:6.2f} {p['p90']:6.2f} "
              f"{p['p99.9']:7.2f} {p['max']:7.2f}")
    cf = rep["conf_flip"]
    print(f"\nscore-threshold flips at {args.min_score}: {cf['n']} ({cf['frac'] * 100:.3f}% of all "
          f"keypoints); confident fraction {cf['ref_conf_frac'] * 100:.1f}% -> "
          f"{cf['test_conf_frac'] * 100:.1f}%")
    wc = rep["worst_confident"]
    print(f"worst confident keypoint: {wc['px']:.2f} px, kpt {wc['keypoint']} ({wc['group']}) "
          f"on {wc['frame']}, ref score {wc['ref_score']:.3f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        json.dump(rep, open(args.out, "w"), indent=1)
        print("[agree] wrote", args.out)


if __name__ == "__main__":
    main()
