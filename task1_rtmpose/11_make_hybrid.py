#!/usr/bin/env python3
"""Build hand-swapped hybrid pose sets, to test whether the pose gap (RESULTS.md 2.5h) is in the hands.

Writes four directories of pkls over the clips present in BOTH sources with identical frame counts:
    ours_m            our keypoints                  -- baseline
    theirs_m          the authors' keypoints         -- ceiling
    theirs_ourhands   their body+face, OUR hands
    ours_theirhands   our body+face, THEIR hands
Only COCO-WholeBody indices 91..132 (both hands) are exchanged, keypoints and scores together.

A file-based script rather than a heredoc on purpose: jetson/run.sh exec-batch is `docker exec`
WITHOUT -i, so stdin never reaches the container and `python3 -` silently reads an empty program and
exits 0. That cost one run.
"""
import argparse, os, pickle, sys
import numpy as np

HANDS = list(range(91, 133))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", required=True)
    ap.add_argument("--theirs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dirs = {k: os.path.join(a.out, k)
            for k in ("ours_m", "theirs_m", "theirs_ourhands", "ours_theirhands")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    def save(d, name, k, s):
        t = os.path.join(d, name + ".part")
        with open(t, "wb") as fh:
            pickle.dump({"keypoints": k, "scores": s}, fh)
        os.replace(t, os.path.join(d, name))   # never a half-written pkl under the final name

    n = miss = shape = 0
    for f in sorted(os.listdir(a.ours)):
        if not f.endswith(".pkl"):
            continue
        pb = os.path.join(a.theirs, f)
        if not os.path.exists(pb):
            miss += 1
            continue
        ka_d = pickle.load(open(os.path.join(a.ours, f), "rb"))
        kb_d = pickle.load(open(pb, "rb"))
        ka = np.asarray(ka_d["keypoints"], np.float32); sa = np.asarray(ka_d["scores"], np.float32)
        kb = np.asarray(kb_d["keypoints"], np.float32); sb = np.asarray(kb_d["scores"], np.float32)
        if ka.shape != kb.shape:
            shape += 1
            continue
        save(dirs["ours_m"], f, ka, sa)
        save(dirs["theirs_m"], f, kb, sb)
        k1, s1 = kb.copy(), sb.copy()
        k1[..., HANDS, :] = ka[..., HANDS, :]; s1[..., HANDS] = sa[..., HANDS]
        save(dirs["theirs_ourhands"], f, k1, s1)
        k2, s2 = ka.copy(), sa.copy()
        k2[..., HANDS, :] = kb[..., HANDS, :]; s2[..., HANDS] = sb[..., HANDS]
        save(dirs["ours_theirhands"], f, k2, s2)
        n += 1
    print(f"[hybrid] {n} matched clips written to {a.out}/*; "
          f"{miss} missing from --theirs, {shape} shape mismatches", flush=True)
    if n == 0:
        print("[hybrid] ERROR: nothing written", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
