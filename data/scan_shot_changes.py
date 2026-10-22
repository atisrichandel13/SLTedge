import pickle, glob, os, json, sys
import numpy as np

LOW, HIGH = 0.50, 0.70      # absolute mean-confidence bands, from the L28 verified case (0.246 vs 0.821)
MIN_RUN   = 8               # >= ~0.33 s at 24 fps; shorter runs are blinks/occlusions, not shot changes
MIN_SIGNER= 0.25            # >= 25% of the clip must be clearly signer, or it is a bad clip not a CHANGE

def runs_below(mask):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None: start = i
        elif not v and start is not None:
            out.append((start, i-1)); start = None
    if start is not None: out.append((start, len(mask)-1))
    return out

def scan(d):
    res = []
    for f in sorted(glob.glob(os.path.join(d, '*.pkl'))):
        o = pickle.load(open(f, 'rb'))
        s = np.asarray(o['scores'], np.float64).reshape(len(o['scores']), -1)
        m = np.nanmean(np.where(np.isfinite(s), s, np.nan), axis=1)
        m = np.nan_to_num(m, nan=0.0)
        n = len(m)
        lo = m < LOW
        signer_frac = float((m > HIGH).mean())
        rr = [(a, b) for a, b in runs_below(lo) if b - a + 1 >= MIN_RUN]
        if rr and signer_frac >= MIN_SIGNER:
            tot = sum(b - a + 1 for a, b in rr)
            res.append({
                'clip': os.path.basename(f)[:-4], 'n_frames': n,
                'runs': rr, 'low_frames': tot, 'low_frac': tot / n,
                'signer_frac': signer_frac,
                'low_mean': float(np.mean([m[a:b+1].mean() for a, b in rr])),
                'high_mean': float(m[m > HIGH].mean()),
                'contrast': float(m[m > HIGH].mean() - np.mean([m[a:b+1].mean() for a, b in rr])),
            })
    return res

for d in sys.argv[1:]:
    r = scan(d)
    n_tot = len(glob.glob(os.path.join(d, '*.pkl')))
    print(json.dumps({'dir': d, 'n_clips': n_tot, 'flagged': r}))
