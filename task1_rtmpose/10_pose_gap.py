"""Where do our keypoints differ from the authors', on the same 400 clips?

Both sets are in the OpenASL square-normalised frame (ours via common/renorm_to_openasl.py), so a
coordinate difference is directly interpretable. Distances are in NORMALISED units; multiply by the
square side to get pixels. Judged only where BOTH sources are confident (score >= 0.3), since a
disagreement about an unseen keypoint is not an extraction-quality difference.
"""
import pickle, sys, numpy as np
BODY=[0]+list(range(3,11)); LH=list(range(91,112)); RH=list(range(112,133))
FACE=list(range(23,40))[::2]+list(range(83,91))+[53]
USED=sorted(set(BODY+LH+RH+FACE))
G={"body":BODY,"left_hand":LH,"right_hand":RH,"face":FACE,"unisign_used":USED}
clips=[c for c in open(sys.argv[1]).read().split() if c]
acc={k:[] for k in G}; bias=[]; nclip=0
for c in clips:
    try:
        a=pickle.load(open(f'results/pkl_split_rtmw_fp16/{c}.pkl','rb'))
        b=pickle.load(open(f'data/openasl_pose/{c}.pkl','rb'))
    except Exception: continue
    ka,sa=np.asarray(a['keypoints'],float),np.asarray(a['scores'],float)
    kb,sb=np.asarray(b['keypoints'],float),np.asarray(b['scores'],float)
    if ka.shape!=kb.shape: continue
    ka,kb=ka[:,0],kb[:,0]; sa,sb=sa[:,0],sb[:,0]
    ok=(sa>=0.3)&(sb>=0.3)
    d=np.linalg.norm(ka-kb,axis=-1)
    for g,idx in G.items():
        m=ok[:,idx]
        if m.any(): acc[g].append(d[:,idx][m])
    mu=ok[:,USED]
    if mu.any(): bias.append((ka[:,USED]-kb[:,USED])[mu].mean(0))
    nclip+=1
print(f"{nclip} clips compared, normalised units (1.0 = the square crop's side)\n")
print(f"{'group':14s} {'n pts':>9s} {'mean':>8s} {'median':>8s} {'p95':>8s} {'<0.01':>7s} {'<0.02':>7s} {'<0.05':>7s}")
for g in G:
    v=np.concatenate(acc[g])
    print(f"{g:14s} {len(v):9,d} {v.mean():8.4f} {np.median(v):8.4f} {np.percentile(v,95):8.4f} "
          f"{100*(v<0.01).mean():6.1f}% {100*(v<0.02).mean():6.1f}% {100*(v<0.05).mean():6.1f}%")
b=np.mean(bias,axis=0)
print(f"\nmean signed offset over consumed keypoints: dx {b[0]:+.5f}  dy {b[1]:+.5f}  |offset| {np.hypot(*b):.5f}")
print("(a large shared offset would mean a systematic framing error; near zero means the")
print(" difference is per-keypoint scatter, not a misaligned coordinate system)")
