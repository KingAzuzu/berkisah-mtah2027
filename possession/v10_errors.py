"""Error breakdown of v8 (LOMO preds, v6 decoder) vs merged truth (K 2), IDSSE TUNE. Diagnostic only.
Spells = maximal alive H/A runs. Greedy one-to-one match by IoU. python -m possession.v10_errors"""
import numpy as np, joblib
from . import v7_decoder as E, v4_lib as V
from .v8_convention import merge
from .data import TUNE
from evaluation import possession_metrics as pm

c = joblib.load(E.CACHE)


def spells(t):
    ch = np.flatnonzero(np.diff(t)) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(t)]
    return [(s, e, t[s]) for s, e in zip(st, en) if t[s] != pm.D]


def iou(a, b):
    i = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    return i / (max(a[1], b[1]) - min(a[0], b[0]))


bins = [(0, 2), (2, 5), (5, 15), (15, 1e9)]
L = {b: [0, 0, 0] for b in bins}          # n, matched@0.3, matched@0.7
cause = {"wrong team": 0, "pred D": 0, "same team, IoU<0.3 (boundary/merge)": 0}
off = {"start": [], "end": []}
npred = ntrue = 0
for m in TUNE:
    fs = [np.load(E.LOG / "v8" / f"m2_s{s}_{m}.npz") for s in range(20, 25)]
    for i in range(2):
        h = c[m][i]; n = len(h["truth"])
        lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
        b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
        p = V.viterbi_scaled(0.5 * lp + 0.5 * h["eR"], 20.0, 5.0, 1 - 0.6 * b)
        t = merge(np.asarray(h["truth"]), 2)
        T, P = spells(t), spells(p)
        npred += len(P); ntrue += len(T)
        for s in T:
            d = (s[1] - s[0]) / 25
            bb = next(x for x in bins if x[0] <= d < x[1])
            best, bi = 0, None
            for q in P:
                if q[2] == s[2] and q[1] > s[0] and q[0] < s[1]:
                    v = iou(s, q)
                    if v > best: best, bi = v, q
            L[bb][0] += 1; L[bb][1] += best >= 0.3; L[bb][2] += best >= 0.7
            if best >= 0.3:
                off["start"].append((bi[0] - s[0]) / 25); off["end"].append((bi[1] - s[1]) / 25)
            else:
                seg = p[s[0]:s[1]]
                fr_d = np.mean(seg == pm.D); fr_same = np.mean(seg == s[2])
                cause["pred D" if fr_d > 0.5 else ("same team, IoU<0.3 (boundary/merge)" if fr_same >= 0.5 else "wrong team")] += 1
print(f"true spells {ntrue}, pred spells {npred}")
print("length | n | matched@0.3 | matched@0.7")
for bb, (n, a, z) in L.items():
    print(f"{bb[0]}-{bb[1] if bb[1] < 1e9 else 'inf'}s | {n} | {a/n:.2f} | {z/n:.2f}")
miss = sum(cause.values())
print("missed causes:", {k: f"{v} ({v/miss:.2f})" for k, v in cause.items()})
for k, v in off.items():
    v = np.array(v)
    print(f"{k} offset s (pred-true): p10/p50/p90 {np.percentile(v,10):.2f}/{np.median(v):.2f}/{np.percentile(v,90):.2f}  |off|<=0.5s {np.mean(abs(v)<=0.5):.2f}  <=1s {np.mean(abs(v)<=1):.2f}")
