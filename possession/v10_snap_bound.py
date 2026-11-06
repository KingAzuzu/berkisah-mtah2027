"""Upper bound (oracle, uses truth): move each predicted label-change frame of v8 to the nearest true change frame within W s,
if one exists; score vs merged truth (K 2). Measures what perfect boundary refinement could give. IDSSE TUNE. Diagnostic only."""
import numpy as np, joblib
from . import v7_decoder as E, v4_lib as V
from .v8_convention import merge
from .data import TUNE
from .score import score

c = joblib.load(E.CACHE)


def snap(p, t, w):
    pc = np.flatnonzero(np.diff(p)) + 1; tc = np.flatnonzero(np.diff(t)) + 1
    if len(tc) == 0: return p
    new = []
    for x in pc:
        j = tc[np.argmin(abs(tc - x))]
        new.append(j if abs(j - x) <= w * 25 else x)
    q = np.empty_like(p); bounds = np.r_[0, new, len(p)]; vals = p[np.r_[0, pc]]
    order = np.argsort(bounds[1:-1], kind="stable"); bounds = np.r_[0, np.array(new)[order], len(p)]; vals = np.r_[vals[0], vals[1:][order]]
    for k in range(len(vals)): q[bounds[k]:bounds[k + 1]] = vals[k]
    return q


preds, tru = {}, {}
for m in TUNE:
    fs = [np.load(E.LOG / "v8" / f"m2_s{s}_{m}.npz") for s in range(20, 25)]
    preds[m], tru[m] = [], []
    for i in range(2):
        h = c[m][i]; n = len(h["truth"])
        lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
        b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
        preds[m].append(V.viterbi_scaled(0.5 * lp + 0.5 * h["eR"], 20.0, 5.0, 1 - 0.6 * b))
        tru[m].append(dict(h, truth=merge(np.asarray(h["truth"]), 2)))
for w in (0, 0.5, 1, 2):
    s = [score(tru[m], [snap(p, h["truth"], w) for p, h in zip(preds[m], tru[m])])["spell_f1_mean"] for m in TUNE]
    print(f"oracle snap W {w} s: {np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s))
