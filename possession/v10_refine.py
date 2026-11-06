"""v10 boundary refiner (diagnostic, not wired). Moves each v8 predicted label-change frame to the best of candidates within +-2 s,
scored by a HistGradientBoosting P(true change of same type at f). Strict LOMO on IDSSE TUNE: refiner for match m trained on the other
4 matches' v8 fold predictions, targets from merged truth (K 2). Design margin 0.05; variant margin 0.15 (stated before running).
python -m possession.v10_refine"""
import json
import numpy as np, joblib
from sklearn.ensemble import HistGradientBoostingClassifier as GB
from . import v7_decoder as E, v4_lib as V, m2_lib as M, tcn_lib as T
from .v8_convention import merge
from .data import TUNE
from .score import score

W, STEP = 50, 5
PROG = E.LOG / "v10" / "progress_A.txt"
def prog(s): PROG.write_text(s + "\n")


def feats(h, lp, b, eR, p):
    n = len(p)
    F = T.frame_features(h)
    F = np.repeat(F, 5, 0); F = np.concatenate([F, np.repeat(F[-1:], n - len(F), 0)])[:n]
    mu, sd = np.nanmean(F, 0), np.nanstd(F, 0) + 1e-6
    F = ((F - mu) / sd).astype(np.float32)
    ch = np.flatnonzero(np.diff(p)) + 1
    rows, meta = [], []
    cl = lambda i: np.clip(i, 0, n - 1)
    for k, x in enumerate(ch):
        a, bb = p[x - 1], p[x]
        for f in range(max(1, x - W), min(n - 1, x + W) + 1, STEP):
            r = [F[f]]
            for o in (12, 25, 50):
                r.append(F[cl(f + o)] - F[cl(f - o)])
            r.append([b[f], b[cl(f - 5):cl(f + 6)].mean(), b[cl(f - 25):cl(f + 26)].max(), (f - x) / 25, abs(f - x) / 25, a, bb,
                      b[x]])
            r.append(lp[cl(f - 12)] - lp[cl(f + 12)]); r.append(eR[cl(f - 12)] - eR[cl(f + 12)]); r.append(lp[f]); r.append(eR[f])
            rows.append(np.concatenate([np.ravel(z) for z in r])); meta.append((k, f, x))
    return np.array(rows, np.float32), np.array(meta), ch


def targets(meta, p, t):
    tc = np.flatnonzero(np.diff(t)) + 1
    out = np.zeros(len(meta), int)
    for j, (k, f, x) in enumerate(meta):
        a, bb = p[x - 1], p[x]
        c = tc[(t[tc - 1] == a) & (t[tc] == bb)]
        if len(c):
            i = np.argmin(abs(c - x))
            if abs(c[i] - x) <= W: out[j] = abs(f - c[i]) <= STEP
    return out


def refine(p, meta, ch, prob, margin):
    q = p.copy(); new = ch.copy(); prev = 0
    for k, x in enumerate(ch):
        sel = meta[:, 0] == k
        fs, ps = meta[sel, 1], prob[sel]
        nxt = ch[k + 1] - 1 if k + 1 < len(ch) else len(p) - 1
        ok = (fs > prev) & (fs <= nxt)
        px = ps[fs == x][0] if (fs == x).any() else 0
        if ok.any():
            j = np.argmax(np.where(ok, ps, -1))
            if ps[j] > px + margin: new[k] = fs[j]
        new[k] = max(new[k], prev + 1); prev = new[k]
    vals = p[np.r_[0, ch]]; b = np.r_[0, new, len(p)]
    for s in range(len(vals)): q[b[s]:b[s + 1]] = vals[s]
    return q


c = joblib.load(E.CACHE); D = M.data2()
S = {}
for mi, m in enumerate(TUNE):
    fs = [np.load(E.LOG / "v8" / f"m2_s{s}_{m}.npz") for s in range(20, 25)]
    S[m] = []
    for i in range(2):
        h = c[m][i]; n = len(h["truth"])
        lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
        b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
        p = V.viterbi_scaled(0.5 * lp + 0.5 * h["eR"], 20.0, 5.0, 1 - 0.6 * b)
        t = merge(np.asarray(h["truth"]), 2)
        X, meta, ch = feats(D[m][i], lp, b, h["eR"], p)
        S[m].append(dict(p=p, t=t, X=X, meta=meta, ch=ch, y=targets(meta, p, t), raw=np.asarray(h["truth"])))
    prog(f"A: features built ({mi + 1}/5)")
del D

res = {"design": {}, "variant": {}}
margins = {"design": 0.05, "variant": 0.15}
base = {"mt": [], "dfl": []}; ref = {k: {"mt": [], "dfl": []} for k in margins}
for fi, m in enumerate(TUNE):
    prog(f"A: LOMO fold {fi + 1}/5")
    tr = [s for k in TUNE if k != m for s in S[k]]
    X = np.concatenate([s["X"] for s in tr]); y = np.concatenate([s["y"] for s in tr])
    clf = GB(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=0).fit(X, y)
    pr = [clf.predict_proba(s["X"])[:, 1] for s in S[m]]
    mt = [dict(truth=s["t"]) for s in S[m]]; dfl = [dict(truth=s["raw"]) for s in S[m]]
    base["mt"].append(score(mt, [s["p"] for s in S[m]])["spell_f1_mean"]); base["dfl"].append(score(dfl, [s["p"] for s in S[m]])["spell_f1_mean"])
    for k, mg in margins.items():
        q = [refine(s["p"], s["meta"], s["ch"], pp, mg) for s, pp in zip(S[m], pr)]
        ref[k]["mt"].append(score(mt, q)["spell_f1_mean"]); ref[k]["dfl"].append(score(dfl, q)["spell_f1_mean"])
bm = np.array(base["mt"])
out = {"matches": list(TUNE), "v8_merged": base["mt"], "v8_dfl": base["dfl"]}
for k in margins:
    a = np.array(ref[k]["mt"]); w = int((a > bm).sum())
    out[k] = dict(margin=margins[k], merged=ref[k]["mt"], dfl=ref[k]["dfl"], mean_merged=float(a.mean()), mean_dfl=float(np.mean(ref[k]["dfl"])),
                  wins=w, gate=bool(a.mean() >= 0.7663 and w >= 4))
    print(k, f"merged {a.mean():.4f}", " ".join(f"{x:.3f}" for x in a), f"wins {w}/5 dfl {np.mean(ref[k]['dfl']):.4f} gate {out[k]['gate']}")
print(f"v8 merged {bm.mean():.4f} dfl {np.mean(base['dfl']):.4f}")
out["v8_mean_merged"] = float(bm.mean()); out["v8_mean_dfl"] = float(np.mean(base["dfl"]))
json.dump(out, open(E.LOG.parent / "results" / "v10_refine.json", "w"), indent=1)
best = max(margins, key=lambda k: out[k]["mean_merged"])
prog(f"A: done {'PASS' if any(out[k]['gate'] for k in margins) else 'FAIL'} {out[best]['mean_merged']:.4f}")
