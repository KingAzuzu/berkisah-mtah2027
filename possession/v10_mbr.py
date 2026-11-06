"""v10 MBR on v8 posteriors (CPU). Fused evidence e = 0.5 TCN logp + 0.5 eR, HMM costs (20, 5) * (1 - 0.6 b_t), as the v8 decoder.
Gibbs posterior over paths = exp(sum e_t/TEMP - scaled switch costs). Candidates: v8 Viterbi, posterior argmax, S=30 FFBS samples;
pick max mean spellF1 against the samples (possession.mbr machinery). Settings stated beforehand: A TEMP 1, B TEMP 2.
Nested LOMO: setting for held-out match chosen by mean spellF1 vs merged truth on the other 4 matches. Score vs merged(K2) truth
(DFL for info). Gate FIXED: mean >= 0.7663 AND >= 4/5 wins vs v8 (0.7563). python -m possession.v10_mbr -> logs/v10/mbr.json"""
import json, math
import numpy as np
from . import v4_lib as V, mbr as M, v7_decoder as E
from .v10_hedge import c, nets, evid, score_vs
from .data import TUNE

ST, SD, S, SETTINGS = 20.0, 5.0, 30, {"A": 1.0, "B": 2.0}
lse3 = M._lse3


def fb(es, sc, st, sd):
    rows, k = es.tolist(), sc.tolist()
    T = len(rows)
    alpha = np.empty((T, 3)); a = rows[0][:]; alpha[0] = a
    cm = lambda t: ((0.0, st * k[t], sd * k[t]), (st * k[t], 0.0, sd * k[t]), (sd * k[t], sd * k[t], 0.0))
    for t in range(1, T):
        c_, r = cm(t), rows[t]
        a = [lse3(a[0] - c_[0][j], a[1] - c_[1][j], a[2] - c_[2][j]) + r[j] for j in range(3)]
        alpha[t] = a
    beta = np.empty((T, 3)); b = [0.0] * 3; beta[T - 1] = b
    for t in range(T - 2, -1, -1):
        c_, r = cm(t + 1), rows[t + 1]
        w = [r[j] + b[j] for j in range(3)]
        b = [lse3(w[0] - c_[i][0], w[1] - c_[i][1], w[2] - c_[i][2]) for i in range(3)]
        beta[t] = b
    logz = float(np.logaddexp.reduce(alpha[-1]))
    return alpha, beta, logz


def ffbs(alpha, sc, st, sd, n, seed=0):
    T = len(alpha)
    k = sc[1:]
    base = np.array([[0, st, sd], [st, 0, sd], [sd, sd, 0]], float)
    cc = base[None] * k[:, None, None]                       # (T-1,i,j)
    rng = np.random.default_rng(seed)
    u = rng.random((T, n))
    lg = alpha[:-1, :, None] - cc
    lg = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)
    cp = np.cumsum(np.exp(lg), axis=1).transpose(0, 2, 1)
    z = np.empty((n, T), np.int8)
    pT = np.exp(alpha[-1] - np.logaddexp.reduce(alpha[-1]))
    cur = np.minimum((u[T - 1][:, None] > np.cumsum(pT)[None]).sum(1), 2)
    z[:, T - 1] = cur
    for t in range(T - 2, -1, -1):
        cur = np.minimum((u[t][:, None] > cp[t][cur]).sum(1), 2)
        z[:, t] = cur
    return z


def mbr_half(e, b, temp):
    sc = 1 - 0.6 * b
    vit = V.viterbi_scaled(e, ST, SD, sc)
    # temperature: scores e/temp, costs unchanged => same as scaling e
    es = e / temp
    alpha, beta, logz = fb(es, sc, ST, SD)
    gamma = np.exp(alpha + beta - logz)
    amax = np.argmax(gamma, axis=1).astype(np.int8)
    smp = ffbs(alpha, sc, ST, SD, S)
    cands = [vit, amax] + list(smp)
    sp = [M.spells_of(x) for x in cands]
    ref = sp[2:]
    util = []
    for ci, s_ in enumerate(sp):
        v = [M.spell_f1(r, s_) for ri, r in enumerate(ref) if ci != 2 + ri]
        util.append(np.mean(v))
    return cands[int(np.argmax(util))], int(np.argmax(util))


if __name__ == "__main__":
    N8 = nets(E.LOG / "v8", "m2")
    base = {m: [V.viterbi_scaled(evid(N8, m, i), ST, SD, 1 - 0.6 * N8[m][i][1]) for i in range(2)] for m in TUNE}
    sm, _ = score_vs(base, "merged"); print("v8", np.mean(sm), flush=True)
    P, picks = {}, {}
    for name, temp in SETTINGS.items():
        P[name] = {}
        for m in TUNE:
            out = [mbr_half(evid(N8, m, i), N8[m][i][1], temp) for i in range(2)]
            P[name][m] = [o[0] for o in out]; picks[f"{name}_{m}"] = [o[1] for o in out]
            print(name, m, "picked", picks[f"{name}_{m}"], flush=True)
        open(E.LOG / "v10" / "progress_B.txt.mbr", "w").write(f"B: MBR setting {name} done")
    res = {"v8": {"merged": sm, "dfl": score_vs(base, "dfl")[0]}, "picks": picks}
    for name in SETTINGS:
        res[name] = {"merged": score_vs(P[name], "merged")[0], "dfl": score_vs(P[name], "dfl")[0]}
        print(name, "merged", np.mean(res[name]["merged"]), "dfl", np.mean(res[name]["dfl"]))
    nested, ch = [], []
    for j, m in enumerate(TUNE):
        oth = [k for k in range(5) if k != j]
        best = max(SETTINGS, key=lambda n: np.mean([res[n]["merged"][k] for k in oth]))
        nested.append(res[best]["merged"][j]); ch.append(best)
    wins = int(sum(a > b for a, b in zip(nested, sm)))
    res["nested"] = {"per_match": nested, "mean": float(np.mean(nested)), "choice": ch, "wins_vs_v8": wins,
                     "pass": bool(np.mean(nested) >= 0.7663 and wins >= 4)}
    print(res["nested"])
    json.dump(res, open(E.LOG / "v10" / "mbr.json", "w"), indent=1)
