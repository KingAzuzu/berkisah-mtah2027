"""v5 study: tag simulator calibrated on label-free agreement, fusion grid on IDSSE LOMO (v4 cB seeds 10-14 + hook folds).
python -m possession.v5_run  -> logs/v5/*.json, logs/v5_progress.txt.  IDSSE TUNE only; CPU Viterbi, no TCN inference
(fold evidence comes from the saved LOMO npz). Rule fixed before running: adopt iff mean spellF1 >= v4 + 0.02 and wins >= 4/5."""
import json, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import joblib

from . import tagfuse as TF
from .data import TUNE, idsse

LOG = Path(__file__).parent / "logs"
CACHE = idsse.DATA / "cache" / "v5_folds.pkl"
KS = (0.0, 0.5, 1.0, 2.0)
SIM_SEEDS = (0, 1, 2)
T0 = time.time()
STAGES = ["fold cache + v4 reference", "calibrate simulator", "k grid", "robustness", "report"]
F = None


def progress(k, frac=0.0, done=False):
    pct = 100 if done else int(100 * (k - 1 + frac) / len(STAGES))
    el = time.time() - T0
    eta = "0s" if done else (f"{int(el / max(pct, 1) * (100 - pct) / 60)}min" if pct else "?")
    (LOG / "v5_progress.txt").write_text(("DONE" if done else f"{pct}% | stage {k}/{len(STAGES)} | ETA {eta}") + "\n")


def build_cache():
    """Per match and half: fused evidence e (T,3), hook scale, truth, from cB seeds 10-14 LOMO npz (as v4_eval.run)."""
    from . import m2_lib as L, pipeline as pl, v4_lib as V
    data, P = L.data2(), L.prep2()
    mods, keep = L.r_models(P)
    R = L.r_preds(P, mods, keep)
    p = pl.base_params(L.PARAMS)
    D = pl.HERE / "logs" / "v4"
    out = {}
    for m in TUNE:
        fs = [np.load(D / f"cB_s{s}_{m}.npz") for s in range(10, 15)]
        halves = []
        for i in range(2):
            n = len(data[m][i]["truth"])
            lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
            e = 0.5 * lp + 0.5 * pl.combine(P[m][i]["team"], R[m][0][i], L.PARAMS["ALPHA"])
            b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
            halves.append({"e": e, "scale": 1 - V.HOOK * b, "truth": data[m][i]["truth"],
                           "st": p["SWITCH_TEAM"], "sd": p["SWITCH_DEAD"], "temp": p.get("TEMP", 1.0)})
        out[m] = halves
    joblib.dump(out, CACHE)
    return out


def decode(h, bias=None):
    from . import v4_lib as V
    e = h["e"].copy()
    if bias is not None:
        e[:, :2] += bias
    t = h["temp"]
    return V.viterbi_scaled(e / t, h["st"] / t, h["sd"] / t, h["scale"])


def _job(args):
    m, i, bias = args
    return decode(F[m][i], bias)


def init(f):
    global F
    F = f


def score_all(pool, per_half_bias):
    from .score import score
    jobs = [(m, i, per_half_bias.get((m, i))) for m in TUNE for i in range(2)]
    preds = pool.map(_job, jobs)
    pr = {}
    for j, (m, i, _) in enumerate(jobs):
        pr.setdefault(m, []).append(preds[j])
    res = {}
    for m in TUNE:
        r = score([{"truth": F[m][i]["truth"]} for i in range(2)], pr[m])
        res[m] = (r["spell_f1_mean"], r["macro_f1"])
    return res, pr


def sims(theta, seed):
    out = {}
    for mi, m in enumerate(TUNE):
        for i in range(2):
            h = F[m][i]
            sp, dead = TF.spells(h["truth"])
            rng = np.random.default_rng([seed, mi, i])
            out[(m, i)] = TF.simulate(h["truth"], sp, dead, theta, rng)
    return out


def sim_agree(theta, pred, seed, reps):
    acc = {}
    for r in range(reps):
        for (m, i), iv in sims(theta, 1000 + seed * 100 + r).items():
            for c, (n, a) in TF.agreement(pred[m][i], iv).items():
                x = acc.setdefault(c, [0, 0]); x[0] += n; x[1] += a
    return {c: (v[1] / v[0] if v[0] else float("nan")) for c, v in acc.items()}


def calibrate(pred, shift=0.0, reps=8):
    """Per code: p_flip (or, if the structure alone is already too noisy, q_trunc) so that simulated agreement with v4 hits the
    real table value - shift. Bisection with common random numbers, one code at a time."""
    zero = {k: (0.0, 0.0) for k in TF.CODES}
    theta, rep = dict(zero), {}
    for c in TF.CODES:
        tgt = TF.CODES[c][3] - shift
        one = lambda pf, q: sim_agree({**zero, c: (pf, q)}, pred, 0, reps).get(c, np.nan)
        a0 = one(0.0, 0.0)
        if a0 >= tgt:
            lo, hi = 0.0, 0.8
            for _ in range(12):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if one(mid, 0.0) > tgt else (lo, mid)
            theta[c] = ((lo + hi) / 2, 0.0)
        else:
            lo, hi = 0.0, 1.0
            for _ in range(12):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if one(0.0, mid) < tgt else (lo, mid)   # agreement rises with q
            theta[c] = (0.0, (lo + hi) / 2)
        rep[c] = {"target": tgt, "struct_agree": a0, "p_flip": theta[c][0], "q_trunc": theta[c][1], "sim_agree": one(*theta[c])}
    return theta, rep


def main():
    global F
    (LOG / "v5").mkdir(exist_ok=True)
    progress(1)
    F = joblib.load(CACHE) if CACHE.exists() else build_cache()
    pool = Pool(4, initializer=init, initargs=(F,))
    base, bpred = score_all(pool, {})
    bm = np.array([base[m][0] for m in TUNE])
    print("v4 reference (cB x5 hook): mean %.4f [%.4f,%.4f] (expect 0.6639)" % (bm.mean(), bm.min(), bm.max()), flush=True)
    progress(2)
    theta, rep = calibrate(bpred)
    for c, r in rep.items():
        print(f"{c:22s} target {r['target']:.3f} struct {r['struct_agree']:.3f} p_flip {r['p_flip']:.3f} q {r['q_trunc']:.3f} -> sim {r['sim_agree']:.3f}", flush=True)
    json.dump(rep, open(LOG / "v5" / "calibration.json", "w"), indent=1)
    progress(3)
    rows = []
    for ki, k in enumerate(KS):
        w = TF.weights(k)
        per = []
        for s in SIM_SEEDS:
            iv = sims(theta, s)
            res, _ = score_all(pool, {key: TF.bias(len(F[key[0]][key[1]]["truth"]), v, w) for key, v in iv.items()})
            per.append([res[m][0] for m in TUNE])
            progress(3, (ki * len(SIM_SEEDS) + len(per)) / (len(KS) * len(SIM_SEEDS)))
        a = np.mean(per, 0)
        rows.append({"k": k, "per_match": dict(zip(TUNE, map(float, a))), "mean": float(a.mean()), "min": float(a.min()),
                     "max": float(a.max()), "wins": int(np.sum(a > bm)), "seed_means": [float(np.mean(p)) for p in per]})
        print(f"k={k}: {a.mean():.4f} [{a.min():.4f},{a.max():.4f}] wins {rows[-1]['wins']}/5 seedmeans {rows[-1]['seed_means']}", flush=True)
    best = max((r for r in rows if r["k"] > 0), key=lambda r: r["mean"])
    passed = bool(best["mean"] >= bm.mean() + 0.02 and best["wins"] >= 4)
    progress(4)
    th2, rep2 = calibrate(bpred, shift=0.1)
    rob = []
    for s in SIM_SEEDS:
        iv = sims(th2, s)
        w = TF.weights(best["k"])
        res, _ = score_all(pool, {key: TF.bias(len(F[key[0]][key[1]]["truth"]), v, w) for key, v in iv.items()})
        rob.append([res[m][0] for m in TUNE])
    ra = np.mean(rob, 0)
    robust = {"k": best["k"], "mean": float(ra.mean()), "min": float(ra.min()), "max": float(ra.max()), "wins": int(np.sum(ra > bm)),
              "per_match": dict(zip(TUNE, map(float, ra))), "calibration": rep2}
    print(f"robustness (agreement -0.1, k={best['k']}): {ra.mean():.4f} [{ra.min():.4f},{ra.max():.4f}] wins {robust['wins']}/5", flush=True)
    out = {"v4": {"mean": float(bm.mean()), "min": float(bm.min()), "max": float(bm.max()), "per_match": dict(zip(TUNE, map(float, bm)))},
           "rows": rows, "best_k": best["k"], "pass": passed, "robust": robust, "theta": {c: list(v) for c, v in theta.items()}}
    json.dump(out, open(LOG / "v5" / "grid.json", "w"), indent=1)
    print("PASS" if passed else "FAIL", flush=True)
    progress(5)


if __name__ == "__main__":
    main()
