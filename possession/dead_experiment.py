"""WS4 experiment: LOMO dead-ball classifier, then LOMO spellF1 with alpha * logit(p_dead) emissions.

    python -m possession.dead_experiment

Nested: for each outer held-out match m, alpha / switch costs are chosen on the other 4 matches using p_dead
predicted by models that saw neither m nor that inner match (train on the remaining 3). The outer score uses
the model trained on the 4 others. Final model: all 5 tuning matches -> possession/models/dead.joblib.
"""
import itertools
import json
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from . import dead, safety
from .data import TUNE
from .lomo import LOG_DIR
from .run_idsse import load
from .score import score

HERE = Path(__file__).parent
SUB = 5            # train on every 5th frame
GRID_ALPHA = (0.5, 1.0, 1.5, 2.0)
GRID_ST = (10.0, 20.0, 30.0)
GRID_SD = (5.0, 10.0, 20.0, 40.0)
REF = {"spell_mean": 0.3017, "spell_min": 0.2804, "spell_max": 0.3259, "macro_mean": 0.7573}


def fit(Xs, ys):
    X = np.concatenate([x[::SUB] for x in Xs])
    y = np.concatenate([a[::SUB] for a in ys])
    return dead.make_model().fit(X, y)


def predict(model, Xs):
    return [model.predict_proba(x)[:, 1] for x in Xs]


def auc_f1(y, p):
    from sklearn.metrics import f1_score, roc_auc_score
    return float(roc_auc_score(y, p)), float(f1_score(y, p > 0.5))


def decode_cell(half_f, half_p, alpha, st, sd, params):
    out = []
    for f, p in zip(half_f, half_p):
        out.append(safety.viterbi(dead.emissions(f, p, alpha, params), st, sd))
    return out


def eval_cell(data_m, feats_m, probs_m, cell, params):
    alpha, st, sd = cell
    preds = decode_cell(feats_m, probs_m, alpha, st, sd, params)
    r = score(data_m, preds)
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}


def main():
    t0 = time.time()
    params = json.loads((HERE / "params.json").read_text())
    params = {**safety.PARAMS, **params}
    data = {m: load(m) for m in TUNE}
    F, Y, names = {}, {}, None
    for m in TUNE:
        built = [dead.build_features(h) for h in data[m]]
        F[m] = [b[0] for b in built]
        names = built[0][1]
        Y[m] = [dead.labels(h) for h in data[m]]
    SF = {m: [safety.features(h) for h in data[m]] for m in TUNE}
    print(f"features {len(names)}  built in {time.time() - t0:.0f}s", flush=True)

    def train_on(ms):
        return fit([x for m in ms for x in F[m]], [y for m in ms for y in Y[m]])

    # outer LOMO classifier
    outer_p, report = {}, {}
    models = {}
    for m in TUNE:
        rest = [k for k in TUNE if k != m]
        models[m] = train_on(rest)
        outer_p[m] = predict(models[m], F[m])
        y, p = np.concatenate(Y[m]), np.concatenate(outer_p[m])
        auc, f1 = auc_f1(y, p)
        # speed-only baseline for the same metric: lower mean speed -> dead
        sp = np.concatenate([-np.nan_to_num(f["speed"], nan=params["V0"]) for f in SF[m]])
        from sklearn.metrics import roc_auc_score
        report[m] = {"auc": auc, "f1_D": f1, "dead_share": float(y.mean()), "auc_speed_only": float(roc_auc_score(y, sp))}
        print(m, report[m], f"{time.time() - t0:.0f}s", flush=True)

    # inner predictions for tuning: model trained without m and k predicts k
    inner_p = {}
    for m in TUNE:
        for k in TUNE:
            if k == m:
                continue
            mod = train_on([j for j in TUNE if j not in (m, k)])
            inner_p[(m, k)] = predict(mod, F[k])
        print("inner done", m, f"{time.time() - t0:.0f}s", flush=True)

    cells = list(itertools.product(GRID_ALPHA, GRID_ST, GRID_SD))
    # inner grid: tasks (m, k, cell)
    tasks = [(m, k, c) for m in TUNE for k in TUNE if k != m for c in cells]
    res = Parallel(n_jobs=4)(delayed(eval_cell)(data[k], SF[k], inner_p[(m, k)], c, params) for m, k, c in tasks)
    inner = {}
    for (m, k, c), r in zip(tasks, res):
        inner.setdefault(m, {}).setdefault(c, []).append(r["spell"])
    print(f"inner grid done {time.time() - t0:.0f}s", flush=True)

    chosen = {m: max(cells, key=lambda c: np.mean(inner[m][c])) for m in TUNE}
    outer = Parallel(n_jobs=4)(delayed(eval_cell)(data[m], SF[m], outer_p[m], chosen[m], params) for m in TUNE)
    held = {}
    for m, r in zip(TUNE, outer):
        held[m] = {"alpha_st_sd": list(chosen[m]), **r, **{"auc": report[m]["auc"], "f1_D": report[m]["f1_D"]}}
        print(m, held[m], flush=True)
    spell = np.array([h["spell"] for h in held.values()])
    summary = {
        "lomo_spell_mean": float(spell.mean()), "lomo_spell_min": float(spell.min()), "lomo_spell_max": float(spell.max()),
        "lomo_macro_mean": float(np.mean([h["macro"] for h in held.values()])),
        "spells_pred_gt": [[h["n_pred"], h["n_gt"]] for h in held.values()],
    }
    print(summary)

    # fixed-params sensitivity: same alpha/st/sd on every match (cells scored on outer predictions; for context only)
    allres = Parallel(n_jobs=4)(delayed(eval_cell)(data[m], SF[m], outer_p[m], c, params) for c in cells for m in TUNE)
    sens = {}
    for i, c in enumerate(cells):
        rr = allres[i * len(TUNE):(i + 1) * len(TUNE)]
        sens[str(c)] = float(np.mean([r["spell"] for r in rr]))

    # permutation importance on one held-out fold (the first), using frame AUC
    from sklearn.metrics import roc_auc_score
    m0 = TUNE[0]
    rng = np.random.default_rng(0)
    X0, y0 = np.concatenate(F[m0])[::3], np.concatenate(Y[m0])[::3]
    base = roc_auc_score(y0, models[m0].predict_proba(X0)[:, 1])
    imp = []
    for j, nm in enumerate(names):
        Xp = X0.copy()
        Xp[:, j] = rng.permutation(Xp[:, j])
        imp.append(base - roc_auc_score(y0, models[m0].predict_proba(Xp)[:, 1]))
    order = np.argsort(imp)[::-1][:20]
    top = [[names[j], float(imp[j])] for j in order]
    print("top importances (AUC drop, fold", m0, ")", top[:10])

    # final model on all 5
    import joblib
    import sklearn
    final = train_on(list(TUNE))
    (HERE / "models").mkdir(exist_ok=True)
    joblib.dump({"model": final, "features": names, "sub": SUB, "trained_on": list(TUNE),
                 "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
                 "windows_s": list(dead.WINDOWS_S)}, HERE / "models" / "dead.joblib")

    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "dead.json").write_text(json.dumps({
        "reference": REF, "classifier": report, "held_out": held, "summary": summary,
        "grid": {"alpha": GRID_ALPHA, "switch_team": GRID_ST, "switch_dead": GRID_SD},
        "fixed_cell_mean_spell_on_outer_preds": sens, "perm_importance_fold": [m0, base, top],
        "n_features": len(names), "seconds": time.time() - t0}, indent=1))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
