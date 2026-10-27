"""Integration experiments (leave-one-match-out, nested), IDSSE tuning matches only.

    python -m possession.integrate_run --cond seed0 --runs s1_rules s1_players s1_ball s2_rules s2_players s2_ball
    python -m possession.integrate_run --cond seed1 --runs s2_players
    python -m possession.integrate_run --cond noise0.3 --runs s2_players         # train + tune on noisy data
    python -m possession.integrate_run --cond noise0.3 --mismatch s2_players     # clean-trained models, clean cells
    python -m possession.integrate_run --freeze s2_players                      # step 4

Runs: s1_* = Viterbi on rules team evidence + D column (rules: safety D column; players: WS4 players-only
model; ball: WS4 with ball features). s2_* = same evidence, forward-backward posterior argmax.
Every learned part is fitted inside folds: for held-out match m the cell (alpha, TEMP, switch costs) is chosen on
the other 4 matches using p_dead from models trained on the 3 matches that are neither m nor that match; the
score on m uses the model trained on the 4 others.
"""
import argparse
import itertools
import json
import time
import zlib

import joblib
import numpy as np
from joblib import Parallel, delayed

from . import dead, pipeline as pl
from .data import TUNE, idsse
from .lomo import LOG_DIR
from .run_idsse import load
from .score import score

SUB = 5
CACHE = idsse.DATA / "cache" / "integrate"
GRIDS = {
    "s1": {"ALPHA": [0.25, 0.5, 1.0, 1.5, 2.0], "TEMP": [1.0], "SWITCH_TEAM": [5.0, 10.0, 20.0, 40.0],
           "SWITCH_DEAD": [2.5, 5.0, 10.0, 20.0, 40.0, 80.0], "DECODER": ["viterbi"]},
    "s2": {"ALPHA": [0.25, 0.5, 1.0], "TEMP": [1.0, 2.0, 3.0], "SWITCH_TEAM": [5.0, 10.0, 20.0, 40.0],
           "SWITCH_DEAD": [5.0, 10.0, 20.0], "DECODER": ["argmax"]},
}
RULES_GRID = {"s1": {"ALPHA": [0.0], "TEMP": [1.0], "SWITCH_TEAM": [5.0, 10.0, 20.0, 40.0],
                     "SWITCH_DEAD": [2.5, 5.0, 10.0, 20.0, 40.0, 80.0], "DECODER": ["viterbi"]},
              "s2": {"ALPHA": [0.0], "TEMP": [1.0, 2.0, 3.0], "SWITCH_TEAM": [5.0, 10.0, 20.0, 40.0],
                     "SWITCH_DEAD": [5.0, 10.0, 20.0, 40.0], "DECODER": ["argmax"]}}


def cells_of(grid):
    return [dict(zip(grid, v)) for v in itertools.product(*grid.values())]


def cell_key(c):
    return json.dumps(c, sort_keys=True)


# --- data and features -------------------------------------------------------------------------

def load_cond(cond):
    if cond.startswith("seed"):
        return {m: load(m, seed=int(cond[4:])) for m in TUNE}
    sigma = float(cond[5:])
    out = {}
    for m in TUNE:
        halves = []
        for i, h in enumerate(load(m)):
            rng = np.random.default_rng([zlib.crc32(m.encode()), i, 77, int(round(sigma * 100))])
            ball = h["ball"].copy()
            ball += (rng.standard_normal(ball.shape) * sigma).astype(np.float32) * (~np.isnan(ball))
            halves.append({**h, "ball": ball})
        out[m] = halves
    return out


def _prep(half):
    cache = {}
    p = pl.base_params()
    pl.team_evidence(half, p, cache)
    X, names = dead.build_features(half)
    return {"team": cache["team"], "e_rules": cache["e_rules"], "X": X, "names": names,
            "y": dead.labels(half), "truth": half["truth"]}


def prep_all(data):
    jobs = [(m, i) for m in TUNE for i in range(len(data[m]))]
    res = Parallel(n_jobs=4)(delayed(_prep)(data[m][i]) for m, i in jobs)
    P = {m: [] for m in TUNE}
    for (m, i), r in zip(jobs, res):
        P[m].append(r)
    return P


def fit(Xs, ys, keep):
    X = np.concatenate([x[::SUB][:, keep] for x in Xs])
    y = np.concatenate([a[::SUB] for a in ys])
    return dead.make_model().fit(X, y)


def train_on(P, ms, keep):
    return fit([h["X"] for m in ms for h in P[m]], [h["y"] for m in ms for h in P[m]], keep)


def predict_p(model, P, m, keep):
    return [model.predict_proba(h["X"][:, keep])[:, 1] for h in P[m]]


def pdead_tables(P, cond, variant, tag):
    """outer[m]: p_dead on m from a model trained on the other 4; inner[(m, k)]: p_dead on k from the model
    trained on the 3 matches that are neither m nor k. Also saves the outer models for step 4 / mismatch."""
    path = CACHE / f"pdead_{tag}_{variant}.pkl"
    if path.exists():
        return joblib.load(path)
    keep = pl.keep_columns(P[TUNE[0]][0]["names"], variant)
    outer, inner = {}, {}
    for m in TUNE:
        mod = train_on(P, [k for k in TUNE if k != m], keep)
        outer[m] = predict_p(mod, P, m, keep)
        joblib.dump({"model": mod, "keep": keep}, CACHE / f"outer_{tag}_{variant}_{m}.joblib")
    for a, b in itertools.combinations(TUNE, 2):
        mod = train_on(P, [k for k in TUNE if k not in (a, b)], keep)
        inner[(a, b)] = predict_p(mod, P, b, keep)   # model excludes both; predicts b (tuning on a held out)
        inner[(b, a)] = predict_p(mod, P, a, keep)
    out = {"outer": outer, "inner": inner}
    joblib.dump(out, path)
    return out


# --- evaluation --------------------------------------------------------------------------------

def _decode_cell(team, e_rules, pds, cell):
    out = []
    for i in range(len(team)):
        e = e_rules[i] if pds is None else pl.combine(team[i], pds[i], cell["ALPHA"])
        out.append(pl.decode(e, cell))
    return out


def _eval_cells(truths, team, e_rules, pds, cells):
    halves = [{"truth": t} for t in truths]
    res = []
    for c in cells:
        r = score(halves, _decode_cell(team, e_rules, pds, c))
        res.append({"spell": r["spell_f1_mean"], "macro": r["macro_f1"],
                    "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]})
    return res


def light(P):
    return {m: [{k: h[k] for k in ("team", "e_rules", "truth")} for h in P[m]] for m in P}


def _task(L, m, pds, cells):
    return _eval_cells([h["truth"] for h in L[m]], [h["team"] for h in L[m]], [h["e_rules"] for h in L[m]],
                       pds, cells)


def run_variant(P, tables, cells, name):
    """tables None = rules-only (no dead model). Returns the report dict (also printed and written)."""
    t0 = time.time()
    tasks = [(m, m, None) for m in TUNE] + [(m, k, (m, k)) for m in TUNE for k in TUNE if k != m]
    L = light(P)
    res = Parallel(n_jobs=4)(delayed(_task)(
        {k: L[k]}, k, None if tables is None else (tables["outer"][m] if key is None else tables["inner"][key]),
        cells) for m, k, key in tasks)
    outer, inner = {}, {m: {} for m in TUNE}
    for (m, k, key), r in zip(tasks, res):
        if key is None:
            outer[m] = r
        else:
            inner[m][k] = r
    n = len(cells)
    held = {}
    for m in TUNE:
        score_c = [np.mean([inner[m][k][c]["spell"] for k in TUNE if k != m]) for c in range(n)]
        ci = int(np.argmax(score_c))
        held[m] = {"cell": cells[ci], **outer[m][ci]}
    spell = np.array([h["spell"] for h in held.values()])
    # best fixed cell by the inner scores pooled over all 20 (m, k) pairs, and its outer (context) score
    pooled = [np.mean([inner[m][k][c]["spell"] for m in TUNE for k in TUNE if k != m]) for c in range(n)]
    fi = int(np.argmax(pooled))
    fixed = {"cell": cells[fi], "inner_pooled_spell": float(pooled[fi]),
             "outer_spell_per_match": [outer[m][fi]["spell"] for m in TUNE],
             "outer_spell_mean": float(np.mean([outer[m][fi]["spell"] for m in TUNE])),
             "outer_macro_mean": float(np.mean([outer[m][fi]["macro"] for m in TUNE]))}
    rep = {"name": name, "lomo_spell_mean": float(spell.mean()), "lomo_spell_min": float(spell.min()),
           "lomo_spell_max": float(spell.max()),
           "lomo_macro_mean": float(np.mean([h["macro"] for h in held.values()])),
           "held_out": held, "fixed_inner_best": fixed, "grid": {k: sorted(set(c[k] for c in cells)) for k in cells[0]},
           "seconds": time.time() - t0}
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / f"integrate_{name}.json").write_text(json.dumps(rep, indent=1))
    show(rep)
    return rep


def run_reference(data, cond):
    from . import safety
    base = {**safety.PARAMS, **json.loads((pl.HERE / "params.json").read_text())}
    held = {}
    for m in TUNE:
        r = score(data[m], [safety.predict(h, base) for h in data[m]])
        held[m] = {"cell": base, "spell": r["spell_f1_mean"], "macro": r["macro_f1"],
                   "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
    spell = np.array([h["spell"] for h in held.values()])
    rep = {"name": f"ref_{cond}", "lomo_spell_mean": float(spell.mean()), "lomo_spell_min": float(spell.min()),
           "lomo_spell_max": float(spell.max()), "lomo_macro_mean": float(np.mean([h["macro"] for h in held.values()])),
           "held_out": held, "fixed_inner_best": {"cell": None, "outer_spell_mean": float("nan")}, "seconds": 0}
    (LOG_DIR / f"integrate_{rep['name']}.json").write_text(json.dumps(rep, indent=1))
    show(rep)


def show(rep):
    for m, h in rep["held_out"].items():
        print(f"  {m} spell {h['spell']:.4f} macro {h['macro']:.4f} spells {h['n_pred']}/{h['n_gt']} cell {cell_key(h['cell'])}")
    print(f"{rep['name']}: LOMO spellF1 {rep['lomo_spell_mean']:.4f} [{rep['lomo_spell_min']:.4f}, "
          f"{rep['lomo_spell_max']:.4f}] macroF1 {rep['lomo_macro_mean']:.4f}  "
          f"fixed-inner-best {cell_key(rep['fixed_inner_best']['cell'])} outer {rep['fixed_inner_best']['outer_spell_mean']:.4f} "
          f"({rep['seconds']:.0f}s)", flush=True)


def run_mismatch(P, cond, run, clean_tag="seed0"):
    """Models trained on clean data (saved outer folds), cells chosen on clean data; scored on this noisy condition."""
    step, variant = run.split("_")
    clean = json.loads((LOG_DIR / f"integrate_{run}_{clean_tag}.json").read_text())
    keep_all = None
    held = {}
    for m in TUNE:
        cell = clean["held_out"][m]["cell"]
        sv = joblib.load(CACHE / f"outer_{clean_tag}_{variant}_{m}.joblib")
        pds = predict_p(sv["model"], P, m, sv["keep"])
        r = _task(light({m: P[m]}), m, pds, [cell])[0]
        held[m] = {"cell": cell, **r}
    spell = np.array([h["spell"] for h in held.values()])
    rep = {"name": f"{run}_{cond}_mismatch", "lomo_spell_mean": float(spell.mean()), "lomo_spell_min": float(spell.min()),
           "lomo_spell_max": float(spell.max()), "lomo_macro_mean": float(np.mean([h["macro"] for h in held.values()])),
           "held_out": held, "fixed_inner_best": {"cell": None, "outer_spell_mean": float("nan")}, "seconds": 0}
    (LOG_DIR / f"integrate_{rep['name']}.json").write_text(json.dumps(rep, indent=1))
    show(rep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", default="seed0")
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--mismatch", nargs="*", default=[])
    ap.add_argument("--freeze", default=None)
    ap.add_argument("--ref", action="store_true", help="reference safety net (params.json) on this condition")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if a.freeze:
        return freeze(a.freeze)
    t0 = time.time()
    data = load_cond(a.cond)
    P = prep_all(data)
    print(f"{a.cond}: features built {time.time() - t0:.0f}s, {len(P[TUNE[0]][0]['names'])} features", flush=True)
    if a.ref:
        run_reference(data, a.cond)
    for run in a.mismatch:
        run_mismatch(P, a.cond, run)
    for run in a.runs:
        step, variant = run.split("_")
        if variant == "rules":
            run_variant(P, None, cells_of(RULES_GRID[step]), f"{run}_{a.cond}")
        else:
            tables = pdead_tables(P, a.cond, variant, a.cond)
            run_variant(P, tables, cells_of(GRIDS[step]), f"{run}_{a.cond}")


def freeze(run):
    """Step 4: params_v2.json from the best fixed cell on the pooled inner folds; final model on all 5 matches."""
    import sklearn
    step, variant = run.split("_")
    rep = json.loads((LOG_DIR / f"integrate_{run}_seed0.json").read_text())
    cell = rep["fixed_inner_best"]["cell"]
    data = load_cond("seed0")
    P = prep_all(data)
    names = P[TUNE[0]][0]["names"]
    keep = pl.keep_columns(names, variant)
    model = train_on(P, list(TUNE), keep)
    models = {"model": model, "keep": keep, "features": [names[i] for i in keep], "variant": variant,
              "sub": SUB, "trained_on": list(TUNE), "windows_s": list(dead.WINDOWS_S),
              "versions": {"sklearn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__}}
    joblib.dump(models, pl.HERE / "models" / "dead_v2.joblib")
    params = {**pl.RULES, **cell, "variant": variant,
              "selection": "best fixed cell by mean spellF1 over the 20 inner (match, match) folds, models not seeing either match"}
    (pl.HERE / "params_v2.json").write_text(json.dumps(params, indent=1))
    print("frozen", json.dumps(params), "features", len(keep))


if __name__ == "__main__":
    main()
