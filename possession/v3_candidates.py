"""v3 single candidates on top of frozen v2, LOMO on the 5 IDSSE tuning matches (seed-0 degraded halves).

    python -m possession.v3_candidates --cand base
    python -m possession.v3_candidates --cand a          # boundary snap (N, R grid)
    python -m possession.v3_candidates --cand b          # cleanup (K, L grid)
    python -m possession.v3_candidates --cand c --sigma 0.5 --seed 0   # restart evidence, simulated tags

Candidate parameters are chosen per held-out match on the inner folds (the other 4 matches, p_dead from models
that saw neither match, using the held-out match's v2 cell); the score is on the held-out match with the model
trained on the other 4. Logs: possession/logs/v3_<cand>[_s<sigma>_seed<k>].json
"""
import argparse
import itertools
import json
import time

import numpy as np
from joblib import Parallel, delayed

from . import pipeline as pl
from . import restarts as R
from .data import TUNE
from .lomo import LOG_DIR
from .run_idsse import load
from .v3_lib import fmt_row, pdead_cached, spell_score, summary, team_of, v2_cells
from .v3_post import arrival_dist, cleanup, snap

GRIDS = {
    "a": {"N": [0, 10, 25, 50, 75], "R": [0.75, 1.0, 1.5, 2.5]},
    "b": {"K": [0, 25, 50, 100, 150, 250], "L": [0, 10, 25, 50, 75]},
    "c": {"PRE": [75, 200, 400], "W_D": [0.0, 1.0, 2.0, 5.0], "POST": [25, 75], "W_T": [0.0, 1.0, 3.0]},
}
NOOP = {"a": {"N": 0, "R": 1.0}, "b": {"K": 0, "L": 0}, "c": {"PRE": 75, "W_D": 0.0, "POST": 25, "W_T": 0.0}}


def cells_of(cand):
    g = GRIDS[cand]
    cells = [dict(zip(g, v)) for v in itertools.product(*g.values())]
    noop = NOOP[cand]
    # a no-op cell first so ties go to "leave v2 alone"
    cells = [c for c in cells if not is_noop(cand, c)]
    return [noop] + cells


def is_noop(cand, c):
    if cand == "a":
        return c["N"] == 0
    if cand == "b":
        return c["K"] == 0 and c["L"] == 0
    return c["W_D"] == 0 and c["W_T"] == 0


def ckey(c):
    return json.dumps(c, sort_keys=True)


def prep_inputs():
    """Per match, per half: truth, team evidence, arrival distances, half itself is not kept."""
    out = {}
    for m in TUNE:
        hs = load(m)
        out[m] = [{"truth": h["truth"], "team": team_of(h), "dist": arrival_dist(h).astype(np.float32),
                   "fp": R.fingerprint(h)} for h in hs]
    return out


def evidence(h, pd, cell):
    return pl.combine(h["team"], pd, cell["ALPHA"])


def unit_paths(halves, pds, cell, cand, c, tags):
    """Decoded (and post-processed) paths for one match's halves."""
    out = []
    for h, pd in zip(halves, pds):
        e = evidence(h, pd, cell)
        if cand == "c" and not is_noop("c", c):
            e = R.adjust(e, tags[h["fp"]], c)
        path = pl.decode(e, cell)
        if cand == "a" and c["N"] > 0:
            path = snap(path, h["dist"], c["N"], c["R"])
        if cand == "b":
            path = cleanup(path, c["K"], c["L"])
        out.append(path)
    return out


def task(m, P, tables, cells_cell, cand, cells, tags):
    cell = cells_cell
    res = {}
    units = [("outer", m, tables["outer"][m])] + [(k, k, tables["inner"][(m, k)]) for k in TUNE if k != m]
    base_cache = {}
    for c in cells:
        key = ckey(c)
        res[key] = {}
        for name, k, pds in units:
            paths = unit_paths(P[k], pds, cell, cand, c, tags)
            res[key][name] = spell_score([{"truth": h["truth"]} for h in P[k]], paths)
    return m, res


def run(cand, tags=None, name=None):
    t0 = time.time()
    P = prep_inputs()
    tables = pdead_cached("seed0")
    cells_by_m = v2_cells()
    cells = cells_of(cand) if cand != "base" else [{}]
    gridc = cells_by_m
    out = Parallel(n_jobs=3)(delayed(task)(m, P, tables, gridc[m], cand if cand != "base" else "b", cells if cand != "base" else [{"K": 0, "L": 0}], tags) for m in TUNE)
    res = dict(out)
    keys = [ckey(c) for c in (cells if cand != "base" else [{"K": 0, "L": 0}])]
    held, outer_all = {}, {}
    for m in TUNE:
        inner_mean = [np.mean([res[m][k][kk]["spell"] for kk in TUNE if kk != m]) for k in keys]
        ci = int(np.argmax(inner_mean))
        held[m] = {"cell": json.loads(keys[ci]), **res[m][keys[ci]]["outer"], "inner_mean": float(inner_mean[ci])}
    # descriptive: every cell's outer score per match and mean
    table = []
    for k in keys:
        sp = [res[m][k]["outer"]["spell"] for m in TUNE]
        table.append({"cell": json.loads(k), "outer_spell": sp, "outer_mean": float(np.mean(sp)),
                      "outer_macro": float(np.mean([res[m][k]["outer"]["macro"] for m in TUNE])),
                      "inner_pooled": float(np.mean([res[m][k][kk]["spell"] for m in TUNE for kk in TUNE if kk != m]))})
    fi = int(np.argmax([t["inner_pooled"] for t in table]))
    rep = {"name": name or cand, "lomo": summary(held), "held_out": held, "fixed_inner_best": table[fi], "table": table,
           "seconds": time.time() - t0}
    (LOG_DIR / f"v3_{name or cand}.json").write_text(json.dumps(rep, indent=1))
    print(fmt_row(name or cand, rep["lomo"]))
    print("  chosen", {m: held[m]["cell"] for m in TUNE})
    print("  fixed inner-best", table[fi]["cell"], "outer mean", round(table[fi]["outer_mean"], 4),
          "per match", [round(x, 3) for x in table[fi]["outer_spell"]], f"({rep['seconds']:.0f}s)", flush=True)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True, choices=["base", "a", "b", "c"])
    ap.add_argument("--sigma", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", type=float, default=0.15)
    a = ap.parse_args()
    if a.cand == "c":
        from .restarts_run import build_tags
        tags, stats = build_tags(a.sigma, a.seed, a.drop)
        print("tags", json.dumps(stats), flush=True)
        run("c", tags, f"c_s{a.sigma}_seed{a.seed}")
    else:
        run(a.cand)


if __name__ == "__main__":
    main()
