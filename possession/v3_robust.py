"""v3 robustness: v2 (optionally plus post/evidence ops) under degrade2 perturbations, LOMO on the 5 tuning matches.

    python -m possession.v3_robust --cond clean wrong_team0.02 ball_noise0.3 dead_missing0.7 --mode mismatch
    python -m possession.v3_robust --cond wrong_team0.02 --mode matched

mismatch: fold models (trained on clean seed-0 data, saved by integrate_run) and the v2 fold cells, scored on the
perturbed halves.  matched: p_dead models retrained inside folds on the perturbed halves (cells stay the clean v2
fold cells; they are not re-tuned).  Optional --ops a|b|combo applies a candidate from a params json on top.
"""
import argparse
import json
import time

import joblib
import numpy as np
from joblib import Parallel, delayed

from . import dead
from . import degrade2
from . import pipeline as pl
from .data import TUNE
from .lomo import LOG_DIR
from .run_idsse import load
from .v3_lib import CACHE, fmt_row, spell_score, summary, v2_cells
from .v3_post import arrival_dist, cleanup, snap


def perturbed(cond):
    if cond == "clean":
        return {m: load(m) for m in TUNE}
    return {m: degrade2.apply(cond, load(m), m) for m in TUNE}


def _mismatch_task(m, halves, cell, ops):
    sv = joblib.load(CACHE / f"outer_seed0_players_{m}.joblib")
    pds = []
    for h in halves:
        X, _ = dead.build_features(h)
        pds.append(sv["model"].predict_proba(X[:, sv["keep"]])[:, 1])
    return m, finish(m, halves, pds, cell, ops)


def finish(m, halves, pds, cell, ops):
    paths = []
    for h, pd in zip(halves, pds):
        cache = {}
        team = pl.team_evidence(h, pl.base_params(), cache)
        e = pl.combine(team, pd, cell["ALPHA"])
        path = pl.decode(e, cell)
        if ops and ops.get("snap"):
            path = snap(path, arrival_dist(h), ops["snap"]["N"], ops["snap"]["R"])
        if ops and ops.get("cleanup"):
            path = cleanup(path, ops["cleanup"]["K"], ops["cleanup"]["L"])
        paths.append(path)
    return spell_score([{"truth": h["truth"]} for h in halves], paths)


def run_mismatch(cond, ops=None):
    data = perturbed(cond)
    cells = v2_cells()
    out = Parallel(n_jobs=3)(delayed(_mismatch_task)(m, data[m], cells[m], ops) for m in TUNE)
    return dict(out)


def run_matched(cond, ops=None):
    from . import integrate_run as ir
    data = perturbed(cond)
    jobs = [(m, i) for m in TUNE for i in range(len(data[m]))]
    res = Parallel(n_jobs=3)(delayed(ir._prep)(data[m][i]) for m, i in jobs)
    P = {m: [] for m in TUNE}
    for (m, i), r in zip(jobs, res):
        P[m].append(r)
    tables = ir.pdead_tables(P, f"v3_{cond}", "players", f"v3_{cond}")
    cells = v2_cells()
    return {m: finish(m, data[m], tables["outer"][m], cells[m], ops) for m in TUNE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cond", nargs="+", required=True)
    ap.add_argument("--mode", choices=["mismatch", "matched"], default="mismatch")
    ap.add_argument("--ops", default=None, help="json file with {'snap': {N, R}, 'cleanup': {K, L}}")
    ap.add_argument("--tag", default="v2")
    a = ap.parse_args()
    ops = json.loads(open(a.ops).read()) if a.ops else None
    for cond in a.cond:
        t0 = time.time()
        r = (run_mismatch if a.mode == "mismatch" else run_matched)(cond, ops)
        s = summary(r)
        name = f"{a.tag}_{cond}_{a.mode}"
        (LOG_DIR / f"v3_robust_{name}.json").write_text(json.dumps({"summary": s, "per_match": r}, indent=1))
        print(fmt_row(name, s), f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
