"""LOMO experiment for the HSMM decoder (WS1). Durations and transitions are fitted inside each fold.

    python -m possession.hsmm_experiment            # full run, writes possession/logs/hsmm.json
    python -m possession.hsmm_experiment --time     # decode-time check on one tuning half

Nested selection: for held-out match m, each grid cell is scored on every other match k with a model
fitted on the 3 matches that are neither m nor k; the cell with the best mean is then used on m with a
model fitted on the 4 others. Held-out truth never enters fitting or selection.
"""
import argparse
import itertools
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import hsmm, safety
from .data import TUNE
from .run_idsse import load
from .score import score

LOG = Path(__file__).with_name("logs")
PARAMS = json.loads(Path(__file__).with_name("params.json").read_text())
GRID = {"wd": [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0], "wt": [0.0, 1.0, 2.0]}
CELLS = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
_C = {}


def evid(mid):
    if mid not in _C:
        halves = load(mid)
        _C[mid] = (halves, [safety.emissions(safety.features(h), {**safety.PARAMS, **PARAMS}) for h in halves])
    return _C[mid]


def run_task(args):
    """Fit on `train`, decode `test` at every grid cell. Returns per cell: score dict and run lengths."""
    train, test = args
    model = hsmm.fit([h["truth"] for m in train for h in evid(m)[0]])
    halves, ev = evid(test)
    out = []
    for c in CELLS:
        preds = [hsmm.decode(e, model, **c) for e in ev]
        r = score(halves, preds)
        runs = [np.concatenate([[s] * 1 for s in []]) for _ in range(0)]
        lens = {j: [] for j in range(3)}
        for p in preds:
            st, ln = hsmm._runs(p)
            for s, n in zip(st, ln):
                lens[int(s)].append(int(n))
        out.append({"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "n_pred": r["n_pred_spells"],
                    "n_gt": r["n_gt_spells"], "lens": lens})
    return test, tuple(train), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="hsmm")
    a = ap.parse_args()
    if a.time:
        halves, ev = evid(TUNE[0])
        model = hsmm.fit([h["truth"] for m in TUNE[1:] for h in evid(m)[0]])
        print("D blocks", model["D"], "block", model["block"])
        for i, e in enumerate(ev):
            t0 = time.time()
            p = hsmm.decode(e, model, 1.0, 1.0)
            p2 = hsmm.decode(e, model, 1.0, 1.0)
            print(f"half {i}: T={len(e)} decode {time.time() - t0:.1f}s (two runs), identical {np.array_equal(p, p2)}")
        return
    tasks = [([k for k in TUNE if k != m], m) for m in TUNE]  # outer
    tasks += [([k for k in TUNE if k not in (m, kk)], kk) for m in TUNE for kk in TUNE if kk != m]  # inner
    uniq = {}
    for t in tasks:
        uniq[(tuple(t[0]), t[1])] = t
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = {(tr, te): o for te, tr, o in pool.imap_unordered(run_task, list(uniq.values()))}
    print(f"all folds done in {time.time() - t0:.0f}s", flush=True)

    held, lens_sel = {}, {0: [], 1: [], 2: []}
    for m in TUNE:
        inner = []
        for kk in TUNE:
            if kk == m:
                continue
            tr = tuple(k for k in TUNE if k not in (m, kk))
            inner.append([r["spell"] for r in res[(tr, kk)]])
        best = int(np.argmax(np.mean(inner, 0)))
        r = res[(tuple(k for k in TUNE if k != m), m)][best]
        held[m] = {"params": CELLS[best], "spell": r["spell"], "macro": r["macro"],
                   "n_pred": r["n_pred"], "n_gt": r["n_gt"]}
        for j in range(3):
            lens_sel[j] += r["lens"][j]
        print(m, json.dumps(CELLS[best]), f"spell {r['spell']:.4f} macro {r['macro']:.4f} spells {r['n_pred']}/{r['n_gt']}")
    spell = np.array([h["spell"] for h in held.values()])
    # fixed-cell table (outer folds only): mean over matches per cell, for transparency (not honest selection)
    table = []
    for ci, c in enumerate(CELLS):
        rs = [res[(tuple(k for k in TUNE if k != m), m)][ci] for m in TUNE]
        table.append({**c, "spell_mean": float(np.mean([r["spell"] for r in rs])),
                      "macro_mean": float(np.mean([r["macro"] for r in rs])),
                      "spells_pred": int(sum(r["n_pred"] for r in rs))})
        print(c, f"spell {table[-1]['spell_mean']:.4f} macro {table[-1]['macro_mean']:.4f} pred {table[-1]['spells_pred']}")
    report = {"name": a.name, "lomo_spell_mean": float(spell.mean()), "lomo_spell_min": float(spell.min()),
              "lomo_spell_max": float(spell.max()),
              "lomo_macro_mean": float(np.mean([h["macro"] for h in held.values()])),
              "held_out": held, "grid": GRID, "cells_outer_fold_table": table,
              "selected_run_lengths_frames": {str(j): lens_sel[j] for j in range(3)}}
    LOG.mkdir(exist_ok=True)
    (LOG / f"{a.name}.json").write_text(json.dumps(report))
    print(f"{a.name}: LOMO spellF1 {report['lomo_spell_mean']:.4f} [{report['lomo_spell_min']:.4f}, "
          f"{report['lomo_spell_max']:.4f}] macroF1 {report['lomo_macro_mean']:.4f}")


if __name__ == "__main__":
    main()
