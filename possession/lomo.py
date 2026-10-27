"""Shared leave-one-match-out evaluation, so every experiment is measured the same way.

    from possession.lomo import evaluate_grid
    report = evaluate_grid(predict, grid, name="hsmm")

predict(half, params, cache) -> per-frame codes (H 0, A 1, D 2). `cache` is a dict private to that
half, reused across grid cells, for anything expensive that does not depend on params (features,
model outputs). For each tuning match the params are chosen on the other four and scored on it, so
the LOMO mean is an honest estimate. Locked matches are never loaded here.

The reference to beat is the current safety net: run `python -m possession.lomo` to print it.
"""
import itertools
import json
from pathlib import Path

import numpy as np

from .data import TUNE
from .run_idsse import load
from .score import score

LOG_DIR = Path(__file__).with_name("logs")


def evaluate_grid(predict, grid, name, matches=TUNE, verbose=True):
    data = {m: load(m) for m in matches}
    caches = {m: [{} for _ in data[m]] for m in matches}
    keys = list(grid)
    table = []
    for values in itertools.product(*grid.values()):
        p = dict(zip(keys, values))
        res = {}
        for m in matches:
            r = score(data[m], [predict(h, p, c) for h, c in zip(data[m], caches[m])])
            res[m] = {"spell": r["spell_f1_mean"], "macro": r["macro_f1"],
                      "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
        table.append((p, res))
        if verbose:
            print(json.dumps(p), " ".join(f"{res[m]['spell']:.4f}" for m in matches),
                  f"mean {np.mean([v['spell'] for v in res.values()]):.4f}", flush=True)

    held = {}
    for m in matches:
        best = max(table, key=lambda row: np.mean([v["spell"] for k, v in row[1].items() if k != m]))
        held[m] = {"params": best[0], **best[1][m]}
    spell = np.array([h["spell"] for h in held.values()])
    best_all = max(table, key=lambda row: np.mean([v["spell"] for v in row[1].values()]))
    report = {
        "name": name,
        "lomo_spell_mean": float(spell.mean()), "lomo_spell_min": float(spell.min()),
        "lomo_spell_max": float(spell.max()),
        "lomo_macro_mean": float(np.mean([h["macro"] for h in held.values()])),
        "lomo_spells_pred_vs_gt": [[h["n_pred"], h["n_gt"]] for h in held.values()],
        "held_out": held, "best_on_all": best_all[0], "grid": grid,
    }
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / f"{name}.json").write_text(json.dumps(report, indent=1))
    print(f"{name}: LOMO spellF1 {report['lomo_spell_mean']:.4f} "
          f"[{report['lomo_spell_min']:.4f}, {report['lomo_spell_max']:.4f}]  "
          f"macroF1 {report['lomo_macro_mean']:.4f}  spells pred/gt {report['lomo_spells_pred_vs_gt']}")
    return report


def _reference_predict(half, p, cache):
    from . import safety
    if "f" not in cache:
        cache["f"] = safety.features(half)
    return safety.predict(half, p, cache["f"])


if __name__ == "__main__":
    params = json.loads(Path(__file__).with_name("params.json").read_text())
    evaluate_grid(_reference_predict, {k: [v] for k, v in params.items()}, name="reference_safety")
