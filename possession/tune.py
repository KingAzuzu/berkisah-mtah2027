"""Grid search for the safety net on the 5 tuning matches, reported leave-one-match-out.

For each match, the parameters are chosen on the other four and scored on it, so the reported mean
is an honest estimate rather than the best cell of the grid. Locked matches are never loaded.

    python -m possession.tune
"""
import itertools
import json

import numpy as np

from . import safety
from .data import TUNE
from .run_idsse import load
from .score import score

# Round 1 (tune1.log): SWITCH_TEAM 5/10/20, SWITCH_DEAD 5/10/20, SCALE 1/3, V0 1.5/2/2.5 -> LOMO 0.2951,
# best V0 and SWITCH_DEAD on the grid edge. Round 2 extends past that edge.
GRID = {
    "SWITCH_TEAM": [10.0, 20.0],
    "SWITCH_DEAD": [20.0, 40.0, 80.0],
    "SCALE": [0.5, 1.0],
    "V0": [0.75, 1.0, 1.25, 1.5],
}


def main():
    data = {m: load(m) for m in TUNE}
    feats = {m: [safety.features(h) for h in data[m]] for m in TUNE}
    keys = list(GRID)
    table = []  # (params, {mid: spellF1})
    for values in itertools.product(*GRID.values()):
        p = dict(zip(keys, values))
        res = {m: score(data[m], [safety.predict(h, p, f) for h, f in zip(data[m], feats[m])])["spell_f1_mean"]
               for m in TUNE}
        table.append((p, res))
        print(json.dumps(p), " ".join(f"{res[m]:.4f}" for m in TUNE), f"mean {np.mean(list(res.values())):.4f}",
              flush=True)

    print("\nleave-one-match-out")
    held = []
    for m in TUNE:
        best = max(table, key=lambda row: np.mean([v for k, v in row[1].items() if k != m]))
        held.append(best[1][m])
        print(f"  {m}  held-out spellF1 {best[1][m]:.4f}  params {json.dumps(best[0])}")
    print(f"LOMO spellF1 mean {np.mean(held):.4f} [{min(held):.4f}, {max(held):.4f}]")
    best = max(table, key=lambda row: np.mean(list(row[1].values())))
    print(f"best on all 5: {json.dumps(best[0])}  mean {np.mean(list(best[1].values())):.4f}")


if __name__ == "__main__":
    main()
