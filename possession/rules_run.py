"""WS2 experiments: LOMO grid over rule thresholds and switch costs, via lomo.evaluate_grid.

    python -m possession.rules_run --stage 1 [--workers 4]

Grid cells are precomputed in a process pool (<= 4) into a memo that predict() reads, so the numbers
are exactly what lomo.evaluate_grid computes; nothing is fitted on truth (grid values only).
"""
import argparse
import itertools
import json
import multiprocessing as mp
import zlib
from functools import lru_cache
from pathlib import Path

from . import rules
from .data import TUNE
from .lomo import evaluate_grid
from .run_idsse import load

BASE = json.loads(Path(__file__).with_name("params.json").read_text())
MEMO = {}
STAGES = {
    # gain radius, kick threshold, hold strengths
    1: {"R": [1.5, 3.0], "BV": [2.0, 4.0], "W_HOLD": [0.3, 1.0], "W_DIFF": [0.0, 0.3], "SWITCH_TEAM": [10.0]},
    # finer: radius, kick threshold, flight strength, team switch cost
    2: {"R": [1.0, 1.5, 2.0], "BV": [1.0, 2.0], "W_HOLD": [0.3], "W_DIFF": [0.3, 0.6, 1.0],
        "SWITCH_TEAM": [5.0, 10.0, 20.0]},
    # edge of stage 2 was R=1.0, BV=1.0: go lower; kick window and "ball nearly still" speed
    3: {"R": [0.75, 1.0], "BV": [0.5, 1.0], "KW": [10, 20], "VSLOW": [1.5, 3.0], "W_HOLD": [0.3],
        "W_DIFF": [0.6], "SWITCH_TEAM": [10.0]},
    # remaining structure: unknown-signature policy, gain strength, ball-gap interpolation, duel radius
    # ablation around the chosen setting: kick validation off (VSLOW huge = every frame valid), hold evidence off
    5: {"R": [1.0], "BV": [0.5], "KW": [20], "VSLOW": [1.5, 1e9], "W_HOLD": [0.0, 0.3], "W_DIFF": [0.0, 0.6],
        "SWITCH_TEAM": [10.0]},
    # final modest grid around the chosen setting, the number reported in results/ws2_rules.md
    6: {"R": [1.0], "BV": [0.5, 1.0], "KW": [10, 20], "W_HOLD": [0.3], "W_DIFF": [0.3, 0.6, 1.0],
        "SWITCH_TEAM": [5.0, 10.0, 20.0]},
    4: {"R": [1.0], "BV": [0.5], "KW": [20], "ALLOW_UNK": [0, 1], "W_VAL": [1.5, 3.0, 6.0], "GAP": [0, 25, 75],
        "DUEL_R": [1.0], "W_HOLD": [0.3], "W_DIFF": [0.6], "SWITCH_TEAM": [10.0]},
}


def _fp(half):
    return zlib.crc32(half["ball"].tobytes()) ^ zlib.crc32(half["xy"][:, :, 0].tobytes())


def _key(fp, p):
    return fp, json.dumps(p, sort_keys=True)


@lru_cache(maxsize=None)
def _w_half(mid, i):
    return load(mid)[i]


_W_CACHE = {}


def _work(task):
    mid, i, p = task
    h = _w_half(mid, i)
    c = _W_CACHE.setdefault((mid, i), {})
    return (mid, i, json.dumps(p, sort_keys=True)), rules.predict(h, p, c)


def predict(half, p, cache):
    k = _key(_fp(half), p)
    if k not in MEMO:
        MEMO[k] = rules.predict(half, p, cache)
    return MEMO[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--grid", type=str, default=None, help="json overriding the stage grid")
    ap.add_argument("--name", type=str, default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    grid = json.loads(a.grid) if a.grid else STAGES[a.stage]
    grid = {**{k: [v] for k, v in BASE.items() if k not in grid}, **grid}
    name = a.name or f"rules_s{a.stage}"
    cells = [dict(zip(grid, v)) for v in itertools.product(*grid.values())]
    data = {m: load(m) for m in TUNE}
    fps = {(m, i): _fp(h) for m in TUNE for i, h in enumerate(data[m])}
    tasks = [(m, i, c) for c in cells for m in TUNE for i in range(len(data[m]))]
    workers = min(a.workers, 4)
    with mp.Pool(workers) as pool:
        for (m, i, ps), path in pool.imap(_work, tasks, chunksize=2):
            MEMO[(fps[(m, i)], ps)] = path
    evaluate_grid(predict, grid, name=name)


if __name__ == "__main__":
    mp.freeze_support()
    main()
