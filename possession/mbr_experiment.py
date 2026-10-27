"""WS3 experiment: MBR decoding on degraded IDSSE tuning matches, LOMO via lomo.evaluate_grid.

    python -m possession.mbr_experiment [--stage 1|2] [--name mbr_x]

Grid-only params: TEMP (emission temperature), S (samples), ST/SD (HMM switch costs for sampling and the
base Viterbi), MULTS (candidate Viterbi cost multipliers), METHOD (viterbi | argmax | mbr), KEEP
(posterior keep-threshold). Locked matches are never loaded (lomo.evaluate_grid uses data.TUNE).
"""
import argparse
import json
import time
from pathlib import Path

from . import mbr, safety
from .lomo import evaluate_grid

BASE = {**safety.PARAMS, **json.loads(Path(__file__).with_name("params.json").read_text())}
MULT_SETS = {"one": (1.0,), "wide": (0.5, 0.75, 1.0, 1.5, 2.0), "low": (0.25, 0.5, 0.75, 1.0)}
KEEPS = (0.0, 0.2, 0.35, 0.5)
TIMES = []


def predict(half, p, cache):
    if "e" not in cache:
        cache["e"] = safety.emissions(safety.features(half), BASE)
    key = (p["TEMP"], p["S"], p["ST"], p["SD"], p["MULT"])
    if key not in cache:
        t0 = time.time()
        cache[key] = mbr.decode_variants(cache["e"], p["ST"], p["SD"], p["TEMP"], p["S"],
                                         MULT_SETS[p["MULT"]], KEEPS)
        TIMES.append(time.time() - t0)
    return cache[key][(p["METHOD"], p["KEEP"])]


GRIDS = {
    1: {"TEMP": [0.5, 1.0, 2.0, 3.0], "S": [50], "ST": [10.0], "SD": [20.0], "MULT": ["one", "wide", "low"],
        "METHOD": ["viterbi", "argmax", "mbr"], "KEEP": [0.0]},
    2: {"TEMP": [0.5, 1.0, 2.0, 3.0], "S": [50], "ST": [5.0, 10.0], "SD": [10.0, 20.0], "MULT": ["wide"],
        "METHOD": ["mbr"], "KEEP": list(KEEPS)},
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--name", default=None)
    ap.add_argument("--grid", default=None, help="JSON grid overriding the stage grid")
    a = ap.parse_args()
    grid = json.loads(a.grid) if a.grid else GRIDS[a.stage]
    t0 = time.time()
    rep = evaluate_grid(predict, grid, a.name or f"mbr_stage{a.stage}", verbose=True)
    print("decode_variants calls %d, mean %.2fs, total wall %.0fs" % (len(TIMES), sum(TIMES) / max(1, len(TIMES)),
                                                                      time.time() - t0))
    for m, h in rep["held_out"].items():
        print(m, json.dumps(h["params"]), "spell %.4f macro %.4f spells %d/%d" % (h["spell"], h["macro"], h["n_pred"], h["n_gt"]))
