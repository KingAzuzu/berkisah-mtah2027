"""Part B: restart evidence on top of the reference safety evidence and the WS2 rules evidence, LOMO.

    python -m possession.restarts_run --sigma 0.5 [--seed 0] [--bases safety rules]

Simulated tags (restarts.simulate_tags) are built one match at a time and keyed by half fingerprint;
lomo.evaluate_grid then runs grid-only (PRE, W_D, POST, W_T; the zero-weight cell is the base itself).
Decoded with safety.viterbi at each base's own switch costs. Single process.
"""
import argparse
import gc
import json
from pathlib import Path

from . import restarts as R
from . import rules, safety
from .data import TUNE, idsse
from .lomo import evaluate_grid
from .run_idsse import load

HERE = Path(__file__).parent
BASE_P = {
    "safety": {**safety.PARAMS, **json.loads((HERE / "params.json").read_text())},
    "rules": {**safety.PARAMS, **rules.PARAMS,
              **json.loads((HERE / "logs" / "rules_final.json").read_text())["best_on_all"]},
}
GRID = {"PRE": [75, 200, 400], "W_D": [0.0, 2.0, 5.0], "POST": [25, 75], "W_T": [0.0, 1.0, 3.0]}


def build_tags(sigma, seed, drop):
    tags, stats = {}, {"n_events": 0, "n_tags": 0, "throw_tags": 0, "throw_wrong": 0, "throw_none": 0}
    for mid in TUNE:
        m = idsse.load_match(mid)
        halves = load(mid)
        ev = R.events_by_half(mid, m)
        del m
        stats["n_events"] += len(ev)
        for hi, h in enumerate(halves):
            true_team = {(e["frame"], e["type"]): e["team"] for e in ev if e["half"] == hi}
            tg = R.simulate_tags(ev, hi, h, mid, sigma, drop, seed)
            tags[R.fingerprint(h)] = tg
            stats["n_tags"] += len(tg)
            for g in tg:
                if g["type"] == "ThrowIn":
                    stats["throw_tags"] += 1
                    stats["throw_wrong"] += g["team"] is not None and g["team"] != true_team[(g["true_frame"], "ThrowIn")]
                    stats["throw_none"] += g["team"] is None
        del halves
        gc.collect()
    return tags, stats


def make_predict(base, tags):
    p0 = BASE_P[base]

    def predict(half, params, cache):
        if "e" not in cache:
            if base == "safety":
                cache["e"] = safety.emissions(safety.features(half), p0)
            else:
                cache["e"] = rules.evidence(half, p0, cache)
        e = R.adjust(cache["e"], tags[R.fingerprint(half)], params)
        return safety.viterbi(e, p0["SWITCH_TEAM"], p0["SWITCH_DEAD"])
    return predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--drop", type=float, default=0.15)
    ap.add_argument("--bases", nargs="+", default=["safety", "rules"])
    a = ap.parse_args()
    tags, stats = build_tags(a.sigma, a.seed, a.drop)
    print("tags", json.dumps(stats), flush=True)
    for base in a.bases:
        name = f"restarts_{base}_s{a.sigma}_seed{a.seed}"
        rep = evaluate_grid(make_predict(base, tags), GRID, name)
        rep["tag_stats"] = stats
        (HERE / "logs" / f"{name}.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
