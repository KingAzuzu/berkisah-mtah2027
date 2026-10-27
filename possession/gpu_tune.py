"""Optional helper: evaluate a grid of decoder params on GPU batches, score each cell with score.score.

    python -m possession.gpu_tune --reference          # reproduce logs/reference_safety.json (LOMO 0.3017)
    python -m possession.gpu_tune --grid-bench         # same grid, GPU vs CPU path, wall time

evaluate_grid_gpu mirrors lomo.evaluate_grid (same per-cell table, same leave-one-match-out selection,
same report) but decodes all cells of a half in one GPU batch. Tuning only: the final labels stay on CPU.

evidence_fn(half, params, cache) -> (T, 3) evidence; decoder params read from each cell:
SWITCH_TEAM, SWITCH_DEAD, optional TEMP (default 1). decoder = "viterbi" or "argmax" (posterior argmax).
Evidence that does not depend on a cell's params can be cached via `cache` exactly as in lomo.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np

from .data import TUNE
from .run_idsse import load
from .score import score

LOG_DIR = Path(__file__).with_name("logs")


def decode_cells(evid, cells, decoder="viterbi", batch=64, dtype=None):
    """evid: list of (T,3), one per cell -> list of (T,) int8 codes. GPU, batches of `batch` cells."""
    import torch
    from . import gpu_decode as gd
    dtype = dtype or torch.float64
    out = []
    for a in range(0, len(cells), batch):
        cs, es = cells[a:a + batch], np.stack(evid[a:a + batch])
        st = [c["SWITCH_TEAM"] for c in cs]
        sd = [c["SWITCH_DEAD"] for c in cs]
        if decoder == "viterbi":
            p = gd.viterbi(es, st, sd, dtype=dtype)
        else:
            p = gd.posterior_argmax(es, st, sd, [c.get("TEMP", 1.0) for c in cs], dtype=dtype)
        out.extend(list(p))
    return out


def evaluate_grid_gpu(evidence_fn, grid, name, matches=TUNE, decoder="viterbi", batch=64, verbose=True,
                      dtype=None, write=True):
    data = {m: load(m) for m in matches}
    caches = {m: [{} for _ in data[m]] for m in matches}
    keys = list(grid)
    cells = [dict(zip(keys, v)) for v in itertools.product(*grid.values())]
    t0 = time.perf_counter()
    preds = {m: [None] * len(cells) for m in matches}  # preds[m][cell] = list of per-half codes
    for m in matches:
        per_half = []
        for h, c in zip(data[m], caches[m]):
            evid = [evidence_fn(h, p, c) for p in cells]
            per_half.append(decode_cells(evid, cells, decoder, batch, dtype))
        for i in range(len(cells)):
            preds[m][i] = [ph[i] for ph in per_half]
    t_decode = time.perf_counter() - t0
    table = []
    for i, p in enumerate(cells):
        res = {}
        for m in matches:
            r = score(data[m], preds[m][i])
            res[m] = {"spell": r["spell_f1_mean"], "macro": r["macro_f1"],
                      "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
        table.append((p, res))
        if verbose:
            print(json.dumps(p), " ".join(f"{res[m]['spell']:.4f}" for m in matches),
                  f"mean {np.mean([v['spell'] for v in res.values()]):.4f}", flush=True)
    report = lomo_report(table, matches, grid, name)
    report["wall_s"] = time.perf_counter() - t0
    report["evidence_plus_gpu_decode_s"] = t_decode
    if write:
        LOG_DIR.mkdir(exist_ok=True)
        (LOG_DIR / f"{name}.json").write_text(json.dumps(report, indent=1))
    print(f"{name}: LOMO spellF1 {report['lomo_spell_mean']:.4f} "
          f"[{report['lomo_spell_min']:.4f}, {report['lomo_spell_max']:.4f}]  "
          f"macroF1 {report['lomo_macro_mean']:.4f}  spells pred/gt {report['lomo_spells_pred_vs_gt']}  "
          f"wall {report['wall_s']:.1f}s", flush=True)
    return report


def lomo_report(table, matches, grid, name):
    """Same selection and fields as lomo.evaluate_grid."""
    held = {}
    for m in matches:
        best = max(table, key=lambda row: np.mean([v["spell"] for k, v in row[1].items() if k != m]))
        held[m] = {"params": best[0], **best[1][m]}
    spell = np.array([h["spell"] for h in held.values()])
    best_all = max(table, key=lambda row: np.mean([v["spell"] for v in row[1].values()]))
    return {"name": name,
            "lomo_spell_mean": float(spell.mean()), "lomo_spell_min": float(spell.min()),
            "lomo_spell_max": float(spell.max()),
            "lomo_macro_mean": float(np.mean([h["macro"] for h in held.values()])),
            "lomo_spells_pred_vs_gt": [[h["n_pred"], h["n_gt"]] for h in held.values()],
            "held_out": held, "best_on_all": best_all[0], "grid": grid}


def safety_evidence(half, p, cache):
    from . import safety
    if "f" not in cache:
        cache["f"] = safety.features(half)
    return safety.emissions(cache["f"], {**safety.PARAMS, **p})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true")
    ap.add_argument("--grid-bench", action="store_true")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    params = json.loads(Path(__file__).with_name("params.json").read_text())
    if a.reference:
        t0 = time.perf_counter()
        r = evaluate_grid_gpu(safety_evidence, {k: [v] for k, v in params.items()}, "gpu_reference_safety")
        ref = json.loads((LOG_DIR / "reference_safety.json").read_text())
        r["reference_lomo_spell_mean"] = ref["lomo_spell_mean"]
        r["abs_diff_vs_reference"] = abs(r["lomo_spell_mean"] - ref["lomo_spell_mean"])
        r["per_match_equal"] = all(r["held_out"][m]["spell"] == ref["held_out"][m]["spell"] for m in ref["held_out"])
        r["note"] = a.note
        (LOG_DIR / "gpu_reference_safety.json").write_text(json.dumps(r, indent=1))
        print("diff vs reference", r["abs_diff_vs_reference"], "per-match identical", r["per_match_equal"])
    if a.grid_bench:
        grid = {"SWITCH_TEAM": [5.0, 10.0, 20.0, 40.0], "SWITCH_DEAD": [10.0, 20.0, 40.0, 80.0],
                "SCALE": [1.0, 2.0], "V0": [1.0, 1.5, 2.0]}  # 96 cells
        g = evaluate_grid_gpu(safety_evidence, grid, "gpu_grid_gpu", verbose=False)
        from .lomo import evaluate_grid

        def cpu_predict(half, p, cache):
            from . import safety
            if "f" not in cache:
                cache["f"] = safety.features(half)
            return safety.predict(half, p, cache["f"])
        t0 = time.perf_counter()
        c = evaluate_grid(cpu_predict, grid, "gpu_grid_cpu", verbose=False)
        t_cpu = time.perf_counter() - t0
        out = {"cells": 96, "gpu_wall_s": g["wall_s"], "cpu_wall_s": t_cpu,
               "gpu_lomo_spell": g["lomo_spell_mean"], "cpu_lomo_spell": c["lomo_spell_mean"],
               "same_held_out": g["held_out"] == c["held_out"], "note": a.note}
        (LOG_DIR / "gpu_grid_bench.json").write_text(json.dumps(out, indent=1))
        print(json.dumps(out))


if __name__ == "__main__":
    main()
