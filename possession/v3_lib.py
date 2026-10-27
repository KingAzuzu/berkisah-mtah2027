"""Shared pieces for the v3 candidate experiments (IDSSE tuning matches only, never the locked ones).

Protocol (same as integrate_run): for held-out match m, the v2 cell comes from the logged v2 LOMO
(`logs/integrate_s1_players_seed0.json`, chosen on the other 4 matches), p_dead on m from a model trained on the
other 4 (`pdead_tables`, cached), and any candidate parameter is chosen on the inner folds (matches k != m, p_dead
from a model that saw neither m nor k) using m's v2 cell. Score only through possession.score.
"""
import json
import os

import joblib
import numpy as np

from . import pipeline as pl
from .data import TUNE, idsse, pm
from .lomo import LOG_DIR
from .run_idsse import load
from .score import score

V2_LOG = LOG_DIR / "integrate_s1_players_seed0.json"
CACHE = idsse.DATA / "cache" / "integrate"
os.environ.setdefault("OMP_NUM_THREADS", "3")


def v2_cells():
    d = json.loads(V2_LOG.read_text())["held_out"]
    return {m: d[m]["cell"] for m in TUNE}


def team_of(half):
    cache = {}
    pl.team_evidence(half, pl.base_params(), cache)
    return cache["team"]


def pdead_cached(tag, variant="players"):
    return joblib.load(CACHE / f"pdead_{tag}_{variant}.pkl")


def decode_evidence(e, cell):
    return pl.decode(e, cell)


def spell_score(halves, preds):
    r = score(halves, preds)
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}


def summary(per_match):
    """per_match {m: {spell, macro, n_pred, n_gt}} -> mean/min/max etc. in TUNE order."""
    s = np.array([per_match[m]["spell"] for m in TUNE])
    return {"spell_mean": float(s.mean()), "spell_min": float(s.min()), "spell_max": float(s.max()),
            "macro_mean": float(np.mean([per_match[m]["macro"] for m in TUNE])),
            "spell": [float(per_match[m]["spell"]) for m in TUNE], "macro": [float(per_match[m]["macro"]) for m in TUNE],
            "n_pred": [int(per_match[m]["n_pred"]) for m in TUNE], "n_gt": [int(per_match[m]["n_gt"]) for m in TUNE]}


def fmt_row(name, s):
    return (f"| {name} | {s['spell_mean']:.4f} | {s['spell_min']:.4f} | {s['spell_max']:.4f} | "
            + " ".join(f"{x:.3f}" for x in s["spell"]) + f" | {s['macro_mean']:.4f} | "
            + " ".join(f"{x:.3f}" for x in s["macro"]) + " | "
            + " ".join(str(x) for x in s["n_pred"]) + " |")


# --- run helpers ------------------------------------------------------------------------------

def runs(path):
    """[(start, stop, code)] maximal runs."""
    path = np.asarray(path)
    if len(path) == 0:
        return []
    ch = np.flatnonzero(np.diff(path)) + 1
    st = np.r_[0, ch]
    en = np.r_[ch, len(path)]
    return [(int(a), int(b), int(path[a])) for a, b in zip(st, en)]


def from_runs(rs, T):
    out = np.empty(T, np.int8)
    for a, b, c in rs:
        out[a:b] = c
    return out


def merged(rs):
    out = []
    for a, b, c in rs:
        if out and out[-1][2] == c:
            out[-1] = (out[-1][0], b, c)
        else:
            out.append((a, b, c))
    return out
