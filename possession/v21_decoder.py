"""v21 step 4: decoder costs chosen by nested LOMO on held-out outputs (IDSSE TUNE, matched3 evaluation, CPU).

4-state Viterbi: H, A, D-after-H, D-after-A (D-after-X is entered only from X, the two D states do not connect). Costs, all
multiplied by (1 - hook * boundary prob) as in v10: st (H<->A), sd_in (alive -> D), sd_out (D -> alive) and an extra `same` when
play restarts with the team that had the ball before the stoppage (72% of IDSSE restarts go to the other team). st=20,
sd_in=sd_out=5, same=0 is exactly v10's 3-state decoder.
Nested selection: for each held-out match, the grid point with the best mean spellF1 on the other 4 matches is applied to it.

python -m possession.v21_decoder NAME RECIPE|none DEAD_TSIM DEAD_VARIANT [SEEDS] -> logs/v21/decoder_NAME.json"""
import itertools
import json
import sys

import numpy as np

from . import pipeline as pl, v21_lomo as H
from .data import TUNE, pm
from .score import join

GRID = {"st": (10.0, 15.0, 20.0, 30.0, 45.0), "sd_in": (2.0, 5.0, 10.0), "sd_out": (2.0, 5.0, 10.0, 20.0), "same": (0.0, 2.0, 5.0)}
BIG = 1e9


def cost_tensor(cfgs):
    """cfgs (C, 4) [st, sd_in, sd_out, same] -> (C, 4, 4) transition costs, states H 0, A 1, DH 2, DA 3."""
    st, si, so, sm = cfgs.T
    C = len(cfgs)
    c = np.full((C, 4, 4), BIG)
    for i in range(4):
        c[:, i, i] = 0.0
    c[:, 0, 1] = c[:, 1, 0] = st
    c[:, 0, 2] = c[:, 1, 3] = si
    c[:, 2, 1] = c[:, 3, 0] = so
    c[:, 2, 0] = c[:, 3, 1] = so + sm
    return c


def viterbi_batch(e, cfgs, scale):
    """e (T, 3) evidence, cfgs (C, 4), scale (T,) -> (C, T) int8 codes H 0 / A 1 / D 2, one path per config."""
    T, C = len(e), len(cfgs)
    base = cost_tensor(np.asarray(cfgs, float))
    em = e[:, [0, 1, 2, 2]]
    score = np.tile(em[0], (C, 1))
    back = np.empty((T, C, 4), np.int8)
    for t in range(1, T):
        cand = score[:, :, None] - base * scale[t]
        arg = cand.argmax(1)
        back[t] = arg
        score = np.take_along_axis(cand, arg[:, None, :], 1)[:, 0, :] + em[t]
    path = np.empty((C, T), np.int8)
    path[:, -1] = score.argmax(1)
    rows = np.arange(C)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = back[t, rows, path[:, t]]
    return np.minimum(path, 2).astype(np.int8)


def spell(truths, preds):
    return float(pm.evaluate(join(truths), join(preds), bootstrap_n=0)["spell_f1_mean"])


def run(recipe, tsim, variant, seeds=(20, 21, 22), esim="m3", fusion=0.5, hook=0.6, alpha=0.5):
    cfgs = np.array(list(itertools.product(*GRID.values())))
    pdead = H.dead_tables()[(tsim, variant)][esim]
    nets = None if recipe == "none" else H.net_outputs(recipe, esim, seeds)
    S = np.zeros((len(cfgs), len(TUNE)))
    for j, m in enumerate(TUNE):
        E = H.load_data(m)["evals"][esim]
        paths = []
        for i in range(2):
            eR = pl.combine(E[i]["team"], pdead[m][i], alpha)
            if nets is None:
                e, sc = eR, np.ones(len(eR))
            else:
                lp, b = nets[m][i]
                e, sc = fusion * lp + (1 - fusion) * eR, 1 - hook * b
            paths.append(viterbi_batch(e, cfgs, sc))
        truths = [h["truth"] for h in E]
        S[:, j] = [spell(truths, [paths[0][c], paths[1][c]]) for c in range(len(cfgs))]
        H.prog(f"decoder grid {m} done (best in-sample {S[:, j].max():.4f})")
    ref = int(np.flatnonzero((cfgs == [20.0, 5.0, 5.0, 0.0]).all(1))[0])
    nested, picks = [], []
    for j, m in enumerate(TUNE):
        others = [k for k in range(len(TUNE)) if k != j]
        c = int(np.argmax(S[:, others].mean(1)))
        nested.append(float(S[c, j])); picks.append(cfgs[c].tolist())
    return {"ref_cfg": cfgs[ref].tolist(), "ref_per_match": S[ref].tolist(), "ref_mean": float(S[ref].mean()),
            "nested_per_match": nested, "nested_mean": float(np.mean(nested)), "picks": picks,
            "best_overall": cfgs[int(np.argmax(S.mean(1)))].tolist(), "best_overall_mean": float(S.mean(1).max())}


if __name__ == "__main__":
    name, recipe, tsim, variant = sys.argv[1:5]
    seeds = tuple(int(s) for s in sys.argv[5].split(",")) if len(sys.argv) > 5 else (20, 21, 22)
    r = run(recipe, tsim, variant, seeds)
    wins = sum(a > b for a, b in zip(r["nested_per_match"], r["ref_per_match"]))
    r["gate"] = {"delta": r["nested_mean"] - r["ref_mean"], "wins": wins,
                 "pass": r["nested_mean"] >= r["ref_mean"] + 0.01 and wins >= 4}
    H.prog(f"{name}: v10 decoder {r['ref_mean']:.4f} | nested {r['nested_mean']:.4f} (delta {r['gate']['delta']:+.4f}, "
           f"wins {wins}/5) -> {'PASS' if r['gate']['pass'] else 'FAIL'} | picks {r['picks']}")
    json.dump(r, open(H.LOG / f"decoder_{name}.json", "w"), indent=1)
