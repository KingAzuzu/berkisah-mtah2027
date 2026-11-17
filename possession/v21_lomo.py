"""v21 LOMO harness on IDSSE TUNE (CPU; GPU used for training when present). Caches under idsse-tests/data/cache/v21 (public data).

  python -m possession.v21_lomo data                       # degrade + features + prep (matched2 / matched3, canonical + 6 DR each);
                                                           # V21_JOBS parallel matches (default 2, ~2-3 GB RAM each)
  python -m possession.v21_lomo nets RECIPE SEED [THREADS] # LOMO nets for one recipe and seed -> logs/v21/nets/*.npz
  python -m possession.v21_lomo dead                       # LOMO dead models (ball / players) trained on m2 or m3 canonical
  python -m possession.v21_lomo eval [CAND ...]            # decode, score, gate -> logs/v21/eval.json

Recipes (labels = DFL truth; v6 recipe otherwise: TCN + boundary head, EMA, mirror/swap aug, mirror TTA, short_w 3, 1500 steps):
  r2   : matched2 canonical + 6 DR (v4_prep ranges), f26, DIL_STD          = the v6 recipe (reference)
  r3   : matched3 canonical + 6 DR (degrade_m3 ranges), f26, DIL_STD
  r3x  : r3 with f32 features
  r3xl : r3x with DIL_LONG (+-27.6 s), crop 600
Evaluation set: the held-out match's matched3 canonical degrade (primary) and matched2 canonical (reported).
Decoder = v10's: e = 0.5 * mean log-prob + 0.5 * pipeline.combine(rules team evidence, p_dead, 0.5); Viterbi 20 / 5 with the
switch cost scaled by (1 - 0.6 * boundary prob)."""
import functools
import json
import os
import sys
import time

import joblib
import numpy as np
from joblib import Parallel, delayed

from . import dead, degrade_m3 as M3, degrade_matched2 as DM2, pipeline as pl, tcn21_lib as L, tcn_lib as T, v4_lib as V, v4_prep
from .data import TUNE, idsse, load_idsse, pm
from .score import score

CACHE = idsse.DATA / "cache" / "v21"
LOG = pl.HERE / "logs" / "v21"
K = 6
RECIPES = {
    "r2": {"sim": "m2", "fset": "f26", "dil": L.DIL_STD, "crop": 300},
    "r3": {"sim": "m3", "fset": "f26", "dil": L.DIL_STD, "crop": 300},
    "r3x": {"sim": "m3", "fset": "f32", "dil": L.DIL_STD, "crop": 300},
    "r3xl": {"sim": "m3", "fset": "f32", "dil": L.DIL_LONG, "crop": 600},
}


def prog(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _variants(m):
    """[(sim, k, halves)] for one match: m2 canonical + 6 DR, m3 canonical + 6 DR."""
    clean = load_idsse(m)
    out = [("m2", 0, DM2.degrade_match(clean, m, 0))]
    out += [("m2", k + 1, v4_prep.degrade_variant(clean, m, k)) for k in range(K)]
    out += [("m3", 0, M3.degrade_match(clean, m, 0))]
    out += [("m3", k, M3.degrade_match(clean, m, 1000 + k, M3.variant(k))) for k in range(1, K + 1)]
    return out


def _prep_half(half):
    cache = {}
    pl.team_evidence(half, pl.base_params(), cache)
    X, names = dead.build_features(half)
    return {"team": cache["team"], "X": X, "names": names, "y": dead.labels(half), "truth": half["truth"],
            "seen": ~np.isnan(half["ball"][:, 0])}


def _build(m):
    feats, evals = {}, {}
    for sim, k, halves in _variants(m):
        feats[(sim, k)] = [L.features(h, "f32") for h in halves]
        if k == 0:
            evals[sim] = [_prep_half(h) for h in halves]
    return m, feats, evals


def build_data():
    CACHE.mkdir(parents=True, exist_ok=True)
    todo = [m for m in TUNE if not (CACHE / f"{m}.pkl").exists()]
    jobs = int(os.environ.get("V21_JOBS", "2"))   # ~2-3 GB RAM per job
    for m, feats, evals in Parallel(n_jobs=min(jobs, len(todo) or 1))(delayed(_build)(m) for m in todo):
        joblib.dump({"feats": feats, "evals": evals}, CACHE / f"{m}.pkl")
        prog(f"data {m} done")


@functools.lru_cache(maxsize=None)
def load_data(m):
    return joblib.load(CACHE / f"{m}.pkl")


def labels(m, i):
    return T.labels5({"truth": load_data(m)["evals"]["m3"][i]["truth"]})


# --- nets ---------------------------------------------------------------------------------------------

def train_nets(recipe, seed, threads=4):
    r = RECIPES[recipe]
    nin = 26 if r["fset"] == "f26" else 32
    D = {m: load_data(m) for m in TUNE}
    out_dir = LOG / "nets"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in TUNE:
        path = out_dir / f"{recipe}_s{seed}_{m}.npz"
        if path.exists():
            continue
        items = []
        for k in TUNE:
            if k == m:
                continue
            for i in range(2):
                y = T.labels5({"truth": D[k]["evals"]["m3"][i]["truth"]})
                fs = [D[k]["feats"][(r["sim"], v)][i][:len(y), :nin] for v in range(K + 1)]
                items.append({"feats": fs, "y": y})
        t0 = time.time()
        mod = L.train21(items, seed=seed, crop=r["crop"], dil=r["dil"], nin=nin, threads=threads)
        res = {}
        for sim in ("m2", "m3"):
            for i in range(2):
                lp, b = L.predict21(mod, D[m]["feats"][(sim, 0)][i][:, :nin])
                res[f"{sim}_lp{i}"], res[f"{sim}_b{i}"] = lp, b
        np.savez_compressed(path, **res)
        prog(f"nets {recipe} s{seed} {m} {time.time() - t0:.0f}s")


def net_outputs(recipe, sim, seeds):
    """{m: [(lp (T,3), b (T,)) per half]} mean over seeds, upsampled to 25 fps."""
    out = {}
    for m in TUNE:
        fs = [np.load(LOG / "nets" / f"{recipe}_s{s}_{m}.npz") for s in seeds]
        E = load_data(m)["evals"][sim]
        out[m] = [(V.upsample(np.mean([f[f"{sim}_lp{i}"] for f in fs], 0), len(E[i]["truth"])),
                   V.upsample(np.mean([f[f"{sim}_b{i}"] for f in fs], 0)[:, None], len(E[i]["truth"]))[:, 0]) for i in range(2)]
    return out


# --- dead models --------------------------------------------------------------------------------------

def dead_tables():
    """p_dead[(train_sim, variant)][eval_sim][m] = per-half p(dead) from a model trained on the other 4 matches."""
    path = CACHE / "pdead.pkl"
    if path.exists():
        return joblib.load(path)
    D = {m: load_data(m)["evals"] for m in TUNE}
    names = D[TUNE[0]]["m3"][0]["names"]
    out = {}
    for tsim in ("m2", "m3"):
        for variant in ("ball", "players"):
            keep = pl.keep_columns(names, variant)
            tab = {"m2": {}, "m3": {}}
            for m in TUNE:
                Xs = [h["X"][::5][:, keep] for k in TUNE if k != m for h in D[k][tsim]]
                ys = [h["y"][::5] for k in TUNE if k != m for h in D[k][tsim]]
                mod = dead.make_model().fit(np.concatenate(Xs), np.concatenate(ys))
                for esim in ("m2", "m3"):
                    tab[esim][m] = [mod.predict_proba(h["X"][:, keep])[:, 1] for h in D[m][esim]]
            out[(tsim, variant)] = tab
            prog(f"dead {tsim} {variant} done")
    joblib.dump(out, path)
    return out


# --- evaluation ---------------------------------------------------------------------------------------

def decode(nets, pdead, esim, fusion=0.5, st=20.0, sd=5.0, hook=0.6, alpha=0.5):
    out = {}
    for m in TUNE:
        E = load_data(m)["evals"][esim]
        hv = []
        for i in range(2):
            eR = pl.combine(E[i]["team"], pdead[m][i], alpha)
            if nets is None:
                hv.append(V.viterbi_scaled(eR, st, sd, np.ones(len(eR))))
                continue
            lp, b = nets[m][i]
            hv.append(V.viterbi_scaled(fusion * lp + (1 - fusion) * eR, st, sd, 1 - hook * b))
        out[m] = hv
    return out


def metrics(pred, esim):
    per, macro, n = [], [], []
    tr, pr, sv = [], [], []
    for m in TUNE:
        E = load_data(m)["evals"][esim]
        r = score([{"truth": h["truth"]} for h in E], pred[m])
        per.append(float(r["spell_f1_mean"])); macro.append(float(r["macro_f1"])); n.append(int(r["n_pred_spells"]))
        tr += [h["truth"] for h in E]; pr += list(pred[m]); sv += [h["seen"] for h in E]
    t, p, s = map(np.concatenate, (tr, pr, sv))
    dd = t == pm.D
    return {"spell": float(np.mean(per)), "per_match": per, "macro": float(np.mean(macro)), "n_pred": n,
            "D_recall_seen": float(np.mean(p[dd & s] == pm.D)), "D_recall_missing": float(np.mean(p[dd & ~s] == pm.D)),
            "D_precision": float(np.mean(t[p == pm.D] == pm.D))}


def show(name, r):
    prog(f"{name:28s} spell {r['spell']:.4f} [{min(r['per_match']):.3f},{max(r['per_match']):.3f}] macro {r['macro']:.4f} "
         f"Drec seen {r['D_recall_seen']:.3f} miss {r['D_recall_missing']:.3f} Dprec {r['D_precision']:.3f} n {sum(r['n_pred'])}")


def gate(cand, best):
    """Plan rule: mean spellF1 >= best + 0.01 and >= 4/5 match wins (matched3 evaluation)."""
    wins = sum(a > b for a, b in zip(cand["per_match"], best["per_match"]))
    return {"delta": cand["spell"] - best["spell"], "wins": wins, "pass": cand["spell"] >= best["spell"] + 0.01 and wins >= 4}


def run_eval(specs, seeds=(20, 21, 22)):
    """specs: 'name=recipe|none:tsim:variant'. Prints matched3 and matched2 metrics; returns {name: {m3, m2}}."""
    pdead = dead_tables()
    res = {}
    for spec in specs:
        name, rest = spec.split("=")
        recipe, tsim, variant = rest.split(":")
        res[name] = {}
        for esim in ("m3", "m2"):
            nets = None if recipe == "none" else net_outputs(recipe, esim, seeds)
            res[name][esim] = metrics(decode(nets, pdead[(tsim, variant)][esim], esim), esim)
            show(f"{name} [{esim}]", res[name][esim])
    return res


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "data":
        build_data()
    elif cmd == "nets":
        train_nets(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]) if len(sys.argv) > 4 else 4)
    elif cmd == "dead":
        dead_tables()
    elif cmd == "eval":
        res = run_eval(sys.argv[3:])
        names = list(res)
        for a in names[1:]:
            g = gate(res[a]["m3"], res[names[0]]["m3"])
            prog(f"gate {a} vs {names[0]}: delta {g['delta']:+.4f} wins {g['wins']}/5 -> {'PASS' if g['pass'] else 'FAIL'}")
        LOG.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(LOG / f"eval_{sys.argv[2]}.json", "w"), indent=1)
