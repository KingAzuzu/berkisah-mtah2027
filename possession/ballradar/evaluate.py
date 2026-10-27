"""Score the cached held-out predictions: ball error, team accuracy, and spellF1 through safety.viterbi.

    python -m possession.ballradar.evaluate

Evidence = safety's dead-ball evidence (D column and its effect on H/A, unchanged) + a team part that is
  ref          nearest-player team evidence only (the reference safety net)
  add          nearest + w * model
  repl_hidden  model * w on frames where the nearest-player team part says nothing, nearest elsewhere
  repl_all     w * model on every frame
model team evidence = log(2 sigmoid(+-z)) with z = log(pH / pA) from the held-out probabilities.
Grid over w, SWITCH_TEAM, SWITCH_DEAD; for each held-out match the cell is chosen on the other four
(as lomo.evaluate_grid does), then scored on it.
"""
import itertools
import json
import pickle
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from .. import safety
from ..data import TUNE, load_idsse, pm
from ..run_idsse import load
from ..score import score
from . import features as fe

ROOT = Path(__file__).resolve().parent.parent
PARAMS = json.loads((ROOT / "params.json").read_text())
WS = (0.5, 1.0, 2.0)
SW_T, SW_D = (5.0, 10.0, 20.0, 30.0), (10.0, 20.0, 30.0)
MODES = ("add", "repl_hidden", "repl_all")


def evidence(half, probs, mode, w, f=None, base=None):
    """(T, 3) log-scores. `base` = (e_dead, team_near) may be passed to avoid recomputation."""
    if base is None:
        f = f or safety.features(half)
        p = {**safety.PARAMS, **PARAMS}
        e_dead = safety.emissions(f, {**p, "R_CLOSE": -1.0})
        base = (e_dead, safety.emissions(f, p) - e_dead)
    e_dead, near = base
    if mode == "ref":
        return e_dead + near
    z = np.clip(np.log(probs[:, 0] + 1e-6) - np.log(probs[:, 1] + 1e-6), -20, 20)
    em = np.zeros_like(near)
    em[:, pm.H] = np.log(2.0) - np.logaddexp(0, -z)
    em[:, pm.A] = np.log(2.0) - np.logaddexp(0, z)
    if mode == "add":
        team = near + w * em
    elif mode == "repl_all":
        team = w * em
    else:
        silent = (near[:, 0] == 0) & (near[:, 1] == 0)
        team = np.where(silent[:, None], w * em, near)
    return e_dead + team


def configs():
    cells = [("ref", 0.0)] + [(m, w) for m in MODES for w in WS]
    return [(m, w, st, sd) for (m, w) in cells for st in SW_T for sd in SW_D]


def match_task(mid):
    halves = load(mid)
    held = pickle.loads((fe.CACHE / f"heldout_{mid}.pkl").read_bytes())
    bases = []
    for h in halves:
        f = safety.features(h)
        p = {**safety.PARAMS, **PARAMS}
        e_dead = safety.emissions(f, {**p, "R_CLOSE": -1.0})
        bases.append((e_dead, safety.emissions(f, p) - e_dead))
    out = {}
    for (mode, w, st, sd) in configs():
        preds = [safety.viterbi(evidence(h, ho["probs"], mode, w, base=b), st, sd)
                 for h, ho, b in zip(halves, held, bases)]
        r = score(halves, preds)
        out[(mode, w, st, sd)] = {"spell": r["spell_f1_mean"], "macro": r["macro_f1"],
                                  "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}
    return mid, out


def ball_and_team(mid):
    deg, clean = load(mid), load_idsse(mid)
    held = pickle.loads((fe.CACHE / f"heldout_{mid}.pkl").read_bytes())
    err_m, err_i, err_alive, acc_alive, acc_hid, acc_seen, acc3, acc_nn, n_hid = [], [], [], [], [], [], [], [], 0
    for d, c, ho in zip(deg, clean, held):
        hidden = np.isnan(d["ball"][:, 0]) & ~np.isnan(c["ball"][:, 0])
        e = np.linalg.norm(ho["ball"] - c["ball"], axis=1)
        err_m.append(e[hidden])
        t = np.arange(len(e))
        seen = ~np.isnan(d["ball"][:, 0])
        li = np.stack([np.interp(t, t[seen], d["ball"][seen, k]) for k in (0, 1)], 1)
        err_i.append(np.linalg.norm(li - c["ball"], axis=1)[hidden])
        alive = d["truth"] < 2
        err_alive.append(e[hidden & alive])
        pred = np.argmax(ho["probs"][:, :2], 1)
        ok = pred == d["truth"]
        acc_alive.append(ok[alive])
        acc_hid.append(ok[alive & hidden])
        acc_seen.append(ok[alive & seen])
        acc3.append(np.argmax(ho["probs"], 1) == d["truth"])
        # nearest-player baseline: team of the nearest player to the CLEAN ball (upper-bound style reference)
        dd = np.linalg.norm(c["xy"] - c["ball"][:, None, :], axis=2)
        dd = np.where(np.isnan(dd), np.inf, dd)
        acc_nn.append((c["team"][np.argmin(dd, 1)] == d["truth"])[alive])
    cat = np.concatenate
    return {"ball_err_hidden_mean": float(cat(err_m).mean()), "ball_err_hidden_median": float(np.median(cat(err_m))),
            "ball_err_hidden_alive_mean": float(cat(err_alive).mean()),
            "ball_err_linear_interp_hidden_mean": float(cat(err_i).mean()),
            "n_hidden_frames": int(len(cat(err_m))),
            "team_acc_alive": float(cat(acc_alive).mean()), "team_acc_alive_hiddenball": float(cat(acc_hid).mean()),
            "team_acc_alive_seenball": float(cat(acc_seen).mean()),
            "acc3_all_frames": float(cat(acc3).mean()),
            "nearest_player_to_clean_ball_acc_alive": float(cat(acc_nn).mean())}


def single(mids):
    """Partial run (some folds only): ball/team numbers and the spellF1 table per match, no LOMO selection."""
    for m in mids:
        print(m, json.dumps(ball_and_team(m)), flush=True)
    with Pool(min(3, len(mids))) as pool:
        tables = dict(pool.map(match_task, mids))
    for m in mids:
        t = tables[m]
        print(f"{m} reference-switch cells (SWITCH_TEAM 10, SWITCH_DEAD 20):")
        for c in configs():
            if c[2] == 10.0 and c[3] == 20.0:
                print(f"   {c[0]:12s} w={c[1]:<4} spell {t[c]['spell']:.4f} macro {t[c]['macro']:.4f} "
                      f"pred/gt {t[c]['n_pred']}/{t[c]['n_gt']}")
        for mode in ("ref",) + MODES:
            b = max((c for c in configs() if c[0] == mode), key=lambda c: t[c]["spell"])
            print(f"   best-on-this-match (optimistic) {mode:12s} {b[1:]} spell {t[b]['spell']:.4f}")
    (ROOT / "logs" / f"ballradar_partial_{'_'.join(mids)}.json").write_text(json.dumps(
        {m: {f"{c[0]}|{c[1]}|{c[2]}|{c[3]}": v for c, v in tables[m].items()} for m in mids}, indent=1))


def main():
    import sys
    if len(sys.argv) > 1:
        return single(sys.argv[1:])
    aux = {m: ball_and_team(m) for m in TUNE}
    for m, a in aux.items():
        print(m, json.dumps(a), flush=True)
    with Pool(3) as pool:
        tables = dict(pool.map(match_task, TUNE))
    cfgs = configs()
    report = {"aux": aux, "modes": {}}
    for mode in ("ref",) + MODES:
        cells = [c for c in cfgs if c[0] == mode]
        if mode == "ref":
            sel_cells = cells
        else:
            sel_cells = cells
        held = {}
        for m in TUNE:
            best = max(sel_cells, key=lambda c: np.mean([tables[k][c]["spell"] for k in TUNE if k != m]))
            held[m] = {"cell": list(best), **tables[m][best]}
        sp = np.array([h["spell"] for h in held.values()])
        best_all = max(sel_cells, key=lambda c: np.mean([tables[k][c]["spell"] for k in TUNE]))
        report["modes"][mode] = {
            "lomo_spell_mean": float(sp.mean()), "lomo_spell_min": float(sp.min()), "lomo_spell_max": float(sp.max()),
            "lomo_macro_mean": float(np.mean([h["macro"] for h in held.values()])),
            "spells_pred_gt": [[h["n_pred"], h["n_gt"]] for h in held.values()],
            "held_out": held, "best_on_all": list(best_all),
            "best_on_all_per_match_spell": {m: tables[m][best_all]["spell"] for m in TUNE},
            "best_on_all_mean_spell": float(np.mean([tables[m][best_all]["spell"] for m in TUNE]))}
        r = report["modes"][mode]
        print(f"{mode:12s} LOMO spellF1 {r['lomo_spell_mean']:.4f} [{r['lomo_spell_min']:.4f}, {r['lomo_spell_max']:.4f}] "
              f"macro {r['lomo_macro_mean']:.4f} pred/gt {r['spells_pred_gt']} best_on_all {r['best_on_all']} "
              f"({r['best_on_all_mean_spell']:.4f})", flush=True)
    report["grid"] = {"WS": WS, "SWITCH_TEAM": SW_T, "SWITCH_DEAD": SW_D}
    report["table"] = {f"{c[0]}|{c[1]}|{c[2]}|{c[3]}": {m: tables[m][c] for m in TUNE} for c in cfgs}
    (ROOT / "logs" / "ballradar_lomo.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
