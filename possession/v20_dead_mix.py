"""v20 (CPU, no TCN retraining): dead-ball evidence that does not rely on a missing ball when the ball is visible.

Finding (public window correctness check, 16 Nov 2026, aggregates): v10 D recall 0.33. Of 4 truth D runs, the two with the ATD ball
seen 100% and slow (2.8, 1.7 m/s; 20 s and 15 s) get v10 D 0.00, while v2 (dead model WITHOUT ball features) gets 0.94 and 1.00.
v2.1 (with ball features) already has D recall 0.25: the ball features learned "ball missing = dead" from the simulator.
Variants fixed beforehand (LOMO dead models on matched2 TUNE, same as m2_lib.r_models, v10 LOMO nets, v10 decoder):
  mix     : p_dead = players-only model where the ball is seen, with-ball model where it is missing   (default choice)
  players : players-only model everywhere
Gate fixed beforehand (no harm; IDSSE cannot show the gain because the simulator lacks the case): avg(DFL, merged) mean
>= v10 - 0.005. The choice between mix and players is NOT made from the public window.
python -m possession.v20_dead_mix -> logs/v20/dead_mix.json"""
import json
import numpy as np
from . import m2_lib as L, pipeline as pl, v4_lib as V, v7_decoder as E
from .m2_lib import IR
from .data import TUNE, pm
from .v10_hedge import nets, c, score_vs


def p_dead(P, variant):
    keep = pl.keep_columns(P[TUNE[0]][0]["names"], variant)
    return {m: IR.predict_p(IR.train_on(P, [k for k in TUNE if k != m], keep), P, m, keep) for m in TUNE}


if __name__ == "__main__":
    data, P = L.data2(), L.prep2()
    pb, pp = p_dead(P, "ball"), p_dead(P, "players")
    N6, N8 = nets(E.LOG / "v6", "e2"), nets(E.LOG / "v8", "m2")
    NA = {m: [((N6[m][i][0] + N8[m][i][0]) / 2, (N6[m][i][1] + N8[m][i][1]) / 2) for i in range(2)] for m in TUNE}
    a = L.PARAMS["ALPHA"]

    def run(get_e):
        return {m: [V.viterbi_scaled(0.5 * NA[m][i][0] + 0.5 * get_e(m, i), 20.0, 5.0, 1 - 0.6 * NA[m][i][1]) for i in range(2)]
                for m in TUNE}

    seen = {m: [~np.isnan(h["ball"][:, 0]) for h in data[m]] for m in TUNE}
    cands = {
        "v10": run(lambda m, i: c[m][i]["eR"]),
        "v10_recomputed": run(lambda m, i: pl.combine(P[m][i]["team"], pb[m][i], a)),
        "mix": run(lambda m, i: pl.combine(P[m][i]["team"], np.where(seen[m][i], pp[m][i], pb[m][i]), a)),
        "players": run(lambda m, i: pl.combine(P[m][i]["team"], pp[m][i], a)),
    }
    res = {}
    for name, Pr in cands.items():
        res[name] = {}
        for tn in ("dfl", "merged"):
            s, cnt = score_vs(Pr, tn)
            res[name][tn] = {"mean": float(np.mean(s)), "per_match": s, "pred_counts": cnt}
        avg = [(x + y) / 2 for x, y in zip(res[name]["dfl"]["per_match"], res[name]["merged"]["per_match"])]
        res[name]["avg"] = float(np.mean(avg))
        t = np.concatenate([h["truth"] for m in TUNE for h in data[m]])
        pr = np.concatenate([x for m in TUNE for x in Pr[m]])
        sv = np.concatenate([x for m in TUNE for x in seen[m]])
        dd = t == pm.D
        res[name]["D_recall_ball_seen"] = float(np.mean(pr[dd & sv] == pm.D))
        res[name]["D_recall_ball_missing"] = float(np.mean(pr[dd & ~sv] == pm.D))
        res[name]["D_precision"] = float(np.mean(t[pr == pm.D] == pm.D))
        print(f"{name:15s} dfl {res[name]['dfl']['mean']:.4f} merged {res[name]['merged']['mean']:.4f} avg {res[name]['avg']:.4f} | "
              f"D recall seen {res[name]['D_recall_ball_seen']:.3f} missing {res[name]['D_recall_ball_missing']:.3f} "
              f"prec {res[name]['D_precision']:.3f}", flush=True)
    for name in ("mix", "players"):
        res[name]["gate_no_harm"] = bool(res[name]["avg"] >= res["v10"]["avg"] - 0.005)
        print(name, "no-harm gate", "PASS" if res[name]["gate_no_harm"] else "FAIL")
    (E.LOG / "v20").mkdir(exist_ok=True)
    json.dump(res, open(E.LOG / "v20" / "dead_mix.json", "w"), indent=1)
