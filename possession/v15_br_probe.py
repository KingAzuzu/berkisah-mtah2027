"""v15 probe (CPU, screen only): add Ball Radar team evidence to the v10 LOMO evidence on IDSSE TUNE.

Ball Radar held-out probs come from the OLD degrade profile (features.CACHE/heldout_{m}.pkl, fold nets not saved), while
v10 evidence uses matched2. Different ball-visibility realisation -> this screen is optimistic. Pass only means: retrain
Ball Radar on matched2 and run the real gate. Fixed beforehand: w in (0.25, 0.5, 1.0), team-only addition
em = log(2 sigmoid(+-z)), z = log pH/pA; screen pass if some w has mean >= v10 + 0.01 on BOTH truths and >= 4/5 wins.
python -m possession.v15_br_probe -> logs/v15/probe.json"""
import json
import pickle
import numpy as np
from . import v7_decoder as E, v4_lib as V
from .ballradar import features as fe
from .data import TUNE, pm
from .v10_hedge import nets, c, score_vs

WS = (0.25, 0.5, 1.0)


def br_em(m):
    out = []
    for i, ho in enumerate(pickle.loads((fe.CACHE / f"heldout_{m}.pkl").read_bytes())):
        p = ho["probs"]
        assert len(p) == len(c[m][i]["truth"]), (m, i, len(p), len(c[m][i]["truth"]))
        z = np.clip(np.log(p[:, 0] + 1e-6) - np.log(p[:, 1] + 1e-6), -20, 20)
        em = np.zeros((len(p), 3))
        em[:, pm.H] = np.log(2.0) - np.logaddexp(0, -z)
        em[:, pm.A] = np.log(2.0) - np.logaddexp(0, z)
        out.append(em)
    return out


if __name__ == "__main__":
    N6, N8 = nets(E.LOG / "v6", "e2"), nets(E.LOG / "v8", "m2")
    NA = {m: [((N6[m][i][0] + N8[m][i][0]) / 2, (N6[m][i][1] + N8[m][i][1]) / 2) for i in range(2)] for m in TUNE}
    EM = {m: br_em(m) for m in TUNE}
    res = {}
    for w in (0.0,) + WS:
        P = {m: [V.viterbi_scaled(0.5 * NA[m][i][0] + 0.5 * c[m][i]["eR"] + w * EM[m][i], 20.0, 5.0,
                                  1 - 0.6 * NA[m][i][1]) for i in range(2)] for m in TUNE}
        res[w] = {}
        for tn in ("dfl", "merged"):
            s, cnt = score_vs(P, tn)
            res[w][tn] = {"mean": float(np.mean(s)), "per_match": s, "pred_counts": cnt}
            print(f"w={w:<5} {tn:6s} {np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s), cnt, flush=True)
    for w in WS:
        ok = all(res[w][tn]["mean"] >= res[0.0][tn]["mean"] + 0.01 and
                 sum(a > b for a, b in zip(res[w][tn]["per_match"], res[0.0][tn]["per_match"])) >= 4
                 for tn in ("dfl", "merged"))
        res[w]["screen_pass"] = ok
        print(f"w={w} screen {'PASS' if ok else 'FAIL'}")
    (E.LOG / "v15").mkdir(exist_ok=True)
    json.dump({str(k): v for k, v in res.items()}, open(E.LOG / "v15" / "probe.json", "w"), indent=1)
