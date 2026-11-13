"""v16 (CPU, no training): control-aware touch merge, from CHALLENGE.md "a simple touch of the ball without control" is not a change.

A predicted X-Y-X with Y shorter than K s is merged into X unless the tracking shows Y in clear control somewhere inside Y
(tag_truth_check.clear_control: ball seen, own player <= 1.5 m, no opponent within 3 m, >= 0.6 s). v8 merges every such Y blindly.
Variants fixed beforehand: base in (v10, v6), K in (2, 3) s. IDSSE TUNE LOMO nets, matched2 seed 0, organiser scorer.
Gate fixed beforehand: on the per-match average of (DFL, merged) spellF1, mean >= v10 + 0.01 and >= 4/5 wins vs v10.
python -m possession.v16_ctrl_merge -> logs/v16/ctrl_merge.json"""
import json
import numpy as np
from . import m2_lib as L, v7_decoder as E, v4_lib as V
from .data import TUNE, pm
from .tag_truth_check import clear_control
from .v10_hedge import nets, c, score_vs


def ctrl_merge(t, cc, k_s, fps=25):
    t = np.asarray(t).copy()
    keep = set()
    while True:
        ch = np.flatnonzero(np.diff(t)) + 1
        st, en = np.r_[0, ch], np.r_[ch, len(t)]
        v = t[st]
        cand = [j for j in range(1, len(v) - 1) if v[j] != pm.D and v[j - 1] == v[j + 1] and v[j - 1] not in (pm.D, v[j])
                and en[j] - st[j] < k_s * fps and (st[j], en[j]) not in keep]
        if not cand:
            return t
        j = min(cand, key=lambda j: en[j] - st[j])
        if np.any(cc[st[j]:en[j]] == v[j]):
            keep.add((st[j], en[j]))
        else:
            t[st[j]:en[j]] = v[j - 1]


def decode(N):
    return {m: [V.viterbi_scaled(0.5 * N[m][i][0] + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * N[m][i][1]) for i in range(2)]
            for m in TUNE}


if __name__ == "__main__":
    assert pm.H == 0 and pm.A == 1
    data = L.data2()
    CC = {m: [clear_control(h["ball"], h["xy"], h["team"]) for h in data[m]] for m in TUNE}
    N6, N8 = nets(E.LOG / "v6", "e2"), nets(E.LOG / "v8", "m2")
    NA = {m: [((N6[m][i][0] + N8[m][i][0]) / 2, (N6[m][i][1] + N8[m][i][1]) / 2) for i in range(2)] for m in TUNE}
    base = {"v10": decode(NA), "v6": decode(N6)}
    cands = {"v10": base["v10"]}
    for b in ("v10", "v6"):
        for k in (2, 3):
            cands[f"{b}+ctrl{k}"] = {m: [ctrl_merge(p, CC[m][i], k) for i, p in enumerate(base[b][m])] for m in TUNE}
    res = {}
    for name, P in cands.items():
        res[name] = {}
        for tn in ("dfl", "merged"):
            s, cnt = score_vs(P, tn)
            res[name][tn] = {"mean": float(np.mean(s)), "per_match": s, "pred_counts": cnt}
        avg = [(a + b) / 2 for a, b in zip(res[name]["dfl"]["per_match"], res[name]["merged"]["per_match"])]
        res[name]["avg"] = {"mean": float(np.mean(avg)), "per_match": avg}
        print(f"{name:10s} dfl {res[name]['dfl']['mean']:.4f} merged {res[name]['merged']['mean']:.4f} avg {np.mean(avg):.4f}",
              " ".join(f"{x:.3f}" for x in avg), res[name]["dfl"]["pred_counts"], flush=True)
    ref = res["v10"]["avg"]
    for name in cands:
        if name == "v10":
            continue
        wins = sum(a > b for a, b in zip(res[name]["avg"]["per_match"], ref["per_match"]))
        res[name]["gate"] = bool(res[name]["avg"]["mean"] >= ref["mean"] + 0.01 and wins >= 4)
        print(f"{name:10s} wins {wins}/5 gate {'PASS' if res[name]['gate'] else 'FAIL'}")
    (E.LOG / "v16").mkdir(exist_ok=True)
    json.dump(res, open(E.LOG / "v16" / "ctrl_merge.json", "w"), indent=1)
