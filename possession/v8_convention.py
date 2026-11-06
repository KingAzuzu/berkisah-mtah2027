"""v8 convention check (IDSSE TUNE, matched2, v6 LOMO log-probs, v6 decoder). Diagnostic, nothing frozen.
CHALLENGE.md: a touch without control / a deflection that does not intercept is NOT a change of possession. DFL truth flips on such
touches. 'merge K' = an alive spell shorter than K s directly between two spells of the other team (X Y X, no D) is relabelled X.
K = 2 s fixed a priori ("keeps the ball a few seconds"); 1 and 3 reported as sensitivity only.
Scores v6 raw and v6 + merge against DFL truth and merged truth. python -m possession.v8_convention"""
import json
import numpy as np, joblib
from . import v7_decoder as E, v4_lib as V
from .data import TUNE
from .score import score
from evaluation import possession_metrics as pm


def merge(t, k_s, fps=25):
    t = np.asarray(t).copy()
    while True:
        ch = np.flatnonzero(np.diff(t)) + 1
        st, en = np.r_[0, ch], np.r_[ch, len(t)]
        v = t[st]
        cand = [j for j in range(1, len(v) - 1) if v[j] != pm.D and v[j - 1] == v[j + 1] and v[j - 1] not in (pm.D, v[j])
                and en[j] - st[j] < k_s * fps]
        if not cand:
            return t
        j = min(cand, key=lambda j: en[j] - st[j])   # shortest first, then re-segment
        t[st[j]:en[j]] = v[j - 1]


def main():
    c = joblib.load(E.CACHE)
    D = E.LOG / "v6"
    preds = {}
    for m in TUNE:
        fs = [np.load(D / f"e2_s{s}_{m}.npz") for s in range(20, 25)]
        p = []
        for i in range(2):
            n = len(c[m][i]["truth"])
            lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
            b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
            p.append(V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b))
        preds[m] = p
    out = {}
    for K in (1, 2, 3):
        rows = {}
        for tname in ("dfl", "merged"):
            for pname in ("v6", "v6+merge"):
                s, cnt = [], []
                for m in TUNE:
                    halves = [dict(h, truth=(merge(h["truth"], K) if tname == "merged" else h["truth"])) for h in c[m]]
                    pr = [merge(x, K) if pname == "v6+merge" else x for x in preds[m]]
                    r = score(halves, pr)
                    s.append(r["spell_f1_mean"]); cnt.append((int(r["n_pred_spells"]), int(r["n_gt_spells"])))
                rows[f"{pname} vs {tname}"] = {"mean": float(np.mean(s)), "per_match": s, "counts": cnt}
                print(K, f"{pname:9s} vs {tname:6s} {np.mean(s):.4f}", " ".join(f"{x:.3f}" for x in s), cnt)
        out[K] = rows
    (E.LOG / "v8").mkdir(exist_ok=True)
    json.dump(out, open(E.LOG / "v8" / "convention.json", "w"), indent=1)


if __name__ == "__main__":
    main()
