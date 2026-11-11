"""Label-free check: why do we label D on clear-control frames? Human only for the competition data; `--idsse` gives the baseline.

    python -m possession.dead_check <your submission csv>     # own terminal, prints aggregates only
    python -m possession.dead_check --idsse                   # IDSSE TUNE: same numbers for v8 (LOMO) and for the truth

Background (tag_truth_check): on the competition match 13.0% of clear-control frames (ball seen, one player within 1.5 m, no
opponent within 3 m, >= 0.6 s) are labelled D, against 5.0% for v8 and 4.7% for the truth on IDSSE. Two readings:
  (a) restarts: the taker stands on the ball during a stoppage, and the real ball is visible there more than in our simulator;
  (b) false D: we cut live possessions with short D runs (H D H with the same player on the ball) -> split possessions.
The two leave different traces: (a) long D runs inside Smart Tagging set-piece intervals, (b) short D runs between the same team.
"""
import sys

import numpy as np

from .tag_truth_check import clear_control

BINS = ((0, 1), (1, 3), (3, 10), (10, 1e9))


def runs(lab):
    ch = np.flatnonzero(np.diff(lab)) + 1
    return np.r_[0, ch], np.r_[ch, len(lab)]


def stats(lab, cc, setp=None, fps=25):
    """lab: int array 0 H / 1 A / 2 D / 3 X; cc: clear_control output; setp: bool mask of set-piece tag frames or None."""
    st, en = runs(lab)
    v = lab[st]
    kD = (cc >= 0) & (lab == 2)
    out = {"clear_D_share": kD.sum() / max((cc >= 0).sum(), 1), "n_D_runs": 0, "bins": {}}
    tot = kD.sum()
    same_short = flank_same = 0
    for j, (s, e) in enumerate(zip(st, en)):
        if v[j] != 2:
            continue
        out["n_D_runs"] += 1
        d = (e - s) / fps
        b = next(x for x in BINS if x[0] <= d < x[1])
        r = out["bins"].setdefault(b, [0, 0])
        r[0] += 1; r[1] += kD[s:e].sum()
        same = 0 < j < len(v) - 1 and v[j - 1] == v[j + 1] and v[j - 1] in (0, 1)
        flank_same += kD[s:e].sum() * same
        same_short += kD[s:e].sum() * (same and d < 3)
    out["clear_D_in_same_team_flank"] = flank_same / max(tot, 1)
    out["clear_D_in_short_same_team_flank"] = same_short / max(tot, 1)
    if setp is not None:
        out["clear_D_in_setpiece_tag"] = (kD & setp).sum() / max(tot, 1)
        out["D_frames_in_setpiece_tag"] = ((lab == 2) & setp).sum() / max((lab == 2).sum(), 1)
    return out


def show(name, o):
    print(f"{name}: clear frames labelled D {o['clear_D_share']:.3f}; D runs {o['n_D_runs']}; of clear-D frames: in D runs between the "
          f"same team {o['clear_D_in_same_team_flank']:.3f}, of which runs < 3 s {o['clear_D_in_short_same_team_flank']:.3f}"
          + (f"; inside set-piece tags {o['clear_D_in_setpiece_tag']:.3f} (all D frames {o['D_frames_in_setpiece_tag']:.3f})" if "clear_D_in_setpiece_tag" in o else ""))
    tot = sum(r[1] for r in o["bins"].values())
    print("   D run length: " + "  ".join(f"{a}-{b if b < 1e9 else 'inf'}s: {r[0]} runs, {r[1] / max(tot, 1):.2f} of clear-D"
                                     for (a, b), r in sorted(o["bins"].items())))


def main(csv):
    from .smart_tagging import mtah_guard
    mtah_guard.refuse_under_claude()
    import games
    from . import bar_ath
    from .tag_check import parse_all
    ball, xy, team = bar_ath.load_atd()
    txt = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=1, dtype=str)
    n = min(len(txt), len(ball))
    lab = np.full(n, 3, np.int8)
    for k, ch in enumerate("HAD"):
        lab[txt[:n] == ch] = k
    setp = np.zeros(n, bool)
    for code, a, b, t in parse_all(games.require(["smart_tagging"], games.GAME)[0]):
        if code == "set pieces":
            setp[max(a, 0):b + 1] = True
    show("competition, your labels", stats(lab, clear_control(ball[:n], xy[:n], team), setp))


def idsse():
    import joblib
    from . import m2_lib as L, v7_decoder as E, v4_lib as V
    from .data import TUNE
    data, c = L.data2(), joblib.load(E.CACHE)
    P, T = [], []
    for m in TUNE:
        fs = [np.load(E.LOG / "v8" / f"m2_s{s}_{m}.npz") for s in range(20, 25)]
        for i, h in enumerate(data[m]):
            t = np.asarray(h["truth"]).astype(np.int8); n = len(t)
            lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
            b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
            p = V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b).astype(np.int8)
            cc = clear_control(h["ball"], h["xy"], h["team"])
            sep = np.array([3], np.int8)   # half boundary, keeps runs from joining across halves
            P.append((np.r_[p, sep], np.r_[cc, -1])); T.append((np.r_[t, sep], np.r_[cc, -1]))
    for name, Z in (("IDSSE v8 LOMO", P), ("IDSSE truth", T)):
        show(name, stats(np.concatenate([z[0] for z in Z]), np.concatenate([z[1] for z in Z])))


if __name__ == "__main__":
    idsse() if sys.argv[1] == "--idsse" else main(sys.argv[1])
