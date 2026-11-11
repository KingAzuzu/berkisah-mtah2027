"""Label-free triangulation: are the Smart Tagging team tags right where our labels disagree with them? Human only for the
competition data; `--idsse` mode runs on IDSSE TUNE (public) to calibrate the yardstick.

    python -m possession.tag_truth_check <your submission csv>     # own terminal, prints aggregates only
    python -m possession.tag_truth_check --idsse                   # IDSSE: how good is the proxy, how good are we when the ball is hidden

Why: v5 rejected the tag team prior assuming that disagreement between tags and our labels (0.59-0.69 for the box / final-third
codes) is tag noise. The other reading is that our labels are wrong there (ball hidden in crowds) and the tags are right.
Yardstick, uses the input tracking only (no ground truth): a "clear control" frame = ball seen, nearest player within R_NEAR,
nearest player of the other team farther than R_OPP, for at least MIN_RUN consecutive frames; its team is a high-precision
proxy of possession. Constants fixed a priori. Per tag code the script prints
  tag=proxy   : tag team vs proxy on clear frames inside the intervals      (high -> tags are right when it can be checked)
  label=proxy : our label vs proxy on the same frames (where we say H/A)    (should be high: we see the ball there)
  tag=label|seen / |hidden : tag team vs our label, split by ball visibility (a drop on hidden frames with tag=proxy high
                              points at our labels, not at the tags)
"""
import sys
from collections import defaultdict

import numpy as np

R_NEAR, R_OPP, MIN_RUN = 1.5, 3.0, 15   # m, m, frames (0.6 s at 25 fps)


def clear_control(ball, xy, team, r_near=R_NEAR, r_opp=R_OPP, min_run=MIN_RUN):
    """-> (T,) int8: 0 home / 1 away on clear-control frames, -1 elsewhere."""
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    dh, da = d[:, team == 0].min(1), d[:, team == 1].min(1)
    ok = ~np.isnan(ball[:, 0]) & (np.minimum(dh, da) <= r_near) & (np.maximum(dh, da) >= r_opp)
    out = np.where(ok, np.where(dh <= da, 0, 1), -1).astype(np.int8)
    ch = np.flatnonzero(np.diff(out)) + 1
    for s, e in zip(np.r_[0, ch], np.r_[ch, len(out)]):
        if out[s] >= 0 and e - s < min_run:
            out[s:e] = -1
    return out


def _r(a, b):
    return a / b if b else float("nan")


def report(lab, ball, xy, team, intervals):
    """lab: (T,) array of 'H'/'A'/'D'/'X'; intervals: (code, first frame, last frame, 'H'|'A'|None). Prints aggregates only."""
    n = min(len(lab), len(ball))
    lab, ball, xy = lab[:n], ball[:n], xy[:n]
    cc, seen = clear_control(ball, xy, team), ~np.isnan(ball[:, 0])
    L = np.full(n, -1, np.int8)
    L[lab == "H"], L[lab == "A"] = 0, 1
    g = lab != "X"
    m, al = g & (cc >= 0), g & (L >= 0)
    print(f"clear-control frames: {_r((m & al).sum(), al.sum()):.3f} of our alive frames (IDSSE truth 0.253); "
          f"of all clear frames we label D {_r((m & (lab == 'D')).sum(), m.sum()):.3f} (IDSSE truth 0.047); "
          f"label=proxy where we say H/A {_r((m & (L == cc)).sum(), (m & al).sum()):.3f} (IDSSE v8 0.990)")
    print(f"our alive frames: ball seen {_r((al & seen).sum(), al.sum()):.3f} (IDSSE v8 0.904)")
    acc = defaultdict(lambda: np.zeros(11))
    for code, a, b, t in intervals:
        if t is None:
            continue
        t = 0 if t == "H" else 1
        s = slice(max(a, 0), b + 1)
        c, l, sn, gg = cc[s], L[s], seen[s], g[s]
        k = gg & (c >= 0)
        k2 = k & (l >= 0)
        vs, vh = gg & (l >= 0) & sn, gg & (l >= 0) & ~sn
        acc[code] += [1, gg.sum(), (gg & sn).sum(), k.sum(), (k & (c == t)).sum(), k2.sum(), (k2 & (l == c)).sum(),
                      vs.sum(), (vs & (l == t)).sum(), vh.sum(), (vh & (l == t)).sum()]
    print(f"{'code':<30}{'n':>5}{'seen':>7}{'clear fr':>10}{'tag=proxy':>11}{'label=proxy':>13}{'tag=label|seen':>16}{'tag=label|hidden':>18}{'hidden fr':>11}")
    for code, r in sorted(acc.items(), key=lambda kv: -kv[1][0]):
        print(f"{code:<30}{int(r[0]):>5}{_r(r[2], r[1]):>7.3f}{int(r[3]):>10}{_r(r[4], r[3]):>11.3f}{_r(r[6], r[5]):>13.3f}"
              f"{_r(r[8], r[7]):>16.3f}{_r(r[10], r[9]):>18.3f}{int(r[9]):>11}")


def main(csv):
    from .smart_tagging import mtah_guard
    mtah_guard.refuse_under_claude()
    import games
    from . import bar_ath
    from .tag_check import parse_all
    ball, xy, team = bar_ath.load_atd()
    lab = np.loadtxt(csv, delimiter=",", skiprows=1, usecols=1, dtype=str)
    report(lab, ball, xy, team, parse_all(games.require(["smart_tagging"], games.GAME)[0]))


def idsse():
    """IDSSE TUNE (matched2 degradation): proxy quality against truth, and our team accuracy by ball visibility."""
    import joblib
    from . import m2_lib as L, v7_decoder as E, v4_lib as V
    from .v8_convention import merge
    from .data import TUNE
    from evaluation import possession_metrics as pm
    data, c = L.data2(), joblib.load(E.CACHE)
    tot = np.zeros(12)
    for m in TUNE:
        for i, h in enumerate(data[m]):
            t = np.asarray(h["truth"]); tm = merge(t, 2)
            cc, seen = clear_control(h["ball"], h["xy"], h["team"]), ~np.isnan(h["ball"][:, 0])
            alive, k = t != pm.D, cc >= 0
            fs = [np.load(E.LOG / "v8" / f"m2_s{s}_{m}.npz") for s in range(20, 25)]
            n = len(t)
            lp = V.upsample(np.mean([f[f"lp{i}"] for f in fs], 0), n)
            b = V.upsample(np.mean([f[f"b{i}"] for f in fs], 0)[:, None], n)[:, 0]
            p = V.viterbi_scaled(0.5 * lp + 0.5 * c[m][i]["eR"], 20.0, 5.0, 1 - 0.6 * b)
            both = alive & (p != pm.D)
            tot += [alive.sum(), (alive & k).sum(), (alive & k & (cc == t)).sum(), (alive & k & (cc == tm)).sum(), k.sum(), (k & ~alive).sum(),
                    (both & seen).sum(), (both & seen & (p == tm)).sum(), (both & ~seen).sum(), (both & ~seen & (p == tm)).sum(),
                    (both & k).sum(), (both & k & (p == cc)).sum()]
    print(f"IDSSE TUNE, matched2. clear-control frames = {_r(tot[1], tot[0]):.3f} of truth-alive frames; {_r(tot[5], tot[4]):.3f} of clear frames are truth D")
    print(f"proxy = DFL truth {_r(tot[2], tot[1]):.3f}; proxy = merged truth {_r(tot[3], tot[1]):.3f} (clear, truth-alive frames)")
    print(f"v8 LOMO team accuracy vs merged truth (both alive): ball seen {_r(tot[7], tot[6]):.3f} ({int(tot[6])} fr), ball hidden {_r(tot[9], tot[8]):.3f} ({int(tot[8])} fr)")
    print(f"v8 label = proxy on clear frames {_r(tot[11], tot[10]):.3f}")


if __name__ == "__main__":
    idsse() if sys.argv[1] == "--idsse" else main(sys.argv[1])
