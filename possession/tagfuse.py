"""v5: Smart Tagging team prior added to the Viterbi emissions, and the IDSSE tag simulator used to choose its weight.

Only the attacking phase codes are used: they name the team in possession. The DEFENDING / defensive-transition
codes duplicate those intervals with the defender as team and are dropped (real tag_check aggregates, 2026-11).
Tag evidence inside an interval of code c with team s in {H, A}: e[:, s] += w_c, e[:, other] -= w_c, so the D column
is untouched. Nothing here loads competition data (tags come in through smart_tagging.load_phases, guarded).
"""
import numpy as np

H, A = 0, 1
GRADED = 158857   # graded frames of the competition match, scales interval counts per IDSSE half

# code: (n instances in the competition file, alive frames in intervals, v4 D share, agree with v4 on alive H/A frames, anchor)
# Real label-free aggregates pasted by the human (tag_check vs v4 labels). anchor: where the simulator starts intervals.
CODES = {
    "build up": (22, 10282, 0.068, 0.943, "alive"),
    "progression": (66, 29192, 0.064, 0.880, "alive"),
    "attacking transition": (24, 9224, 0.010, 0.683, "alive"),
    "long ball": (29, 8607, 0.100, 0.681, "alive"),
    "ball in final third": (93, 46289, 0.340, 0.685, "alive"),
    "creating chances": (30, 11899, 0.440, 0.664, "alive"),
    "ball in the box": (72, 24192, 0.259, 0.630, "alive"),
    "players in the box": (97, 45414, 0.473, 0.591, "alive"),
    "set pieces": (59, 15207, 0.336, 0.606, "dead"),
    "goals": (4, 1667, 0.899, 0.786, "dead"),
}
JIT = 37            # start / end jitter sigma, frames (1.5 s)
SIG = 0.4           # lognormal sigma of interval length (mean fixed to the real mean)


def length(code):
    n, alive, d, _, _ = CODES[code]
    return alive / n / (1.0 - d)


def weights(k, rel=None):
    """w_c = k * logit(reliability_c), reliability = real agreement with v4 (a lower bound on the tag's accuracy)."""
    rel = rel or {c: v[3] for c, v in CODES.items()}
    return {c: float(k * np.log(r / (1 - r))) for c, r in rel.items()}


def bias(n, intervals, w):
    """intervals [(code, a, b, team 0/1)] in half-local frames, b inclusive -> (n, 2) additive emission bias."""
    d = np.zeros((n + 1, 2))
    for code, a, b, s in intervals:
        if code not in w:
            continue
        a, b = max(int(a), 0), min(int(b), n - 1)
        if b < a:
            continue
        d[a, s] += w[code]; d[b + 1, s] -= w[code]
        d[a, 1 - s] -= w[code]; d[b + 1, 1 - s] += w[code]
    return np.cumsum(d, 0)[:n]


def spells(truth):
    """Alive same-team runs (dead gaps ignored): [(start, end, team)], and dead runs: [(start, end of next spell, next team)]."""
    idx = np.flatnonzero(truth < 2)
    if len(idx) == 0:
        return [], []
    t = truth[idx]
    cut = np.flatnonzero(np.diff(t) != 0) + 1
    st, en = np.r_[0, cut], np.r_[cut - 1, len(idx) - 1]
    sp = [(int(idx[a]), int(idx[b]), int(t[a])) for a, b in zip(st, en)]
    dead = []
    isd = (truth == 2).astype(np.int8)
    d0 = np.flatnonzero(np.diff(np.r_[0, isd]) == 1)
    starts = np.array([s[0] for s in sp])
    for a in d0:
        j = np.searchsorted(starts, a)
        if j < len(sp):
            dead.append((int(a), sp[j][1], sp[j][2]))
    return sp, dead


def simulate(truth, sp, dead, theta, rng, codes=None):
    """One simulated tag file for a half. theta = {code: (p_flip, q_trunc)}. Interval = anchor start (a spell start, or the
    start of a dead run for set pieces / goals) + jitter, length lognormal with the real mean; spells are drawn in proportion to
    their length (phases sit in long possessions); with prob q_trunc the interval is clean (no jitter, cut at the spell end)
    instead of spilling into neighbouring possessions; the team of the anchor spell is flipped with p_flip."""
    n = len(truth)
    out = []
    for code in (codes or CODES):
        pool = sp if CODES[code][4] == "alive" else dead
        if not pool:
            continue
        p_flip, q = theta[code]
        L = length(code)
        pr = np.array([e - s + 1 for s, e, _ in pool], float) if CODES[code][4] == "alive" else np.ones(len(pool))
        pr /= pr.sum()
        for _ in range(rng.poisson(CODES[code][0] * n / GRADED)):
            s, e, team = pool[rng.choice(len(pool), p=pr)]
            a = s + int(round(rng.normal(0, JIT)))
            b = a + L * rng.lognormal(-0.5 * SIG ** 2, SIG)
            if rng.random() < q:       # clean tag: starts at the spell start, ends with the spell
                a, b = s, min(b, e)
            if rng.random() < p_flip:
                team = 1 - team
            a, b = max(a, 0), min(int(round(b)), n - 1)
            if b > a:
                out.append((code, a, b, team))
    return out


def agreement(pred, intervals):
    """{code: [alive frames, agreeing frames]} of predictions vs interval teams (what tag_check prints)."""
    cs = [np.r_[0, np.cumsum(pred == s)] for s in (H, A)]
    acc = {}
    for code, a, b, s in intervals:
        r = acc.setdefault(code, [0, 0])
        h, aw = cs[0][b + 1] - cs[0][a], cs[1][b + 1] - cs[1][a]
        r[0] += int(h + aw)
        r[1] += int((h, aw)[s])
    return acc
