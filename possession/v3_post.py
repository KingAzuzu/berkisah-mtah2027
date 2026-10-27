"""Post-decoding operations for v3: (a) receiver-arrival boundary snap, (b) short-gap merge / short-spell absorb."""
import numpy as np

from .data import pm
from .v3_lib import from_runs, merged, runs


def arrival_dist(half):
    """(T, 2) distance from the seen ball to the nearest home / away player; NaN where the ball is not seen."""
    ball, xy, team = half["ball"], half["xy"], half["team"]
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    out = np.full((len(ball), 2), np.nan)
    with np.errstate(all="ignore"):
        for k in (0, 1):
            out[:, k] = np.fmin.reduce(np.where(team == k, d, np.nan), axis=1)
    return out


def snap(path, dist, N, R):
    """Move each direct H|A boundary to the nearest 'arrival' frame within +-N frames: a frame where the ball is
    seen, the new team's nearest player is within R and was not (seen and within R) the frame before. Boundary
    stays inside the adjacent runs (cannot cross another boundary). Unchanged if no arrival in the window."""
    rs = runs(path)
    T = len(path)
    out = [list(r) for r in rs]
    for i in range(1, len(out)):
        a0, b0, c0 = out[i - 1]
        a1, b1, c1 = out[i]
        if c0 == pm.D or c1 == pm.D or c0 == c1:
            continue
        b = a1                                  # first frame of the new team
        lo, hi = max(a0 + 1, b - N), min(b1 - 1, b + N)   # new boundary t in [lo, hi]: runs keep >= 1 frame
        d = dist[:, c1]
        near = np.isfinite(d) & (d < R)
        entry = near.copy()
        entry[1:] &= ~near[:-1]
        cand = np.flatnonzero(entry[lo:hi + 1]) + lo
        if len(cand) == 0:
            continue
        t = int(cand[np.argmin(np.abs(cand - b))])
        out[i - 1][1] = t
        out[i][0] = t
    return from_runs([tuple(r) for r in out], T)


def cleanup(path, K, L):
    """(1) fill D runs shorter than K frames lying between two runs of the same team; (2) absorb team spells shorter
    than L frames into the longer neighbouring run (shortest first). K = 0 / L = 0 disable each step."""
    T = len(path)
    rs = merged(runs(path))
    if K > 0:
        out = list(rs)
        for i in range(1, len(rs) - 1):
            a, b, c = rs[i]
            if c == pm.D and b - a < K and rs[i - 1][2] == rs[i + 1][2] != pm.D:
                out[i] = (a, b, rs[i - 1][2])
        rs = merged(out)
    if L > 0:
        while True:
            short = [(b - a, i) for i, (a, b, c) in enumerate(rs) if c != pm.D and b - a < L]
            if not short or len(rs) == 1:
                break
            _, i = min(short)
            a, b, c = rs[i]
            left = rs[i - 1] if i > 0 else None
            right = rs[i + 1] if i < len(rs) - 1 else None
            if left is None:
                tgt = right[2]
            elif right is None:
                tgt = left[2]
            elif left[2] == right[2]:
                tgt = left[2]
            else:
                tgt = left[2] if (left[1] - left[0]) >= (right[1] - right[0]) else right[2]
            rs[i] = (a, b, tgt)
            rs = merged(rs)
    return from_runs(rs, T)
