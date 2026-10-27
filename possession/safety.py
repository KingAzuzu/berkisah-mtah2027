"""Safety net: nearest-player evidence and dead-ball evidence, decoded with a 3-state Viterbi.

Per frame, three log-scores (H, A, D), each 0 when that frame says nothing:
  team  ball seen and someone within R_CLOSE: logit (d_away - d_home) / SCALE, as log(2 sigmoid)
        so an uninformative frame costs both teams nothing. Ball unseen or in flight: 0.
  dead  players' mean speed (1 s rolling) below V0 says dead; the ball off the pitch says dead.
Viterbi then charges SWITCH_TEAM for H <-> A and SWITCH_DEAD for any alive <-> D change, which is
what turns frame evidence into whole possessions. Every parameter is tuned on IDSSE only.
"""
import numpy as np

from .data import FPS, HALF_LEN, HALF_WID, pm

PARAMS = {"SCALE": 1.0, "R_CLOSE": 3.0, "V0": 2.0, "VS": 0.5, "OUT": 3.0,
          "SWITCH_TEAM": 20.0, "SWITCH_DEAD": 20.0}
MAX_SPEED = 12.0  # m/s; faster is a slot handed to another player, not running


def _rolling_mean(x, n):
    """Centred NaN-aware mean over n frames."""
    ok = ~np.isnan(x)
    c = np.r_[0, np.cumsum(np.where(ok, x, 0))]
    k = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - n // 2), np.minimum(len(x), i + n // 2 + 1)
    cnt = k[hi] - k[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def features(half):
    ball, xy, team = half["ball"], half["xy"], half["team"]
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    with np.errstate(all="ignore"):
        d_home = np.fmin.reduce(np.where(team == 0, d, np.nan), axis=1)
        d_away = np.fmin.reduce(np.where(team == 1, d, np.nan), axis=1)
        v = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=2) * FPS
        v[v > MAX_SPEED] = np.nan
        speed = _rolling_mean(np.nanmean(v, axis=1), FPS)
    out = (np.abs(ball[:, 0]) > HALF_LEN + 1) | (np.abs(ball[:, 1]) > HALF_WID + 1)
    return {"d_home": d_home, "d_away": d_away, "speed": speed, "out": out}


def emissions(f, p):
    T = len(f["speed"])
    e = np.zeros((T, 3))
    with np.errstate(invalid="ignore"):
        close = np.fmin(f["d_home"], f["d_away"]) < p["R_CLOSE"]  # NaN (ball unseen) compares False
        both = np.isfinite(f["d_home"]) & np.isfinite(f["d_away"])
        gap = np.where(both, f["d_away"] - f["d_home"], np.where(np.isfinite(f["d_home"]), 5.0, -5.0))
    z = np.clip(gap / p["SCALE"], -20, 20)
    e[close, pm.H] = np.log(2.0) - np.logaddexp(0, -z[close])
    e[close, pm.A] = np.log(2.0) - np.logaddexp(0, z[close])
    s = np.nan_to_num(f["speed"], nan=p["V0"])
    dead = np.clip((p["V0"] - s) / p["VS"], -20, 20)  # log-odds of dead from the players' pace
    dead = dead + np.where(f["out"], p["OUT"], 0.0)
    e[:, pm.D] += np.log(2.0) - np.logaddexp(0, -dead)
    e[:, :pm.D] += (np.log(2.0) - np.logaddexp(0, dead))[:, None]
    return e


def viterbi(e, switch_team, switch_dead):
    """Max-score path over H, A, D. Pure Python on floats: 3 states is faster than numpy per step."""
    cost = [[0.0, switch_team, switch_dead], [switch_team, 0.0, switch_dead],
            [switch_dead, switch_dead, 0.0]]
    T = len(e)
    back = np.empty((T, 3), np.int8)
    score = list(e[0])
    rows = e.tolist()
    for t in range(1, T):
        new, r = [0.0, 0.0, 0.0], rows[t]
        for j in range(3):
            best, arg = score[0] - cost[0][j], 0
            for i in (1, 2):
                v = score[i] - cost[i][j]
                if v > best:
                    best, arg = v, i
            new[j] = best + r[j]
            back[t, j] = arg
        score = new
    path = np.empty(T, np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(half, params=None, f=None):
    p = {**PARAMS, **(params or {})}
    return viterbi(emissions(f if f is not None else features(half), p), p["SWITCH_TEAM"], p["SWITCH_DEAD"])
