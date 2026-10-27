"""WS4: dead-ball classifier. Per-frame features from a degraded half, a gradient-boosted p(dead), and
emissions that use alpha * logit(p_dead) instead of the speed-based D log-odds in safety.emissions.

Features use only what BAR-ATH tracking gives: anonymous team-labelled slots, a ball that is NaN when unseen,
no known attacking direction. So geometry is direction-invariant (|x|, |y|, products of the two team sides).
Windows of 1 / 3 / 10 s, past [t-n+1, t] and future [t, t+n-1]; look-ahead is allowed (#15).

Nothing here loads competition data. Inference is CPU, HistGradientBoosting random_state=0.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import numpy as np  # noqa: E402

from . import safety  # noqa: E402
from .data import FPS, HALF_LEN, HALF_WID, pm  # noqa: E402

MAX_SPEED = safety.MAX_SPEED
WINDOWS_S = (1, 3, 10)
CENTRE_R, CORNER_R, LINE_M = 9.15, 4.0, 2.0
BOX_X, BOX_Y = 16.5, 20.16  # penalty area: depth, half width


def _csum(x):
    ok = ~np.isnan(x)
    return np.r_[0, np.cumsum(np.where(ok, x, 0.0))], np.r_[0, np.cumsum(ok)]


def window_mean(x, n, future):
    """NaN-aware mean of x over [t-n+1, t] (past) or [t, t+n-1] (future)."""
    T = len(x)
    c, k = _csum(x)
    i = np.arange(T)
    lo, hi = (i, np.minimum(T, i + n)) if future else (np.maximum(0, i - n + 1), i + 1)
    cnt = k[hi] - k[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _since_seen(seen):
    """Frames since the last True (past), and until the next True (future); capped at 30 s."""
    T = len(seen)
    cap = 30 * FPS
    idx = np.arange(T)
    last = np.maximum.accumulate(np.where(seen, idx, -10 ** 9))
    nxt = np.minimum.accumulate(np.where(seen, idx, 10 ** 9)[::-1])[::-1]
    return np.minimum(idx - last, cap).astype(float), np.minimum(nxt - idx, cap).astype(float)


def _nan_count(mask, valid):
    return np.where(valid, mask.sum(axis=1), np.nan)


def base_series(half):
    """Dict name -> (T,) float of instantaneous features, before windowing."""
    ball, xy, team = half["ball"], half["xy"], half["team"]
    T, S, _ = xy.shape
    s = {}
    with np.errstate(all="ignore"):
        v = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=2) * FPS
        v[v > MAX_SPEED] = np.nan
        seen = ~np.isnan(xy[:, :, 0])
        s["n_players"] = seen.sum(axis=1).astype(float)
        vv = ~np.isnan(v)
        nv = vv.sum(axis=1)
        s["spd_mean"] = np.nanmean(v, axis=1)
        s["spd_med"] = np.nanmedian(v, axis=1)
        s["spd_q75"] = np.nanquantile(v, 0.75, axis=1)
        s["spd_q90"] = np.nanquantile(v, 0.90, axis=1)
        s["spd_lt1"] = np.where(nv > 0, np.sum(v < 1.0, axis=1) / np.maximum(nv, 1), np.nan)
        s["spd_lt2"] = np.where(nv > 0, np.sum(v < 2.0, axis=1) / np.maximum(nv, 1), np.nan)
        s["spd_gt4"] = np.where(nv > 0, np.sum(v > 4.0, axis=1) / np.maximum(nv, 1), np.nan)
        ax, ay = np.abs(xy[:, :, 0]), np.abs(xy[:, :, 1])
        anyp = s["n_players"] > 0
        s["near_line"] = _nan_count(ay > HALF_WID - LINE_M, anyp)
        cd = np.minimum(np.hypot(HALF_LEN - ax, HALF_WID - ay), 1e9)
        s["near_corner"] = _nan_count(cd < CORNER_R, anyp)
        s["near_corner_wide"] = _nan_count(cd < 10.0, anyp)
        s["in_goal_area"] = _nan_count((ax > HALF_LEN - 5.5) & (ay < 9.16), anyp)
        s["in_box"] = _nan_count((ax > HALF_LEN - BOX_X) & (ay < BOX_Y), anyp)
        s["in_centre"] = _nan_count(np.hypot(xy[:, :, 0], xy[:, :, 1]) < CENTRE_R, anyp)
        s["in_centre_big"] = _nan_count(np.hypot(xy[:, :, 0], xy[:, :, 1]) < 2 * CENTRE_R, anyp)
        s["mean_abs_x"] = np.nanmean(ax, axis=1)
        s["mean_abs_y"] = np.nanmean(ay, axis=1)
        for k, nm in ((0, "h"), (1, "a")):
            m = team == k
            x, y = xy[:, m, 0], xy[:, m, 1]
            ns = (~np.isnan(x)).sum(axis=1)
            s[f"n_{nm}"] = ns.astype(float)
            s[f"cx_{nm}"] = np.nanmean(x, axis=1)
            s[f"sx_{nm}"] = np.nanstd(x, axis=1)
            s[f"sy_{nm}"] = np.nanstd(y, axis=1)
            s[f"side_{nm}"] = np.abs(np.nanmean(np.sign(x), axis=1))  # 1 = whole team on one side
            s[f"in_c_{nm}"] = np.where(ns > 0, np.sum(np.hypot(x, y) < CENTRE_R, axis=1), np.nan)
        # kickoff: teams on opposite sides, no one in the opposition half of the centre circle is not knowable,
        # so use the product of side signs: -1 = separated, +1 = same side
        s["cx_prod"] = np.sign(s["cx_h"]) * np.sign(s["cx_a"])
        s["cx_gap"] = np.abs(s["cx_h"] - s["cx_a"])
        s["spread_all"] = np.nanstd(xy[:, :, 0], axis=1) + np.nanstd(xy[:, :, 1], axis=1)

        # ball
        bx, by = ball[:, 0], ball[:, 1]
        bseen = ~np.isnan(bx)
        s["b_seen"] = bseen.astype(float)
        s["b_abs_x"], s["b_abs_y"] = np.abs(bx), np.abs(by)
        s["b_out"] = np.where(bseen, ((np.abs(bx) > HALF_LEN + 1) | (np.abs(by) > HALF_WID + 1)).astype(float), np.nan)
        s["b_line_d"] = np.where(bseen, np.minimum(HALF_WID - np.abs(by), HALF_LEN - np.abs(bx)), np.nan)
        s["b_centre_d"] = np.hypot(bx, by)
        s["b_corner_d"] = np.hypot(HALF_LEN - np.abs(bx), HALF_WID - np.abs(by))
        # ball speed over the gap between consecutive sightings (m/s), only when the gap is short
        idx = np.arange(T)
        last = np.maximum.accumulate(np.where(bseen, idx, -1))
        prev = np.r_[-1, last[:-1]]
        gap = idx - prev
        okb = bseen & (prev >= 0) & (gap <= 5)
        px = np.where(prev >= 0, bx[np.maximum(prev, 0)], np.nan)
        py = np.where(prev >= 0, by[np.maximum(prev, 0)], np.nan)
        sp = np.hypot(bx - px, by - py) * FPS / np.maximum(gap, 1)
        s["b_speed"] = np.where(okb, np.minimum(sp, 50.0), np.nan)
        d = np.linalg.norm(xy - ball[:, None, :], axis=2)
        s["d_nearest"] = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
        s["d_nearest"][~np.isfinite(s["d_nearest"])] = np.nan
        s["n_near5"] = np.where(bseen, np.sum(d < 5.0, axis=1), np.nan)
        s["n_near1_5"] = np.where(bseen, np.sum(d < 1.5, axis=1), np.nan)
        # nearest player of each team: gap reveals a clear possessor
        dh = np.fmin.reduce(np.where(team == 0, d, np.nan), axis=1)
        da = np.fmin.reduce(np.where(team == 1, d, np.nan), axis=1)
        s["d_gap"] = np.abs(dh - da)
        s["d_min_team"] = np.fmin(dh, da)
    return s


# series that get rolling windows (everything above except pure slow-moving counts is fine to window)
WINDOWED = ["n_players", "spd_mean", "spd_med", "spd_q75", "spd_lt1", "spd_lt2", "spd_gt4", "near_line",
            "near_corner", "near_corner_wide", "in_goal_area", "in_box", "in_centre", "in_centre_big",
            "side_h", "side_a", "cx_prod", "cx_gap", "spread_all", "in_c_h", "in_c_a",
            "b_seen", "b_out", "b_line_d", "b_centre_d", "b_corner_d", "b_speed", "d_nearest", "n_near5",
            "n_near1_5", "d_gap", "d_min_team"]


def build_features(half):
    """(T, F) float32 feature matrix and the list of feature names."""
    s = base_series(half)
    cols = dict(s)
    for name in WINDOWED:
        x = s[name]
        for w in WINDOWS_S:
            n = w * FPS
            cols[f"{name}_p{w}"] = window_mean(x, n, False)
            cols[f"{name}_f{w}"] = window_mean(x, n, True)
    # speed trend: change in pace between future and past 3 s (a stop or restart shows as a step)
    cols["spd_step3"] = cols["spd_mean_f3"] - cols["spd_mean_p3"]
    cols["spd_step10"] = cols["spd_mean_f10"] - cols["spd_mean_p10"]
    since, until = _since_seen(~np.isnan(half["ball"][:, 0]))
    cols["b_since_seen"], cols["b_until_seen"] = since, until
    names = sorted(cols)
    X = np.stack([cols[k] for k in names], axis=1).astype(np.float32)
    return X, names


def labels(half):
    return (half["truth"] == pm.D).astype(np.int8)


def make_model(max_iter=200):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=15,
                                          min_samples_leaf=100, l2_regularization=1.0,
                                          early_stopping=False, random_state=0)


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def emissions(f, p_dead, alpha, p):
    """Team evidence exactly as safety.emissions, D log-odds replaced by alpha * logit(p_dead)."""
    T = len(p_dead)
    quiet = {**f, "speed": np.full(T, np.nan), "out": np.zeros(T, bool)}  # dead term -> exactly 0
    e = safety.emissions(quiet, p)
    dead = np.clip(alpha * logit(p_dead), -20, 20)
    e[:, pm.D] += np.log(2.0) - np.logaddexp(0, -dead)
    e[:, :pm.D] += (np.log(2.0) - np.logaddexp(0, dead))[:, None]
    return e
