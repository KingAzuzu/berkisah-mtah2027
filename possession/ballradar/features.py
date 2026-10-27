"""Half -> 5 Hz model inputs, and the cache of degraded IDSSE training halves (seeds 0..2).

A half (data.py representation) becomes, at every STRIDE-th frame:
    slot  (T5, S, 4) float32  x, y, vx, vy of every track slot, 0 where empty
    mask  (T5, S)    bool     slot has a position
    team  (S,)       int8     0 home, 1 away
    ball  (T5, 2)    float32  observed ball (0 where unseen)
    seen  (T5,)      bool     ball observed
and, for training/scoring only: truth5 (T5,) codes, ball_true (T5, 2) clean ball.
Velocity is a +-0.2 s central difference taken at 25 fps (one-sided if only one neighbour exists).
"""
import pickle

import numpy as np

from ..data import FPS, TUNE, idsse, load_idsse
from ..degrade import degrade_match

STRIDE = 5  # 25 fps -> 5 Hz
CACHE = idsse.DATA / "cache" / "ballradar"
SEEDS = (0, 1, 2)


def prep(half, with_truth=False, clean_ball=None):
    ball, xy, team = half["ball"], half["xy"], half["team"]
    T = len(ball)
    idx = np.arange(0, T, STRIDE)
    ip, im = np.minimum(idx + STRIDE, T - 1), np.maximum(idx - STRIDE, 0)
    x0, xp, xm = xy[idx], xy[ip], xy[im]
    okp, okm = ~np.isnan(xp[..., 0]), ~np.isnan(xm[..., 0])
    ok0 = ~np.isnan(x0[..., 0])
    with np.errstate(all="ignore"):
        v_both = (xp - xm) * FPS / np.maximum(ip - im, 1)[:, None, None]
        v_p = (xp - x0) * FPS / np.maximum(ip - idx, 1)[:, None, None]
        v_m = (x0 - xm) * FPS / np.maximum(idx - im, 1)[:, None, None]
    v = np.where((okp & okm)[..., None], v_both, np.where(okp[..., None], v_p, np.where(okm[..., None], v_m, 0.0)))
    v = np.where(ok0[..., None], np.nan_to_num(v), 0.0)
    slot = np.concatenate([np.nan_to_num(x0), v], axis=2).astype(np.float32)
    b = ball[idx]
    seen = ~np.isnan(b[:, 0])
    out = {"slot": slot, "mask": ok0, "team": team.astype(np.int8),
           "ball": np.nan_to_num(b).astype(np.float32), "seen": seen}
    if with_truth:
        out["truth5"] = half["truth"][idx].astype(np.int8)
        out["ball_true"] = np.nan_to_num(clean_ball[idx]).astype(np.float32)
        out["ball_true_ok"] = ~np.isnan(clean_ball[idx, 0])
    return out


def build(mid, seed):
    """Cached 5 Hz arrays for one tuning match and degrade seed: list of two halves."""
    assert mid in TUNE, "locked matches are never loaded"
    path = CACHE / f"{mid}_s{seed}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    clean = load_idsse(mid)
    deg = degrade_match(clean, mid, seed)
    out = [prep(d, True, c["ball"]) for d, c in zip(deg, clean)]
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(out, protocol=4))
    return out


if __name__ == "__main__":
    import time
    for m in TUNE:
        for s in SEEDS:
            t = time.time()
            r = build(m, s)
            print(m, s, [len(h["seen"]) for h in r], f"seen {np.mean(np.concatenate([h['seen'] for h in r])):.3f}",
                  f"{time.time() - t:.0f}s", flush=True)
