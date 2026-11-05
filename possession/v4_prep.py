"""v4 data prep (IDSSE TUNE, SkillCorner, Metrica only). python -m possession.v4_prep dr|sc
dr: K domain-randomised matched2 degrade variants of every TUNE match (constants drawn from ranges around the profile,
    fixed seeds), features cached; labels do not depend on the variant. Variant 0 = canonical matched2 (existing cache).
sc: SkillCorner features (no extra degrade: broadcast tracking is already partial; ball seen share is real)."""
import sys
import numpy as np
import joblib

from . import degrade_matched as M, degrade_matched2 as DM2, tcn_lib as T
from .data import TUNE, idsse, load_idsse

K = 6
CACHE = idsse.DATA / "cache"
RANGES = {  # (module, name, lo, hi): around the matched2 profile (BALL_SEEN .593, miss med 4 s, tracklet 35 s, gap 3.5 s, jitter 1 cm, P_DEAD .63)
    (M, "BALL_SEEN", 0.48, 0.70), (DM2, "M_MISS_MED_S", 2.5, 6.0), (M, "TRK_MED_S", 20.0, 60.0),
    (M, "GAP_MED_S", 2.0, 6.0), (M, "BALL_JITTER_M", 0.005, 0.04), (DM2, "P_DEAD", 0.45, 0.80)}
RANGES = sorted(RANGES, key=lambda r: r[1])


def variant(k):
    rng = np.random.default_rng([4040, k])
    return {name: (mod, float(rng.uniform(lo, hi))) for mod, name, lo, hi in RANGES}


def degrade_variant(halves, mid, k):
    v = variant(k)
    old = {n: getattr(m, n) for n, (m, _) in v.items()}
    try:
        for n, (m, x) in v.items():
            setattr(m, n, x)
        return DM2.degrade_match(halves, mid, seed=1000 + k)
    finally:
        for n, (m, _) in v.items():
            setattr(m, n, old[n])


def prog(msg):
    print(msg, flush=True)


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "dr":
        out = {}
        for m in TUNE:
            hv = load_idsse(m)
            out[m] = [[T.frame_features(h) for h in degrade_variant(hv, m, k)] for k in range(K)]
            prog(f"dr {m} done; seen share v0..: " + str([round(float(np.mean([np.isfinite(h['ball'][:, 0]).mean() for h in degrade_variant(hv, m, k)])), 2) for k in range(2)]))
        joblib.dump(out, CACHE / "tcn_feats_dr.pkl")
    else:
        from . import scdata as S
        out = {}
        for mid in S.match_ids():
            hv = S.load_match(mid)
            prog(f"sc {mid} {S.stats(hv)}")
            out[mid] = ([T.frame_features(h) for h in hv], [h["truth"][:len(h["truth"]) // T.SUB * T.SUB].reshape(-1, T.SUB)[:, 2].astype(np.int64) for h in hv])
        joblib.dump(out, idsse.DATA / "external" / "tcn_feats_sc.pkl")
