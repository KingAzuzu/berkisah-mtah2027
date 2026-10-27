"""Test A: corner-marking style from identity-free tracking features (see PREREGISTRATION.md)."""
import json
import os

for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # tiny problems: BLAS threads only thrash
    os.environ[v] = "1"
from multiprocessing import Pool

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, spearmanr

import idsse
from corners import corner_kicks

WIN, LAG, FPS = 75, 5, 25  # 3 s window at full 25 fps; speed = displacement over 5 frames (0.2 s)
N_REP = 20
BOX_X, BOX_Y = 52.5 - 16.5 - 2.0, 20.16 + 2.0  # penalty area + 2 m margin, attackers' frame (goal at +x)


def in_box(p):
    return (p[..., 0] >= BOX_X) & (np.abs(p[..., 1]) <= BOX_Y)


def dists(a, b):
    """(n, 2), (m, 2) -> (n, m) euclidean distances."""
    return np.hypot(*(a[:, None] - b[None]).transpose(2, 0, 1))


def speeds(p):
    """(T, P, 2) positions -> (T, P) speed from 5-frame backward differences (per slot), NaN for first LAG frames."""
    v = np.full(p.shape[:2], np.nan)
    v[LAG:] = np.hypot(*(p[LAG:] - p[:-LAG]).transpose(2, 0, 1)) / (LAG / FPS)
    return v


def reference(D, A):
    """Stickiness with clean IDs. D: defending outfield (WIN+1, P, 2), A: attacking outfield; last frame = kick."""
    out = []
    for j in np.where(in_box(D[-1]))[0]:
        d = np.stack([np.hypot(*(D[t, j] - A[t]).T) for t in range(WIN)])  # (WIN, Pa)
        d[np.isnan(d)] = np.inf
        ok = np.isfinite(d.min(1))
        if ok.sum() < WIN // 2:
            continue
        out.append(np.bincount(d[ok].argmin(1)).max() / ok.sum())  # share of frames with the modal nearest attacker
    return np.mean(out) if out else np.nan


def degrade(P, f0, rng):
    """(T, players, 2) from absolute frame f0: permute player slots every 2 s, then drop 10% of positions."""
    f = np.arange(f0, f0 + len(P))
    out = np.empty_like(P)
    for b in np.unique(f // 50):
        sel = f // 50 == b
        out[sel] = P[sel][:, rng.permutation(P.shape[1])]
    out[rng.random(out.shape[:2]) < 0.10] = np.nan
    return out


def identity_free(D, A):
    """D, A: (WIN+LAG, P, 2), all players of each team (GK included, no roles). Returns (speed ratio, pair distance).
    Speeds are differenced per slot, so on shuffled data they glitch at the shuffle points."""
    vD, vA = speeds(D)[LAG:], speeds(A)[LAG:]
    D, A = D[LAG:], A[LAG:]
    bD, bA = in_box(D), in_box(A)
    sd, sa = vD[bD], vA[bA]
    ratio = np.nanmean(sd) / np.nanmean(sa) if np.isfinite(sd).any() and np.isfinite(sa).any() else np.nan
    pd = []
    for t in range(WIN):
        d, a = D[t][bD[t]], A[t][bA[t]]
        if len(d) and len(a):
            c = dists(d, a)
            r, cc = linear_sum_assignment(c)
            pd.append(c[r, cc].mean())
    return ratio, (np.mean(pd) if pd else np.nan)


def one_match(mid):
    m = idsse.load_match(mid)
    cs, dropped = corner_kicks(m, idsse.load_events(mid))
    rows = []
    for ci, c in enumerate(cs):
        h, k = m["halves"][c["half"]], c["kick"]
        if k - WIN - LAG < 0:
            continue
        sl = slice(k - WIN - LAG, k + 1)
        D = idsse.team_xy(h, c["def"], c["att"])[sl].astype(float)  # frame where the attackers attack +x
        A = idsse.team_xy(h, c["att"], c["att"])[sl].astype(float)
        gkD, gkA = h["teams"][c["def"]]["gk"], h["teams"][c["att"]]["gk"]
        row = {**c, "ref": reference(D[LAG:][:, ~gkD], A[LAG:][:, ~gkA]), "clean": identity_free(D[:-1], A[:-1]), "deg": []}
        for rep in range(N_REP):
            rng = np.random.default_rng([rep, ci, int(mid[-2:], 36)])
            row["deg"].append(identity_free(degrade(D[:-1], k - WIN - LAG, rng), degrade(A[:-1], k - WIN - LAG, rng)))
        rows.append(row)
    return rows, dropped


def zidx(ratio, dist):
    z = lambda x: (x - np.mean(x)) / np.std(x)
    return z(ratio) - z(dist)


def main():
    with Pool(7) as pool:
        res = pool.map(one_match, list(idsse.MATCHES))
    rows = [r for rs, _ in res for r in rs]
    ok = [r for r in rows if np.isfinite(r["ref"]) and np.isfinite(r["clean"]).all()]
    ref = np.array([r["ref"] for r in ok])
    cl = np.array([r["clean"] for r in ok])
    idx = zidx(cl[:, 0], cl[:, 1])
    rho1 = spearmanr(idx, ref)
    out = {"n_corner_events": sum(len(rs) + d for rs, d in res),
           "n_dropped_no_kick_frame": {mid: d for mid, (_, d) in zip(idsse.MATCHES, res)},
           "n_valid": len(ok), "per_match": {mid: sum(r["match"] == mid for r in ok) for mid in idsse.MATCHES},
           "kick_offset_from_eventtime_s_median": float(np.median([r["off_s"] for r in rows])),
           "ball_to_flag_0.2s_after_kick_median_m": float(np.median([r["ball_flag_dist"] for r in rows])),
           "A1_rho": rho1.statistic, "A1_p": rho1.pvalue,
           "A1_rho_speed_ratio_only": spearmanr(cl[:, 0], ref).statistic,
           "A1_rho_neg_pair_distance_only": spearmanr(-cl[:, 1], ref).statistic,
           "A1_rho_exploratory_log_speed_ratio": spearmanr(zidx(np.log(np.clip(cl[:, 0], 1e-3, None)), cl[:, 1]), ref).statistic,
           "ref_mean": ref.mean(), "ref_sd": ref.std()}
    rhos = []
    for rep in range(N_REP):
        dg = np.array([r["deg"][rep] for r in ok])
        g = np.isfinite(dg).all(1)
        rhos.append(spearmanr(zidx(dg[g, 0], dg[g, 1]), ref[g]).statistic)
    out.update({"A3_rho_primary_rep0": rhos[0], "A3_rho_reps": rhos, "A3_rho_mean": np.mean(rhos), "A3_rho_sd": np.std(rhos)})
    teams = {}
    for r, i in zip(ok, idx):
        teams.setdefault(r["def"], []).append(i)
    big = {t: v for t, v in teams.items() if len(v) >= 5}
    out["A2_corners_by_defending_team"] = {t: len(v) for t, v in teams.items()}
    out["A2_teams_used_mean_index"] = {t: float(np.mean(v)) for t, v in big.items()}
    if len(big) >= 2:
        kw = kruskal(*big.values())
        out["A2_H"], out["A2_p"] = kw.statistic, kw.pvalue
    json.dump(out, open(idsse.ROOT / "results_a.json", "w"), indent=1, default=float)
    np.savez(idsse.ROOT / "a_points.npz", idx=idx, ref=ref, team=[r["def"] for r in ok])
    print(json.dumps({k: v for k, v in out.items() if k != "A3_rho_reps"}, indent=1, default=float))


if __name__ == "__main__":
    main()
