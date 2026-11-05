"""SkillCorner open data (github.com/SkillCorner/opendata, MIT licence, (c) SkillCorner; credit SkillCorner).
Broadcast tracking 10 Hz, partial camera view. Other games only. Files live in
idsse-tests/data/external/skillcorner (gitignored).
Labels: H/A = team of the last non-null `possession.group` (held through null/loose frames, DFL-like); possession runs
shorter than MIN_RUN samples (0.2 s) are dropped (the previous team is held). D is NOT derivable reliably (possession is
null both in flight and when dead, ball-out is not flagged) so truth is 0/1, or -1 = unknown (before the first
possession): D is masked in training (team loss only).
Resampled 10 -> 25 fps (linear interpolation only between two observed samples, else NaN)."""
import json
from pathlib import Path
import numpy as np

from .data import idsse

SC = Path(idsse.DATA) / "external" / "skillcorner" / "repo" / "data" / "matches"
MIN_RUN = 2
HZ = 10


def match_ids():
    return sorted(p.name for p in SC.iterdir() if (p / f"{p.name}_tracking_extrapolated.jsonl").stat().st_size > 10_000_000)


def _interp(a, tq):
    """a (N,...) float with NaN at 10 Hz -> values at fractional sample times tq; NaN unless both neighbours observed."""
    i = np.minimum(np.floor(tq).astype(int), len(a) - 2)
    w = (tq - i).reshape((-1,) + (1,) * (a.ndim - 1))
    return a[i] * (1 - w) + a[i + 1] * w


def hold_labels(grp):
    """grp (N,) int8 in {-1,0,1} -> (N,) team held through nulls, runs < MIN_RUN dropped, -1 before first run."""
    idx = np.flatnonzero(grp >= 0)
    runs = []   # [start, team, n_samples]
    for i in idx:
        if runs and runs[-1][1] == grp[i]:
            runs[-1][2] += 1
        else:
            runs.append([i, int(grp[i]), 1])
    keep = []
    for r in runs:
        if r[2] < MIN_RUN:
            continue
        if keep and keep[-1][1] == r[1]:
            continue
        keep.append(r)
    out = np.full(len(grp), -1, np.int8)
    for j, r in enumerate(keep):
        out[r[0]:keep[j + 1][0] if j + 1 < len(keep) else len(grp)] = r[1]
    return out


def load_match(mid):
    d = SC / mid
    meta = json.load(open(d / f"{mid}_match.json", encoding="utf-8"))
    home = meta["home_team"]["id"]
    team_of = {p["id"]: (0 if p["team_id"] == home else 1) for p in meta["players"]}
    rows = [json.loads(l) for l in open(d / f"{mid}_tracking_extrapolated.jsonl", encoding="utf-8")]
    out = []
    for per in (1, 2):
        R = [r for r in rows if r["period"] == per]
        N = len(R)
        ball = np.full((N, 2), np.nan, np.float32)
        grp = np.full(N, -1, np.int8)
        pids = sorted({p["player_id"] for r in R for p in r["player_data"] if p["player_id"] in team_of})
        slot = {p: i for i, p in enumerate(pids)}
        xy = np.full((N, len(pids), 2), np.nan, np.float32)
        for k, r in enumerate(R):
            b = r["ball_data"]
            if b["is_detected"] and b["x"] is not None:
                ball[k] = (b["x"], b["y"])
            grp[k] = {"home team": 0, "away team": 1}.get(r["possession"]["group"], -1)
            for p in r["player_data"]:
                if p["is_detected"] and p["player_id"] in slot and p["x"] is not None:
                    xy[k, slot[p["player_id"]]] = (p["x"], p["y"])
        seen = np.isfinite(ball[:, 0])
        outside = seen & ((np.abs(ball[:, 0]) > 53.5) | (np.abs(ball[:, 1]) > 35))
        truth10 = hold_labels(grp)
        tq = np.arange(int((N - 1) * 25 / HZ)) * HZ / 25
        out.append({"ball": _interp(ball, tq).astype(np.float32), "xy": _interp(xy, tq).astype(np.float32),
                    "team": np.array([team_of[p] for p in pids], np.int8),
                    "truth": truth10[np.minimum(np.round(tq).astype(int), N - 1)],
                    "ball_out_share": float(outside.sum() / max(1, seen.sum())),
                    "null_share": float((grp < 0).mean())})
    return out


def stats(halves):
    lens = []
    for h in halves:
        x = h["truth"]
        b = np.flatnonzero(np.r_[True, x[1:] != x[:-1], True])
        lens += [(b[k + 1] - b[k]) / 25 for k in range(len(b) - 1) if x[b[k]] >= 0]
    cat = lambda f: np.concatenate([f(h) for h in halves])
    return {"known_share": round(float((cat(lambda h: h["truth"]) >= 0).mean()), 3), "mean_spell_s": round(float(np.mean(lens)), 2),
            "n_spells": len(lens), "ball_seen_25": round(float(cat(lambda h: np.isfinite(h["ball"][:, 0])).mean()), 3),
            "null_possession_share": round(float(np.mean([h["null_share"] for h in halves])), 3),
            "ball_outside_pitch_of_seen": round(float(np.mean([h["ball_out_share"] for h in halves])), 4),
            "players_per_frame": round(float(cat(lambda h: np.isfinite(h["xy"][..., 0]).sum(1)).mean()), 1)}


if __name__ == "__main__":
    import joblib
    out = {}
    for mid in match_ids():
        hv = load_match(mid)
        print(mid, [len(h["ball"]) for h in hv], stats(hv), flush=True)
        out[mid] = hv
    joblib.dump(out, SC.parent.parent.parent / "skillcorner_halves.pkl")
