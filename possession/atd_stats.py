"""Aggregate input statistics of the tactical ATD over graded frames. Run by a human in their own
terminal, never inside Claude Code.

    $env:MTAH_DATA_DIR = "D:\\mtah-data"
    idsse-tests\\.venv\\Scripts\\python -m possession.atd_stats

Prints and writes (data/stats/atd_stats.json, gitignored) aggregate numbers only: no per-frame rows,
no labels (none exist for the ATD). Paste back the printed block.

compute_stats() takes arrays and is tested on synthetic data in tests/test_diagnostics.py. Only
main() and load_inputs() touch the competition data, and both call the guard first.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
import mtah_guard  # noqa: E402

FPS = 25
PCTS = (10, 25, 50, 75, 90, 99)
FAST_MS = 40.0        # m/s: ball speeds above this are treated as likely detection jumps
OUTSIDE_M = 1.0       # metres beyond the touchline / goal line
STILL_WINDOW = 25     # frames (1 s)
STILL_MAX_M = 0.5     # net displacement over the window below which the ball counts as nearly stationary
OUT_JSON = ROOT / "data" / "stats" / "atd_stats.json"


def run_lengths(mask):
    """Lengths of the True runs of a 1-D boolean array."""
    p = np.r_[False, np.asarray(mask, bool), False].astype(np.int8)
    e = np.diff(p)
    return np.flatnonzero(e == -1) - np.flatnonzero(e == 1)


def summary(values, scale=1.0, pcts=PCTS):
    v = np.asarray(values, np.float64) * scale
    v = v[np.isfinite(v)]
    out = {"count": int(len(v))}
    if len(v):
        out["mean"] = round(float(v.mean()), 3)
        for q, x in zip(pcts, np.percentile(v, pcts)):
            out[f"p{q}"] = round(float(x), 3)
    return out


def compute_stats(ball, xy, team, blocks, pitch, fps=FPS):
    """ball (T,2), xy (T,S,2) in metres centred on the pitch, team (S,) 0/1, blocks [(half, first, last)]."""
    seen_runs, miss_runs, tracklets, speeds, still_speeds = [], [], [], [], []
    n_seen = n_frames = n_out = n_fast = 0
    objs = {"all": [], "home": [], "away": []}
    ever = np.zeros(xy.shape[1], bool)
    frames_active = np.zeros(xy.shape[1], np.int64)
    half_l, half_w = pitch[0] / 2, pitch[1] / 2
    for _, a, b in blocks:
        bl, p = ball[a:b + 1], xy[a:b + 1]
        seen = ~np.isnan(bl[:, 0])
        n_frames += len(bl)
        n_seen += int(seen.sum())
        seen_runs.append(run_lengths(seen))
        miss_runs.append(run_lengths(~seen))
        pres = ~np.isnan(p[:, :, 0])
        for k, name in ((None, "all"), (0, "home"), (1, "away")):
            cols = pres if k is None else pres[:, team == k]
            objs[name].append(cols.sum(axis=1))
        ever |= pres.any(axis=0)
        frames_active += pres.sum(axis=0)
        for s in range(p.shape[1]):
            if pres[:, s].any():
                tracklets.append(run_lengths(pres[:, s]))
        sb = bl[seen]
        n_out += int(np.sum((np.abs(sb[:, 0]) > half_l + OUTSIDE_M) | (np.abs(sb[:, 1]) > half_w + OUTSIDE_M)))
        step = np.linalg.norm(bl[1:] - bl[:-1], axis=1)  # NaN unless both frames seen
        v = step * fps
        speeds.append(v[np.isfinite(v)])
        n_fast += int(np.sum(v[np.isfinite(v)] > FAST_MS))
        if len(bl) > STILL_WINDOW:
            net = np.linalg.norm(bl[STILL_WINDOW:] - bl[:-STILL_WINDOW], axis=1)
            still = net < STILL_MAX_M  # NaN compares False
            vv = v[:len(still)]
            still_speeds.append(vv[still & np.isfinite(vv)])
    cat = lambda xs: np.concatenate(xs) if xs else np.zeros(0)  # noqa: E731
    speeds, still_speeds = cat(speeds), cat(still_speeds)
    return {
        "graded_frames": n_frames,
        "ball_seen_fraction": round(n_seen / max(n_frames, 1), 4),
        "ball_seen_run_s": summary(cat(seen_runs), 1 / fps),
        "ball_missing_run_s": summary(cat(miss_runs), 1 / fps),
        "objects_per_frame": {k: summary(cat(v)) for k, v in objs.items()},
        "slots_total": int(xy.shape[1]),
        "slots_ever_active": int(ever.sum()),
        "slots_active_over_1pct_of_frames": int(np.sum(frames_active > 0.01 * max(n_frames, 1))),
        "slot_tracklet_s": summary(cat(tracklets), 1 / fps),
        "ball_step_m_both_frames_seen": summary(speeds / fps, pcts=PCTS + (99.9,)),
        "ball_speed_ms_both_frames_seen": summary(speeds),
        "share_speed_over_40ms": round(n_fast / max(len(speeds), 1), 5),
        "share_seen_outside_pitch_1m": round(n_out / max(n_seen, 1), 5),
        "ball_speed_ms_nearly_stationary": summary(still_speeds, pcts=PCTS + (99.9,)),
        "assumptions": {"fast_ms": FAST_MS, "outside_m": OUTSIDE_M,
                        "still": f"net move < {STILL_MAX_M} m over {STILL_WINDOW} frames, both ends seen"},
    }


def _line(label, s, unit=""):
    keys = ["count", "mean"] + [k for k in s if k.startswith("p")]
    return f"{label:<28}" + "  ".join(f"{k} {s[k]}" for k in keys if k in s) + (f"  [{unit}]" if unit else "")


def format_report(st):
    o = st["objects_per_frame"]
    lines = [
        "ATD input statistics over graded frames (aggregates only)",
        f"graded frames {st['graded_frames']}   ball seen fraction {st['ball_seen_fraction']}",
        _line("ball seen runs", st["ball_seen_run_s"], "s"),
        _line("ball missing runs", st["ball_missing_run_s"], "s"),
        _line("objects/frame all", o["all"]),
        _line("objects/frame home", o["home"]),
        _line("objects/frame away", o["away"]),
        f"slots total {st['slots_total']}  ever active {st['slots_ever_active']}  "
        f"active >1% of frames {st['slots_active_over_1pct_of_frames']}",
        _line("slot tracklets", st["slot_tracklet_s"], "s"),
        _line("ball step both seen", st["ball_step_m_both_frames_seen"], "m/frame"),
        _line("ball speed both seen", st["ball_speed_ms_both_frames_seen"], "m/s"),
        f"share speed > {FAST_MS:g} m/s {st['share_speed_over_40ms']}",
        f"share seen outside pitch by > {OUTSIDE_M:g} m {st['share_seen_outside_pitch_1m']}",
        _line("speed, nearly stationary", st["ball_speed_ms_nearly_stationary"], "m/s"),
    ]
    return "\n".join(lines)


def load_inputs():
    mtah_guard.refuse_under_claude()
    from possession import bar_ath
    from games import GAME, half_video_blocks, GAMES
    ball, xy, team = bar_ath.load_atd()
    return ball, xy, team, half_video_blocks(GAME), GAMES[GAME]["pitch"]


def main():
    mtah_guard.refuse_under_claude()
    st = compute_stats(*load_inputs())
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(st, indent=1))
    print(format_report(st))
    print(f"json: {OUT_JSON}")


if __name__ == "__main__":
    main()
