"""HANDOVER section 6, idea 1: is the tactical camera's screen centre a usable estimate of the ball?
Run by a human in their own terminal, never inside Claude Code.

    $env:MTAH_DATA_DIR = "D:\\mtah-data"
    idsse-tests\\.venv\\Scripts\\python -m possession.camera_centre

For each graded frame with a valid homography and a seen ATD ball: map pixel (960, 540) through the
inverse homography to normalised pitch coordinates, convert to metres, take the distance to the ATD
ball. Prints aggregates only and writes them to data/stats/camera_centre.json (gitignored).

Pre-set bar (HANDOVER section 6, fixed before seeing any result): median distance < 10 m means use it.
Not changed here.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
import mtah_guard  # noqa: E402

BAR_M = 10.0                 # HANDOVER idea 1: use it if the median is below this
CENTRE_PX = (960.0, 540.0)   # 1920 x 1080 video
OUT_JSON = ROOT / "data" / "stats" / "camera_centre.json"


def centre_distances(H, valid, ball, blocks, pitch):
    """Return (distance_m, ball_x_m) for graded frames with valid homography and seen ball.

    H (T,3,3) maps normalised pitch (0-1) to pixels; ball (T,2) is in metres centred on the pitch.
    """
    size = np.array(pitch, np.float64)
    graded = np.zeros(len(ball), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    n = min(len(H), len(ball))
    use = graded[:n] & np.asarray(valid[:n], bool) & ~np.isnan(ball[:n, 0])
    use &= np.all(np.isfinite(H[:n].reshape(n, 9)), axis=1)
    idx = np.flatnonzero(use)
    Hu = H[idx]
    ok = np.abs(np.linalg.det(Hu)) > 1e-12
    idx, Hu = idx[ok], Hu[ok]
    p = np.linalg.solve(Hu, np.tile([CENTRE_PX[0], CENTRE_PX[1], 1.0], (len(idx), 1))[:, :, None])[:, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        nxy = p[:, :2] / p[:, 2:3]
    centre_m = nxy * size - size / 2
    good = np.all(np.isfinite(centre_m), axis=1)
    centre_m, idx = centre_m[good], idx[good]
    b = ball[idx].astype(np.float64)
    return np.linalg.norm(centre_m - b, axis=1), b[:, 0]


def summarise(dist, ball_x, pitch):
    third = pitch[0] / 6
    parts = {"left": ball_x < -third, "middle": np.abs(ball_x) <= third, "right": ball_x > third}
    out = {"frames_used": int(len(dist))}
    if len(dist):
        out["median_m"] = round(float(np.median(dist)), 2)
        out["mean_m"] = round(float(dist.mean()), 2)
        for q in (25, 75, 90):
            out[f"p{q}_m"] = round(float(np.percentile(dist, q)), 2)
        out["median_by_ball_third_m"] = {
            k: (round(float(np.median(dist[m])), 2) if m.any() else None) for k, m in parts.items()}
        out["frames_by_ball_third"] = {k: int(m.sum()) for k, m in parts.items()}
        out["pass"] = bool(out["median_m"] < BAR_M)
    out["bar_median_m"] = BAR_M
    return out


def format_report(st):
    t = st.get("median_by_ball_third_m", {})
    lines = [
        "Camera centre vs ATD ball (graded frames, valid homography, ball seen)",
        f"frames used {st['frames_used']}",
    ]
    if st["frames_used"]:
        lines += [
            f"distance m: median {st['median_m']}  mean {st['mean_m']}  "
            f"p25 {st['p25_m']}  p75 {st['p75_m']}  p90 {st['p90_m']}",
            f"median m by ball third: left {t['left']}  middle {t['middle']}  right {t['right']}",
            f"pre-set bar (HANDOVER idea 1): median < {BAR_M:g} m  ->  {'PASS' if st['pass'] else 'FAIL'}",
        ]
    else:
        lines.append(f"no usable frames; bar median < {BAR_M:g} m  ->  FAIL")
    return "\n".join(lines)


def load_inputs():
    mtah_guard.refuse_under_claude()
    from possession import bar_ath
    from games import GAME, GAMES, half_video_blocks
    from load_homography import load
    ball, _, _ = bar_ath.load_atd()
    H, _, valid = load()
    return H, valid, ball, half_video_blocks(GAME), GAMES[GAME]["pitch"]


def main():
    mtah_guard.refuse_under_claude()
    H, valid, ball, blocks, pitch = load_inputs()
    st = summarise(*centre_distances(H, valid, ball, blocks, pitch), pitch)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(st, indent=1))
    print(format_report(st))
    print(f"json: {OUT_JSON}")


if __name__ == "__main__":
    main()
