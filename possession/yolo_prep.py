"""Human only. Stage A of the label-free YOLO check (+ a team-label sanity count). Submission venv, no new dependency.

    idsse-tests\\.venv\\Scripts\\python -m possession.yolo_prep

Writes data/stats/yolo_prep.npz (gitignored, local only): every 10th video frame with a valid homography -> frame index, H (pitch
0-1 -> pixel), ATD ball in metres (NaN if unseen), pitch size. Prints aggregates only:
- frames sampled, share with ATD ball seen;
- team-label check for the jersey idea: share of frames with more than 11 players labelled for one team (mislabel or sub overlap).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
import mtah_guard  # noqa: E402

OUT = ROOT / "data" / "stats" / "yolo_prep.npz"
STEP = 10


def main():
    mtah_guard.refuse_under_claude()
    from possession import bar_ath
    from games import GAME, GAMES
    from load_homography import load
    ball, xy, team = bar_ath.load_atd()
    H, _, valid = load()
    pitch = np.array(GAMES[GAME]["pitch"], np.float64)
    n = min(len(H), len(ball))
    ok = np.asarray(valid[:n], bool) & np.all(np.isfinite(H[:n].reshape(n, 9)), axis=1)
    idx = np.flatnonzero(ok)
    idx = idx[idx % STEP == 0]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, idx=idx, H=H[idx].astype(np.float64), ball=ball[idx].astype(np.float64), pitch=pitch)
    on = ~np.isnan(xy[..., 0])
    nh, na = (on & (team == 0)).sum(1), (on & (team == 1)).sum(1)
    live = (nh + na) >= 18
    print(f"sampled frames {len(idx)}; ATD ball seen {np.mean(~np.isnan(ball[idx, 0])):.3f}")
    print(f"team labels (frames with >= 18 players): >11 home {np.mean(nh[live] > 11):.4f}, >11 away {np.mean(na[live] > 11):.4f}, "
          f"<10 home {np.mean(nh[live] < 10):.4f}, <10 away {np.mean(na[live] < 10):.4f}, frames {int(live.sum())}")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
