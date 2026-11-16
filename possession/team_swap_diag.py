"""Human only. Where do the team-label swaps happen, and do they start with a track jump (ID switch)? Aggregates only.

    idsse-tests\\.venv\\Scripts\\python -m possession.team_swap_diag

Context: team_label_check found one team > 11 in 18% of frames, 83% with the swap signature. IDSSE simulation (v19): random swaps
cost ~0.01 spellF1, swaps of the player nearest the ball cost ~0.03-0.05. This tells which case the real data is.
- teleport = a slot moves > 3 m between consecutive frames where it is present (75 m/s: impossible, so an ID switch);
- imbalance episode = run of frames with one team > 11 and the other < 11;
Prints: teleports, share of imbalance-episode starts within +-5 frames of a teleport (repairable by tracklet stitching),
share of imbalance frames where the ball (seen) is within 3 m of a player of the over-count team that teleported into the episode,
and the share of imbalance frames with the ball seen within 5 m of ANY over-count player.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "starter"))
import mtah_guard  # noqa: E402


def main():
    mtah_guard.refuse_under_claude()
    from possession import bar_ath
    ball, xy, team = bar_ath.load_atd()
    on = ~np.isnan(xy[..., 0])
    nh, na = (on & (team == 0)).sum(1), (on & (team == 1)).sum(1)
    imb = ((nh > 11) & (na < 11)) | ((na > 11) & (nh < 11))
    over = np.where(nh > 11, 0, np.where(na > 11, 1, -1))
    step = np.linalg.norm(np.diff(xy, axis=0), axis=2)
    tele = (step > 3) & on[1:] & on[:-1]
    tt, ts = np.nonzero(tele)
    tele_t = np.zeros(len(xy), bool); tele_t[tt + 1] = True
    starts = np.flatnonzero(imb[1:] & ~imb[:-1]) + 1
    near_tele = np.array([tele_t[max(s - 5, 0):s + 6].any() for s in starts]) if len(starts) else np.zeros(0)
    seen = ~np.isnan(ball[:, 0])
    d = np.linalg.norm(xy - ball[:, None, :], axis=2)
    d = np.where(on, d, np.inf)
    f = np.flatnonzero(imb & seen & (over >= 0))
    close_over = np.array([(d[t, team == over[t]] <= 5).any() for t in f]) if len(f) else np.zeros(0)
    nearest_over = np.array([team[np.argmin(d[t])] == over[t] for t in f]) if len(f) else np.zeros(0)
    print(f"teleports {len(tt)} ({len(tt) / len(xy) * 25 * 60:.2f} per min); imbalance frames {imb.mean():.4f}, episodes {len(starts)}, "
          f"median episode {np.median(np.diff(np.r_[starts, len(xy)])[:len(starts)]) / 25 if len(starts) else 0:.1f} s (upper bound)")
    print(f"episode starts within 5 frames of a teleport: {near_tele.mean() if len(starts) else float('nan'):.3f}")
    print(f"imbalance frames with ball seen: {len(f)}; over-count team has a player within 5 m of the ball {close_over.mean():.3f}; "
          f"nearest player to the ball is from the over-count team {nearest_over.mean():.3f}")
    bal = np.flatnonzero(~imb & seen & ((nh + na) >= 18))[::50]
    print(f"reference (balanced frames): a home player nearest the ball {np.mean([team[np.argmin(d[t])] == 0 for t in bal]):.3f}")


if __name__ == "__main__":
    main()
