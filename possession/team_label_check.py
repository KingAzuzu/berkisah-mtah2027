"""Human only. Why do frames have > 11 players of one team? Mislabel (swap) vs duplicate tracklets. Aggregates only.

    idsse-tests\\.venv\\Scripts\\python -m possession.team_label_check

Over frames with >= 18 players:
- swap signature: one team > 11 while the other < 11 in the same frame;
- duplicate signature: two same-team players within 0.5 m of each other;
- share of player-frames in tracklets whose label disagrees with the team that tracklet's position side suggests at kickoff is not
  computed (needs orientation); instead: tracklet count per team and median tracklet length.
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
    _, xy, team = bar_ath.load_atd()
    on = ~np.isnan(xy[..., 0])
    nh, na = (on & (team == 0)).sum(1), (on & (team == 1)).sum(1)
    live = (nh + na) >= 18
    swap = ((nh > 11) & (na < 11)) | ((na > 11) & (nh < 11))
    over = (nh > 11) | (na > 11)
    dup = np.zeros(len(xy), bool)
    for t in np.flatnonzero(live & over):
        for k in (0, 1):
            p = xy[t, on[t] & (team == k)]
            if len(p) > 1:
                d = np.linalg.norm(p[:, None] - p[None], axis=2)
                if (d[np.triu_indices(len(p), 1)] < 0.5).any():
                    dup[t] = True
    print(f"frames >= 18 players {int(live.sum())}; one team > 11: {over[live].mean():.4f}")
    print(f"  of those: swap signature (other team < 11) {swap[live & over].mean():.3f}; duplicate within 0.5 m {dup[live & over].mean():.3f}")
    print(f"  players on pitch per frame: median total {np.median((nh + na)[live]):.0f}, total > 22 {np.mean((nh + na)[live] > 22):.4f}")
    L = on.sum(0)
    for k, nm in ((0, "home"), (1, "away")):
        s = team == k
        print(f"  {nm}: tracklet slots {int(s.sum())}, median frames per slot {np.median(L[s]):.0f}")


if __name__ == "__main__":
    main()
