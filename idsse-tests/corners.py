"""Corner kick frames: align event time to tracking and refine to the ball-status kick frame."""
import numpy as np

import idsse

WIN_S = 6  # search +-6 s around the event time for the dead->alive switch


def corner_kicks(m, ev):
    """Per corner: half, kick frame, attacking (taker) team, defending team.

    EventTime of a CornerKick is not the kick (the ball is often 10-30 m from the flag there). We take the
    ball-status switch dead -> alive (DFL BallStatus 0 -> 1) nearest to EventTime, within +-6 s, as the kick.
    """
    out, dropped = [], 0
    for c in ev["corners"]:
        r = idsse.to_frame(m, c["time"])
        h = m["halves"][r[0]] if r else None
        cand = []
        if r:
            lo, hi = max(1, r[1] - WIN_S * 25), min(h["T"], r[1] + WIN_S * 25)
            st = h["status"]
            cand = [j for j in range(lo, hi) if st[j - 1] == 0 and st[j] == 1]
        if not cand:
            dropped += 1
            continue
        k = min(cand, key=lambda j: abs(j - r[1]))
        b = h["ball"][min(k + 5, h["T"] - 1)]  # 0.2 s after kick the ball should still be near a flag
        dflag = float(np.hypot(abs(b[0]) - 52.5, abs(b[1]) - 34.0))
        out.append({"match": m["id"], "half": r[0], "kick": k, "off_s": (k - r[1]) / 25, "att": c["team"],
                    "def": idsse.other_team(m, c["team"]), "ball_flag_dist": dflag})
    return out, dropped
