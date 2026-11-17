"""Real-tracking ball statistics for the matched3 simulator. HUMAN ONLY, own terminal (never inside Claude Code):

    $env:MTAH_DATA_DIR = "D:\\mtah-data"
    idsse-tests\\.venv\\Scripts\\python -m possession.atd_stats3

Label-free: 'dead' is v2's own prediction (rules + players-only dead model, frozen, never reads ball visibility), the same proxy as
ball_loss_check. Prints aggregates only (shares, run-length quantiles, speed / acceleration percentiles), the same quantities
v21_simstats computes on IDSSE, so the two can be compared line by line. Paste back the printed block, nothing else."""
import numpy as np

from . import pipeline, v21_simstats as S
from .bar_ath import load_atd, mtah_guard
from .data import FPS, pm


def main():
    mtah_guard.refuse_under_claude()
    from games import GAME, half_video_blocks
    ball, xy, team = load_atd()
    p, models = pipeline.load_params(), pipeline.load_models()
    halves = []
    for _, a, b in half_video_blocks(GAME):
        h = {"ball": ball[a:b + 1], "xy": xy[a:b + 1], "team": team}
        halves.append({"ball": h["ball"], "truth": pipeline.predict(h, p, models)})
    st = S.ball_stats(halves)
    # missing-run histogram (seconds): is there a hole at 1-5 s (tracker fills short gaps)?
    mr = np.array([(e - s) / FPS for h in halves for s, e in S.M3.runs(np.isnan(h["ball"][:, 0]))])
    edges = (0, 0.2, 0.5, 1, 2, 3, 5, 10, 20, 60, 1e9)
    hist = np.histogram(mr, bins=edges)[0]
    # within stoppages (v2 D runs >= 3 s): share hidden end to end / never hidden / mixed
    whole, never, mixed = 0, 0, 0
    for h in halves:
        seen = ~np.isnan(h["ball"][:, 0])
        for s, e in S.M3.runs(h["truth"] == pm.D):
            if e - s < 3 * FPS:
                continue
            f = seen[s:e].mean()
            whole += f == 0; never += f == 1; mixed += 0 < f < 1
    print("atd_stats3 (real tracking, dead = v2 proxy)")
    print(" ".join(f"{k}={v if not isinstance(v, float) else round(v, 3)}" for k, v in st.items()))
    print("missing-run counts by length (s): " + " ".join(f"[{a:g},{b:g})={n}" for a, b, n in zip(edges[:-1], edges[1:], hist)))
    print(f"stoppages >= 3 s: hidden whole {whole}, seen whole {never}, mixed {mixed}")


if __name__ == "__main__":
    main()
