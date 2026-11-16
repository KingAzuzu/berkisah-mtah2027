"""Human only. Correctness check on the public 5-minute window (RULES section 2: checking, not tuning): why is v10's D recall 0.33?

    idsse-tests\\.venv\\Scripts\\python -m possession.public_dead_diag

For each ground-truth D run inside the graded window: start (s from window start), duration, share labelled D by v10 and by v2,
ATD ball seen share, median ATD ball speed when seen (m/s), and the label v10 gives most often there. One short line per run.
The answer must be read as "which case does the pipeline mishandle", never as a parameter to set from these numbers.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
import mtah_guard  # noqa: E402

GT = ROOT / "evaluation" / "gt" / "BAR-ATH_possession_gt_public.csv"
PRED = {"v10": ROOT / "evaluation" / "results" / "v10.csv", "v2": ROOT / "evaluation" / "results" / "v2.csv"}


def main():
    mtah_guard.refuse_under_claude()
    import possession_metrics as pm
    from possession import bar_ath
    gt = pm.label_codes(pm.read_label_csv(GT))
    pr = {k: pm.label_codes(pm.read_label_csv(p)) for k, p in PRED.items() if p.exists()}
    ball, _, _ = bar_ath.load_atd()
    D = pm.LABELS.index("D")
    graded = gt != pm.LABELS.index("X")
    g0 = np.flatnonzero(graded)[0]
    isd = (gt == D) & graded
    ch = np.flatnonzero(np.diff(np.r_[0, isd.astype(np.int8), 0]))
    seen = ~np.isnan(ball[:, 0])
    spd = np.r_[np.nan, np.linalg.norm(np.diff(ball, axis=0), axis=1) * 25]
    print(f"graded frames {graded.sum()}, truth D frames {isd.sum()}, truth D runs {len(ch) // 2}")
    for s, e in zip(ch[::2], ch[1::2]):
        sh = " ".join(f"{k} D {np.mean(v[s:e] == D):.2f}" for k, v in pr.items())
        top = pm.LABELS[np.bincount(pr["v10"][s:e], minlength=4).argmax()] if "v10" in pr else "?"
        sp = spd[s:e][seen[s:e]]
        print(f"t {(s - g0) / 25:6.1f} s  dur {(e - s) / 25:5.1f} s  {sh}  v10 mostly {top}  ball seen {seen[s:e].mean():.2f}  "
              f"ball speed med {np.nanmedian(sp) if len(sp) else float('nan'):.1f} m/s")
    alive_called_d = graded & (gt != D) & (pr["v10"] == D)
    print(f"truth alive frames labelled D by v10: {alive_called_d.sum()}")


if __name__ == "__main__":
    main()
