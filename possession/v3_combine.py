"""Combination: restart evidence (c, fixed inner-best cell per tag config) then cleanup (b, K 0 L 25, the fold choice).
Outer models (trained on the other 4 matches), v2 fold cells. Prints/saves per tag config; informational."""
import json

import numpy as np

from . import pipeline as pl
from . import restarts as R
from .data import TUNE
from .lomo import LOG_DIR
from .restarts_run import build_tags
from .v3_candidates import prep_inputs
from .v3_lib import fmt_row, pdead_cached, spell_score, summary, v2_cells
from .v3_post import cleanup

CFG = [(0.5, 0), (1.5, 0), (0.5, 1), (1.5, 1)]


def main():
    P = prep_inputs()
    tables, cells = pdead_cached("seed0"), v2_cells()
    out = {}
    for sigma, seed in CFG:
        tags, _ = build_tags(sigma, seed, 0.15)
        c = json.loads((LOG_DIR / f"v3_c_s{sigma}_seed{seed}.json").read_text())["fixed_inner_best"]["cell"]
        for L in (0, 25):
            res = {}
            for m in TUNE:
                paths = []
                for h, pd in zip(P[m], tables["outer"][m]):
                    e = R.adjust(pl.combine(h["team"], pd, cells[m]["ALPHA"]), tags[h["fp"]], c)
                    paths.append(cleanup(pl.decode(e, cells[m]), 0, L))
                res[m] = spell_score([{"truth": h["truth"]} for h in P[m]], paths)
            s = summary(res)
            name = f"c_s{sigma}_seed{seed}" + ("+b25" if L else "")
            out[name] = {"cell_c": c, **s}
            print(fmt_row(name, s), flush=True)
    (LOG_DIR / "v3_combine.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
