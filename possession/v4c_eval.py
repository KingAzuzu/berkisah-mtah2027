"""Confirmation scoring (fresh seeds 10-14), fixed rule: pass iff mean spellF1 >= 0.661 and wins vs v3 >= 4/5.
python -m possession.v4c_eval -> logs/v4c_eval.json. Hook decoder (HOOK 0.6) only."""
import json
from . import v4_eval as E
from .data import TUNE
from . import pipeline as pl

if __name__ == "__main__":
    out = []
    for cfg in ("cA", "cB"):
        r = E.row(f"{cfg} x5 seeds 10-14 +hook", E.run(cfg, tuple(range(10, 15)), True))
        r["pass"] = bool(r["mean"] >= 0.661 and r["wins_vs_v3"] >= 4)
        out.append(r)
        print(r["name"], f"{r['mean']:.4f} [{r['min']:.4f},{r['max']:.4f}] wins {r['wins_vs_v3']}/5 macro {r['macro']:.4f} pass={r['pass']}",
              " ".join(f"{r['per_match'][m]:.3f}" for m in TUNE), flush=True)
    json.dump(out, open(pl.HERE / "logs" / "v4c_eval.json", "w"), indent=1)
