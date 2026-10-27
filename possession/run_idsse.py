"""Score a method on degraded IDSSE, per match plus mean and spread.

    python -m possession.run_idsse                      # baseline and safety net on the 5 tuning matches
    python -m possession.run_idsse --clean              # same, on clean IDSSE (ball always seen)

The locked matches (data.LOCKED) are refused unless --locked is given; that is for the one final
check after the method is frozen.
"""
import argparse
import pickle

import numpy as np

from . import baseline, safety
from .data import LOCKED, TUNE, idsse, load_idsse
from .degrade import degrade_match
from .score import score, summarise


def load(mid, clean=False, seed=0):
    if clean:
        return load_idsse(mid)
    cache = idsse.DATA / "cache" / f"{mid}_degraded{seed}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    halves = degrade_match(load_idsse(mid), mid, seed)
    cache.parent.mkdir(exist_ok=True)
    cache.write_bytes(pickle.dumps(halves, protocol=4))
    return halves


def describe(halves):
    seen = np.mean(np.concatenate([~np.isnan(h["ball"][:, 0]) for h in halves]))
    objects = np.mean(np.concatenate([np.sum(~np.isnan(h["xy"][:, :, 0]), axis=1) for h in halves]))
    dead = np.mean(np.concatenate([h["truth"] for h in halves]) == 2)
    return f"ball seen {seen:.3f}  objects {objects:.1f}  truth D {dead:.3f}"


METHODS = {"baseline50": lambda h: baseline.predict(h, 50), "safety": safety.predict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--locked", action="store_true", help="final check only, after the method is frozen")
    ap.add_argument("--method", nargs="+", default=list(METHODS))
    args = ap.parse_args()
    mids = LOCKED if args.locked else TUNE

    data = {m: load(m, args.clean) for m in mids}
    for m in mids:
        print(m, describe(data[m]))
    for name in args.method:
        res = {m: score(data[m], [METHODS[name](h) for h in data[m]]) for m in mids}
        for m, r in res.items():
            print(f"  {name:<11} {m}  spellF1 {r['spell_f1_mean']:.4f}  macroF1 {r['macro_f1']:.4f}  "
                  f"spells {r['n_pred_spells']}/{r['n_gt_spells']}")
        s = summarise(res)
        print(f"{name:<11} spellF1 mean {s['spell_mean']:.4f} [{s['spell_min']:.4f}, {s['spell_max']:.4f}]  "
              f"macroF1 mean {s['macro_mean']:.4f}  (n={s['n']})")


if __name__ == "__main__":
    main()
