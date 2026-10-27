"""Score predictions with the organiser's possession_metrics, halves joined by one X frame.

The single X frame between halves is what valid_runs() splits on, so no possession or transition is
scored across halftime -- the same thing the 20k-frame hole does on BAR-ATH.
"""
import numpy as np

from .data import pm


def join(per_half):
    gap = np.array([pm.X], np.int8)
    parts = []
    for i, codes in enumerate(per_half):
        if i:
            parts.append(gap)
        parts.append(np.asarray(codes, np.int8))
    return np.concatenate(parts)


def score(halves, preds):
    """possession_metrics.evaluate() for one match; preds is a list of per-half code arrays."""
    return pm.evaluate(join([h["truth"] for h in halves]), join(preds))


def summarise(results):
    """{mid: evaluate() dict} -> mean and spread of the two headline numbers."""
    spell = np.array([r["spell_f1_mean"] for r in results.values()])
    macro = np.array([r["macro_f1"] for r in results.values()])
    return {"spell_mean": float(spell.mean()), "spell_min": float(spell.min()), "spell_max": float(spell.max()),
            "macro_mean": float(macro.mean()), "n": len(spell)}
