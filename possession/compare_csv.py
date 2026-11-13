"""Compare two submission CSVs. Human only (they are competition outputs); prints aggregates only.

    python -m possession.compare_csv <a.csv> <b.csv>
"""
import sys

import numpy as np

import mtah_guard


def spells(lab):
    ch = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(lab)]
    k = np.isin(lab[st], ("H", "A"))
    return int(k.sum()), float(((en - st)[k]).mean() / 25) if k.any() else 0.0


def main(a, b):
    mtah_guard.refuse_under_claude()
    A, B = (np.loadtxt(p, delimiter=",", skiprows=1, usecols=1, dtype=str) for p in (a, b))
    g = (A != "X") & (B != "X")
    print(f"frame agreement {np.mean(A[g] == B[g]):.4f}; H/A swaps {np.mean(np.isin(A[g], ('H', 'A')) & np.isin(B[g], ('H', 'A')) & (A[g] != B[g])):.4f}")
    for name, L in (("a", A), ("b", B)):
        n, d = spells(L)
        print(f"{name}: possessions {n}, mean {d:.1f} s, D share {np.mean(L[L != 'X'] == 'D'):.3f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
