"""GPU decoders vs the CPU reference on real degraded IDSSE halves (tuning matches only).

    python -m tests.test_gpu_decode --small                 # small exactness checks (pytest not installed in the venv;
                                                            # the test_* functions are also pytest-compatible)
    python -m tests.test_gpu_decode --exact [--cells 24]    # full exactness report -> logs/gpu_decode_exact.json
    python -m tests.test_gpu_decode --bench 16 64 256       # CPU-sequential vs GPU-batched timings
Locked matches are never loaded (data.LOCKED is asserted absent).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from possession import gpu_decode as gd, mbr, safety  # noqa: E402
from possession.data import LOCKED, TUNE  # noqa: E402
from possession.run_idsse import load  # noqa: E402

LOGS = ROOT / "possession" / "logs"
MATCH = TUNE[0]
assert MATCH not in LOCKED


def make_cells(n, seed=0):
    """n parameter cells spread around the safety net: SCALE, V0, VS, switch costs, temperature."""
    rng = np.random.default_rng(seed)
    return [{"SCALE": float(rng.choice([0.5, 1.0, 2.0])), "V0": float(rng.choice([1.0, 1.5, 2.0])),
             "VS": float(rng.choice([0.5, 1.0])), "SWITCH_TEAM": float(rng.choice([3, 5, 10, 20, 40])),
             "SWITCH_DEAD": float(rng.choice([5, 10, 20, 40])), "TEMP": float(rng.choice([0.5, 1.0, 2.0]))}
            for _ in range(n)]


def cell_evidence(f, cells):
    return np.stack([safety.emissions(f, {**safety.PARAMS, **c}) for c in cells])


def path_score(e, path, st, sd):
    """Score safety.viterbi maximises: sum e[t, z_t] - switch costs (float64, same order of operations)."""
    cost = np.array([[0.0, st, sd], [st, 0.0, sd], [sd, sd, 0.0]])
    return float(e[np.arange(len(path)), path].sum() - cost[path[:-1], path[1:]].sum())


def exact_report(half, cells, dtype=torch.float64):
    f = safety.features(half)
    E = cell_evidence(f, cells)
    st = np.array([c["SWITCH_TEAM"] for c in cells])
    sd = np.array([c["SWITCH_DEAD"] for c in cells])
    tp = np.array([c["TEMP"] for c in cells])
    gv = gd.viterbi(E, st, sd, dtype=dtype)
    ga = gd.posterior_argmax(E, st, sd, tp, dtype=dtype)
    rows = []
    for i, c in enumerate(cells):
        cv = safety.viterbi(E[i], st[i], sd[i])
        es = E[i] / tp[i]
        alpha, beta, logz = mbr.forward_backward(es, st[i], sd[i])
        gamma = mbr.marginals(alpha, beta, logz)
        ca = np.argmax(gamma, 1).astype(np.int8)
        dv, da = np.flatnonzero(cv != gv[i]), np.flatnonzero(ca != ga[i])
        row = {"cell": c, "viterbi_diff": int(len(dv)), "argmax_diff": int(len(da))}
        if len(dv):
            s_cpu, s_gpu = path_score(E[i], cv, st[i], sd[i]), path_score(E[i], gv[i].astype(np.int8), st[i], sd[i])
            row["viterbi_rel_score_diff"] = abs(s_cpu - s_gpu) / max(abs(s_cpu), 1e-300)
        if len(da):
            g = gamma[da]
            top2 = np.sort(g, 1)[:, -2:]
            row["argmax_max_gap_at_diff"] = float((top2[:, 1] - top2[:, 0]).max())  # 0 = exact tie
        rows.append(row)
    return rows


def test_exact_small():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA")
    half = load(MATCH)[0]
    half = {k: (v[:6000] if k in ("ball", "xy", "truth") else v) for k, v in half.items()}  # keep it quick
    rows = exact_report(half, make_cells(4))
    assert all(r["viterbi_diff"] == 0 for r in rows), rows
    assert all(r["argmax_diff"] == 0 or r.get("argmax_max_gap_at_diff", 1) < 1e-9 for r in rows), rows


def test_padded_halves():
    if not torch.cuda.is_available():
        raise unittest.SkipTest("no CUDA")
    h0, h1 = load(MATCH)
    h0 = {k: (v[:5000] if k in ("ball", "xy", "truth") else v) for k, v in h0.items()}
    h1 = {k: (v[:3500] if k in ("ball", "xy", "truth") else v) for k, v in h1.items()}
    cell = {**safety.PARAMS, "SWITCH_TEAM": 10.0, "SWITCH_DEAD": 20.0}
    es = [safety.emissions(safety.features(h), cell) for h in (h0, h1)]
    T = max(len(e) for e in es)
    E = np.zeros((2, T, 3))
    for i, e in enumerate(es):
        E[i, :len(e)] = e
    L = [len(e) for e in es]
    v = gd.viterbi(E, 10.0, 20.0, lengths=L)
    a = gd.posterior_argmax(E, 10.0, 20.0, 1.0, lengths=L)
    for i, e in enumerate(es):
        assert (v[i, :L[i]] == safety.viterbi(e, 10.0, 20.0)).all()
        alpha, beta, logz = mbr.forward_backward(e, 10.0, 20.0)
        assert (a[i, :L[i]] == np.argmax(mbr.marginals(alpha, beta, logz), 1)).all()
        assert (v[i, L[i]:] == -1).all()


def _gpu_sync():
    torch.cuda.synchronize()


def bench(ns, half, dtype_list=(torch.float64, torch.float32)):
    f = safety.features(half)
    out = {"T": len(half["ball"]), "contention_note": None, "rows": []}
    for n in ns:
        cells = make_cells(n, seed=1)
        E = cell_evidence(f, cells)
        st = np.array([c["SWITCH_TEAM"] for c in cells])
        sd = np.array([c["SWITCH_DEAD"] for c in cells])
        tp = np.array([c["TEMP"] for c in cells])
        row = {"N": n}
        t0 = time.perf_counter()
        cpu_v = [safety.viterbi(E[i], st[i], sd[i]) for i in range(n)]
        row["cpu_viterbi_s"] = time.perf_counter() - t0
        k = min(n, 16)  # forward-backward CPU is timed on k cells and scaled linearly
        t0 = time.perf_counter()
        for i in range(k):
            mbr.forward_backward(E[i] / tp[i], st[i], sd[i])
        row["cpu_fb_s"] = (time.perf_counter() - t0) * n / k
        row["cpu_fb_timed_cells"] = k
        for dt in dtype_list:
            nm = "f64" if dt == torch.float64 else "f32"
            gd.viterbi(E[:2], st[:2], sd[:2], dtype=dt)  # warm-up
            _gpu_sync()
            t0 = time.perf_counter()
            gv = gd.viterbi(E, st, sd, dtype=dt)
            _gpu_sync()
            row[f"gpu_viterbi_{nm}_s"] = time.perf_counter() - t0
            row[f"gpu_viterbi_{nm}_diff_frames"] = int(sum((gv[i] != cpu_v[i]).sum() for i in range(n)))
            t0 = time.perf_counter()
            gd.posterior_argmax(E, st, sd, tp, dtype=dt)
            _gpu_sync()
            row[f"gpu_fb_{nm}_s"] = time.perf_counter() - t0
        print(json.dumps(row), flush=True)
        out["rows"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--exact", action="store_true")
    ap.add_argument("--cells", type=int, default=24)
    ap.add_argument("--bench", type=int, nargs="*")
    ap.add_argument("--note", default="")
    args = ap.parse_args()
    if args.small:
        test_exact_small()
        test_padded_halves()
        print("small tests passed")
    halves = load(MATCH)
    if args.exact:
        res = {"match": MATCH, "cells": args.cells, "halves": []}
        for hi, half in enumerate(halves):
            for nm, dt in (("f64", torch.float64), ("f32", torch.float32)):
                rows = exact_report(half, make_cells(args.cells), dt)
                summ = {"half": hi, "dtype": nm, "T": len(half["ball"]),
                        "viterbi_cells_differing": sum(r["viterbi_diff"] > 0 for r in rows),
                        "viterbi_frames_differing": sum(r["viterbi_diff"] for r in rows),
                        "argmax_cells_differing": sum(r["argmax_diff"] > 0 for r in rows),
                        "argmax_frames_differing": sum(r["argmax_diff"] for r in rows),
                        "max_viterbi_rel_score_diff": max([r.get("viterbi_rel_score_diff", 0.0) for r in rows]),
                        "max_argmax_gap_at_diff": max([r.get("argmax_max_gap_at_diff", 0.0) for r in rows]),
                        "rows_with_diffs": [r for r in rows if r["viterbi_diff"] or r["argmax_diff"]]}
                print(json.dumps({k: v for k, v in summ.items() if k != "rows_with_diffs"}), flush=True)
                res["halves"].append(summ)
        LOGS.mkdir(exist_ok=True)
        (LOGS / "gpu_decode_exact.json").write_text(json.dumps(res, indent=1))
    if args.bench:
        res = bench(args.bench, halves[0])
        res["contention_note"] = args.note
        res["match"] = MATCH
        (LOGS / "gpu_decode_bench.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
