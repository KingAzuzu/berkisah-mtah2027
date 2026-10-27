"""Minimum Bayes risk decoding on the safety-net HMM (WS3).

Same 3-state model as safety.viterbi: path score = sum_t e_t(z_t)/TEMP - sum switches cost(z_{t-1}, z_t),
i.e. a Gibbs distribution over paths with log transition penalties -cost. Costs are in the units of
the (scaled) emissions. Steps:

  1. log-space forward / backward (marginals gamma_t(k)).
  2. forward-filtering backward-sampling: S paths, numpy Generator with a fixed seed.
  3. candidates: safety.viterbi at several multiples of the switch costs, the posterior argmax, and
     every sample (leave-one-out against itself).
  4. pick the candidate with the highest mean spellF1 against the samples. spellF1 is the organiser's
     possession_metrics machinery (possession_spells, match_spells, one valid run per half). Only the
     pairwise overlap enumeration is replaced by an equivalent sweep (check_fast_overlap() asserts
     equality with possession_metrics.spell_scores), because the O(n*m) loop is too slow for noisy samples.
  5. optional keep-threshold: spells whose mean posterior marginal is below KEEP become D.

Look-ahead only; deterministic; CPU.
"""
import math
from bisect import bisect_right

import numpy as np

from . import safety
from .data import pm

H, A, D = pm.H, pm.A, pm.D
NEG = -1e300


def cost_matrix(st, sd):
    return [[0.0, st, sd], [st, 0.0, sd], [sd, sd, 0.0]]


def _lse3(a, b, c):
    m = a if a > b else b
    if c > m:
        m = c
    return m + math.log(math.exp(a - m) + math.exp(b - m) + math.exp(c - m))


def forward_backward(es, st, sd):
    """es (T,3) scaled emissions -> alpha (T,3), beta (T,3), log Z. Pure Python 3-state loops."""
    c = cost_matrix(st, sd)
    T = len(es)
    rows = es.tolist()
    alpha = np.empty((T, 3))
    a = rows[0][:]
    alpha[0] = a
    for t in range(1, T):
        r = rows[t]
        a = [_lse3(a[0] - c[0][j], a[1] - c[1][j], a[2] - c[2][j]) + r[j] for j in range(3)]
        alpha[t] = a
    beta = np.empty((T, 3))
    b = [0.0, 0.0, 0.0]
    beta[T - 1] = b
    for t in range(T - 2, -1, -1):
        r = rows[t + 1]
        w = [r[j] + b[j] for j in range(3)]
        b = [_lse3(w[0] - c[i][0], w[1] - c[i][1], w[2] - c[i][2]) for i in range(3)]
        beta[t] = b
    logz = float(np.logaddexp.reduce(alpha[-1]))
    return alpha, beta, logz


def marginals(alpha, beta, logz):
    return np.exp(alpha + beta - logz)


def ffbs(alpha, st, sd, n, seed=0):
    """n paths (n, T) int8 from the exact posterior, by backward sampling on the filtered alphas."""
    T = len(alpha)
    c = np.array(cost_matrix(st, sd))
    rng = np.random.default_rng(seed)
    u = rng.random((T, n))
    # cp[t, j, :] = cumulative P(z_t = i | z_{t+1} = j, y_1..t)
    lg = alpha[:-1, :, None] - c[None]                       # (T-1, i, j)
    lg = lg - np.logaddexp.reduce(lg, axis=1, keepdims=True)
    cp = np.cumsum(np.exp(lg), axis=1).transpose(0, 2, 1)   # (T-1, j, i)
    z = np.empty((n, T), np.int8)
    pT = np.exp(alpha[-1] - np.logaddexp.reduce(alpha[-1]))
    cur = np.minimum((u[T - 1][:, None] > np.cumsum(pT)[None]).sum(1), 2)
    z[:, T - 1] = cur
    for t in range(T - 2, -1, -1):
        cur = np.minimum((u[t][:, None] > cp[t][cur]).sum(1), 2)
        z[:, t] = cur
    return z


# --- spellF1 between two paths -----------------------------------------------------------------

def spells_of(codes):
    return pm.possession_spells(np.asarray(codes, np.int8), [(0, len(codes))])


def fast_overlaps(gt, pred):
    """Same list as pm.spell_overlaps, by a sweep (both lists are sorted and disjoint)."""
    starts = [s for s, _, _ in pred]
    out = []
    for i, (gs, ge, gc) in enumerate(gt):
        j = max(0, bisect_right(starts, gs) - 1)
        while j < len(pred) and pred[j][0] < ge:
            ps, pe, pc = pred[j]
            if pc == gc and pe > gs:
                inter = min(ge, pe) - max(gs, ps)
                union = max(ge, pe) - min(gs, ps)
                out.append((inter / float(union), i, j))
            j += 1
    out.sort(key=lambda item: (-item[0], item[1], item[2]))
    return out


def spell_f1(gt, pred):
    """Mean spellF1 over pm.SPELL_IOUS from precomputed spell lists (same maths as pm.spell_scores)."""
    ov = fast_overlaps(gt, pred)
    f1s = []
    for th in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(ov, len(gt), len(pred), th)
        p = tp / float(tp + fp) if tp + fp else 0.0
        r = tp / float(tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def check_fast_overlap(paths):
    for a in paths:
        for b in paths:
            ref = pm.spell_scores(a, b, [(0, len(a))])["mean_f1"]
            got = spell_f1(spells_of(a), spells_of(b))
            assert abs(ref - got) < 1e-12, (ref, got)


# --- the decoder ---------------------------------------------------------------------------------

def drop_low_posterior(path, gamma, keep):
    if keep <= 0:
        return path
    out = path.copy()
    for s, e, k in spells_of(path):
        if gamma[s:e, k].mean() < keep:
            out[s:e] = D
    return out


def decode_variants(e, st, sd, temp=1.0, n=20, mults=(1.0,), keeps=(0.0,), seed=0, info=None):
    """All decoders on one evidence array. Returns {(method, keep): path}.

    methods: viterbi (at st, sd), argmax (posterior marginal), mbr (candidates as in the module doc).
    """
    es = e / temp
    alpha, beta, logz = forward_backward(es, st, sd)
    gamma = marginals(alpha, beta, logz)
    samples = ffbs(alpha, st, sd, n, seed)
    vit = {m: safety.viterbi(es, st * m, sd * m) for m in mults}
    base = vit[1.0] if 1.0 in vit else safety.viterbi(es, st, sd)
    amax = np.argmax(gamma, axis=1).astype(np.int8)

    cands = [vit[m] for m in mults] + [amax] + list(samples)
    cand_sp = [spells_of(c) for c in cands]
    ref_sp = cand_sp[len(mults) + 1:]
    util = np.zeros(len(cands))
    for ci, sp in enumerate(cand_sp):
        tot, cnt = 0.0, 0
        for ri, rs in enumerate(ref_sp):
            if ci == len(mults) + 1 + ri:
                continue
            tot += spell_f1(rs, sp)
            cnt += 1
        util[ci] = tot / cnt
    best = int(np.argmax(util))
    if info is not None:
        info.update(best=best, n_cands=len(cands), util_best=float(util[best]),
                    util_viterbi=float(util[list(mults).index(1.0)]) if 1.0 in mults else None,
                    picked="viterbi*%g" % mults[best] if best < len(mults) else
                    ("argmax" if best == len(mults) else "sample"),
                    n_spells_picked=len(cand_sp[best]))
    chosen = {"viterbi": base, "argmax": amax, "mbr": cands[best]}
    return {(m, k): drop_low_posterior(p, gamma, k) for m, p in chosen.items() for k in keeps}
