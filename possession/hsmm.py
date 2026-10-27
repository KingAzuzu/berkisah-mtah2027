"""Explicit-duration semi-Markov Viterbi over H / A / D (WS1).

    fit(truths, block=5) -> model     duration pmfs + segment transitions, from training truth only
    decode(e, model, wd, wt)          (T, 3) evidence -> (T,) int8 codes

Score of a path = sum of per-frame evidence + wd * sum of log duration pmf + wt * sum of log segment
transition probabilities (no self transitions). The first segment of a half is censored (no duration
cost). Decoding runs on blocks of `block` frames (5 fps), vectorised over the segment length; then every
internal boundary is refined at full rate within +/- `block` frames on the evidence alone.
Durations beyond Dmax (pooled 99.5th percentile of training segments) use a geometric tail per state.
Emission temperature is not a separate parameter: only the ratios of evidence : wd : wt matter.
Deterministic, CPU, pure numpy.
"""
import numpy as np

NEG = -1e18


def _runs(codes):
    c = np.asarray(codes)
    cut = np.flatnonzero(np.diff(c)) + 1
    s = np.r_[0, cut]
    e = np.r_[cut, len(c)]
    return c[s], e - s


def fit(truths, block=5, q=99.5, smooth=0.2):
    """truths: list of per-half truth code arrays (H 0, A 1, D 2) of the TRAINING matches only."""
    durs = {0: [], 1: [], 2: []}
    trans = np.ones((3, 3))  # add-one smoothing
    for t in truths:
        st, ln = _runs(t)
        for s, n in zip(st, ln):
            durs[int(s)].append(max(1, int(round(n / block))))
        for a, b in zip(st[:-1], st[1:]):
            trans[a, b] += 1
    allq = np.concatenate([np.array(v) for v in durs.values()])
    D = max(8, int(np.ceil(np.percentile(allq, q))))
    np.fill_diagonal(trans, 0)
    lt = np.log(trans / trans.sum(1, keepdims=True) + 1e-12)
    np.fill_diagonal(lt, NEG)
    d = np.arange(1, D + 1)
    ld = np.zeros((3, D))
    logr = np.zeros(3)
    for j in range(3):
        x = np.array(durs[j])
        tail = x >= D
        m = float(np.clip(tail.mean(), 1e-3, 0.5))
        r = 1.0 - 1.0 / (1.0 + (x[tail].mean() - D + 1.0 if tail.any() else 1.0))
        logr[j] = np.log(np.clip(r, 0.5, 0.995))
        pmf = np.zeros(D)
        for k, c in zip(*np.unique(np.minimum(x, D), return_counts=True)):
            sig = smooth * k + 0.5
            g = np.exp(-0.5 * ((d - k) / sig) ** 2)
            pmf += c * g / g.sum()
        pmf = pmf[:D - 1] / pmf[:D - 1].sum() * (1 - m)
        pmf = np.maximum(pmf, 1e-7)
        ld[j, :D - 1] = np.log(pmf)
        ld[j, D - 1] = np.log(m * (1 - np.exp(logr[j])))  # P(d == D) opens the geometric tail
    return {"block": block, "D": D, "ld": ld, "lt": lt, "logr": logr}


def _blocks(e, block):
    T = len(e)
    Tb = -(-T // block)
    pad = np.zeros((Tb * block, 3))
    pad[:T] = np.nan_to_num(e)
    return pad.reshape(Tb, block, 3).sum(1)


def decode(e, model, wd=1.0, wt=1.0, refine=True):
    block, D, ld, lt, logr = (model[k] for k in ("block", "D", "ld", "lt", "logr"))
    T = len(e)
    eb = _blocks(e, block)
    Tb = len(eb)
    cs = np.vstack([np.zeros((1, 3)), np.cumsum(eb, 0)])  # (Tb+1, 3) prefix sums at boundaries
    rld = (wd * ld[:, ::-1]).copy()  # aligned with d = D..1
    wlt = wt * np.where(lt < -1e17, NEG, lt)
    Dend = np.full((Tb + 1, 3), NEG)
    X = np.full((3, Tb + 1 + D), NEG)  # X[j][s + D] = best score into a segment of j starting at s, minus cs
    F = np.full((Tb + 1, 3), NEG)  # best score of a segment in its geometric tail ending at boundary b
    Fstart = np.zeros((Tb + 1, 3), np.int64)
    blen = np.zeros((Tb + 1, 3), np.int64)  # length (blocks) of the last segment of the best path to (b, j)
    blen_init = np.zeros((Tb + 1, 3), bool)
    for b in range(1, Tb + 1):
        if b > 1:
            for j in range(3):
                i1, i2 = [i for i in range(3) if i != j]
                v = np.maximum(Dend[b - 1, i1] + wlt[i1, j], Dend[b - 1, i2] + wlt[i2, j])
                X[j, b - 1 + D] = v - cs[b - 1, j]
        for j in range(3):
            rest = X[j, b:b + D] + rld[j]
            # index 0 is d = D (tail opener), 1.. are d = D-1 .. 1
            k = int(np.argmax(rest[1:]))
            m = rest[1 + k]
            g = rest[0]
            fprev = F[b - 1, j] + (cs[b, j] - cs[b - 1, j]) + wd * logr[j]
            if g + cs[b, j] >= fprev:
                F[b, j], Fstart[b, j] = g + cs[b, j], b - D
            else:
                F[b, j], Fstart[b, j] = fprev, Fstart[b - 1, j]
            cands = (m + cs[b, j], F[b, j], cs[b, j])  # normal, tail, first segment (censored)
            c = int(np.argmax(cands))
            Dend[b, j] = cands[c]
            if c == 0:
                blen[b, j] = D - 1 - k
            elif c == 1:
                blen[b, j] = b - Fstart[b, j]
            else:
                blen[b, j], blen_init[b, j] = b, True
    # traceback
    segs = []
    b, j = Tb, int(np.argmax(Dend[Tb]))
    while b > 0:
        n = int(blen[b, j])
        s = b - n
        segs.append((s, b, j))
        if blen_init[b, j] or s <= 0:
            break
        i1, i2 = [i for i in range(3) if i != j]
        j2 = i1 if Dend[s, i1] + wlt[i1, j] >= Dend[s, i2] + wlt[i2, j] else i2
        b, j = s, j2
    segs.reverse()
    bounds = [s * block for s, _, _ in segs][1:]  # frame index of each internal boundary
    states = [j for _, _, j in segs]
    if refine and bounds:
        e0 = np.nan_to_num(e)
        c25 = np.vstack([np.zeros((1, 3)), np.cumsum(e0, 0)])
        for k in range(len(bounds)):
            left, right = states[k], states[k + 1]
            lo = (bounds[k - 1] if k else 0) + 1
            hi = (bounds[k + 1] if k + 1 < len(bounds) else T) - 1
            cands = sorted(range(-block, block + 1), key=lambda x: (abs(x), x))
            best, bb = None, bounds[k]
            for sft in cands:
                x = bounds[k] + sft
                if not (lo <= x <= hi) or x > T:
                    continue
                v = (c25[x, left] - c25[bounds[k], left]) - (c25[x, right] - c25[bounds[k], right])
                if best is None or v > best:
                    best, bb = v, x
            bounds[k] = bb
    edges = [0] + bounds + [T]
    out = np.empty(T, np.int8)
    for k, jj in enumerate(states):
        out[edges[k]:edges[k + 1]] = jj
    return out
