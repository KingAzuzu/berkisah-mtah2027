"""GPU-batched 3-state decoders for parameter tuning (NOT for the final submission, which stays CPU).

Batch dimension B = parameter cells (optionally several halves, padded, via `lengths`). The time
recursion is sequential (one short kernel group per frame); everything else is vectorised over B and
the 3 states.

  viterbi(e, st, sd)            same model and tie-breaking as safety.viterbi
  posterior_argmax(e, st, sd, temp)   same HMM as mbr.forward_backward / mbr.marginals + np.argmax

Inputs
  e        (T, 3) shared by all cells, or (B, T, 3) one evidence array per cell (rows past `lengths` ignored)
  st, sd   scalar or (B,) switch costs (H<->A, alive<->D)
  temp     scalar or (B,) temperature; emissions are divided by it, costs are not (as in mbr.py)
  lengths  optional (B,) true lengths when cells are padded to a common T
Returns numpy int8 (B, T); padded frames hold -1.

Exactness: in float64 Viterbi is bit-identical arithmetic (subtract, max, add) with the same first-index
tie-breaking (strict > in order 0,1,2) as safety.viterbi. Forward-backward uses torch.logsumexp, which
agrees with mbr._lse3 up to rounding (~1e-15), so argmax can differ only at near-exact ties.
"""
import numpy as np
import torch

NEG_STATE = -1


def _dev(device):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def _prep(e, st, sd, temp, lengths, dtype, device):
    """-> E (T,B,3) contiguous, C (B,3,3), lengths (B,) long on device, B, T."""
    e, st, sd, temp, lengths, B = _expand(e, st, sd, temp, lengths)
    T = e.shape[1]
    E = torch.as_tensor(np.ascontiguousarray(e), device=device)
    if temp is not None:
        E = E / torch.as_tensor(np.array(temp), device=device)[:, None, None]
    E = E.to(dtype).transpose(0, 1).contiguous()  # (T, B, 3)
    z = np.zeros(B)
    C = np.stack([np.stack([z, st, sd], 1), np.stack([st, z, sd], 1), np.stack([sd, sd, z], 1)], 1)  # (B,i,j)
    C = torch.as_tensor(C, device=device).to(dtype)
    L = np.full(B, T) if lengths is None else np.asarray(lengths)
    return E, C, torch.as_tensor(L, device=device, dtype=torch.long), B, T


def _first_argmax3(v0, v1, v2):
    """First index of the maximum among three tensors (strict > in order 0,1,2, like safety.viterbi)."""
    a = (v1 > v0).to(torch.int8)
    best = torch.maximum(v0, v1)
    return torch.where(v2 > best, torch.full_like(a, 2), a)


def _viterbi_chunk(e, st, sd, lengths, dtype, device):
    E, C, L, B, T = _prep(e, st, sd, None, lengths, dtype, device)
    S = torch.empty_like(E)
    S[0] = E[0]
    tmp = torch.empty(B, 3, 3, dtype=E.dtype, device=device)
    m = torch.empty(B, 3, dtype=E.dtype, device=device)
    for t in range(1, T):
        torch.sub(S[t - 1].unsqueeze(2), C, out=tmp)
        torch.amax(tmp, 1, out=m)
        torch.add(m, E[t], out=S[t])
    # back[t, b, j] = first argmax_i (S[t-1, b, i] - C[b, i, j])
    back = torch.empty(T, B, 3, dtype=torch.int8, device=device)
    back[0] = 0
    step = max(1, int(2e8 // (B * 9 * E.element_size())))
    for s in range(1, T, step):
        V = S[s - 1:min(T, s + step) - 1].unsqueeze(3) - C[None]  # (Tc,B,i,j)
        back[s:s + V.shape[0]] = _first_argmax3(V[:, :, 0], V[:, :, 1], V[:, :, 2])
    idx = torch.arange(B, device=device)
    last = _first_argmax3(*[S[L - 1, idx, k] for k in range(3)])  # np.argmax is also first-index
    back, last, L = back.cpu().numpy(), last.cpu().numpy(), L.cpu().numpy()
    path = np.empty((B, T), np.int8)
    cur = np.zeros(B, np.int8)
    ar = np.arange(B)
    for t in range(T - 1, -1, -1):
        hit = L - 1 == t
        cur = np.where(hit, last, cur)
        path[:, t] = cur
        cur = back[t, ar, cur]
    for b in range(B):
        path[b, L[b]:] = NEG_STATE
    return path


def _expand(e, st, sd, temp=None, lengths=None):
    """Normalise to e (B,T,3) (a broadcast view when shared) and per-cell (B,) vectors."""
    e = np.asarray(e, np.float64)
    B = max(np.size(st), np.size(sd), 1 if temp is None else np.size(temp),
            1 if lengths is None else len(lengths), e.shape[0] if e.ndim == 3 else 1)
    if e.ndim == 2:
        e = np.broadcast_to(e, (B,) + e.shape)
    vec = lambda x: np.broadcast_to(np.asarray(x, np.float64), (B,))  # noqa: E731
    return (e, vec(st), vec(sd), None if temp is None else vec(temp),
            None if lengths is None else np.asarray(lengths), B)


def _chunked(fn, e, st, sd, temp, lengths, B, max_b, dtype, device):
    out = []
    for a in range(0, B, max_b):
        s = slice(a, min(B, a + max_b))
        args = [e[s], st[s], sd[s]] + ([] if temp is None else [temp[s]])
        out.append(fn(*args, None if lengths is None else lengths[s], dtype, device))
    return np.concatenate(out, 0)


def viterbi(e, st, sd, lengths=None, dtype=torch.float64, device=None, max_b=512):
    e, st, sd, _, lengths, B = _expand(e, st, sd, None, lengths)
    return _chunked(_viterbi_chunk, e, st, sd, None, lengths, B, max_b, dtype, _dev(device))


def forward_backward(e, st, sd, temp=1.0, lengths=None, dtype=torch.float64, device=None):
    """-> (alpha, beta, logz) as torch tensors (T,B,3),(T,B,3),(B,) on device. One chunk (no max_b split)."""
    device = _dev(device)
    E, C, L, B, T = _prep(e, st, sd, temp, lengths, dtype, device)
    alpha = torch.empty_like(E)
    alpha[0] = E[0]
    for t in range(1, T):
        x = alpha[t - 1].unsqueeze(2) - C                  # (B,i,j)
        torch.add(torch.logsumexp(x, 1), E[t], out=alpha[t])
    beta = torch.zeros_like(E)
    ragged = bool((L < T).any())
    if ragged:
        zero = torch.zeros(B, 3, dtype=E.dtype, device=device)
    b = beta[T - 1]
    for t in range(T - 2, -1, -1):
        w = E[t + 1] + b                                   # (B,j)
        nb = torch.logsumexp(w.unsqueeze(1) - C, 2)        # (B,i)
        if ragged:
            nb = torch.where((t + 1 < L)[:, None], nb, zero)
        beta[t] = nb
        b = beta[t]
    idx = torch.arange(B, device=device)
    logz = torch.logsumexp(alpha[L - 1, idx], 1)
    return alpha, beta, logz


def _pa_chunk(e, st, sd, temp, lengths, dtype, device):
    alpha, beta, logz = forward_backward(e, st, sd, temp, lengths, dtype, device)
    T, B, _ = alpha.shape
    L = np.full(B, T) if lengths is None else np.asarray(lengths)
    out = np.empty((B, T), np.int8)
    step = max(1, int(1e8 // (B * 3 * alpha.element_size())))
    for s in range(0, T, step):
        g = torch.exp(alpha[s:s + step] + beta[s:s + step] - logz[None, :, None])
        out[:, s:s + g.shape[0]] = _first_argmax3(g[..., 0], g[..., 1], g[..., 2]).cpu().numpy().T
    for b in range(B):
        out[b, L[b]:] = NEG_STATE
    return out


def posterior_argmax(e, st, sd, temp=1.0, lengths=None, dtype=torch.float64, device=None, max_b=128):
    e, st, sd, temp, lengths, B = _expand(e, st, sd, temp, lengths)
    return _chunked(_pa_chunk, e, st, sd, temp, lengths, B, max_b, dtype, _dev(device))
