"""v21 TCN library: the v4/v6 recipe (TCN + boundary head, EMA, mirror + team-swap augmentation, mirror TTA, short-spell weights)
made trainable on CPU (GPU when available), with configurable dilations and two feature sets:
  f26: tcn_lib.frame_features (unchanged).
  f32: f26 + 6 dead-ball cues that are invariant under team swap and pitch mirror: share of players slower than 1 and 2 m/s,
       players more than 1 m outside the pitch, teams in opposite halves (kick-off shape), ball outside the pitch, ball distance
       to the nearest line (both NaN when the ball is not seen).
Receptive field per side = 2 * sum(dil) steps at 5 Hz (two dilated convs per block): DIL_STD 74 steps = 14.8 s, DIL_LONG 138 = 27.6 s.
Frozen tcn_lib / v4_lib are untouched."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import tcn_lib as T, v4_lib as V
from .data import FPS, HALF_LEN, HALF_WID
from .safety import MAX_SPEED

SUB = T.SUB
DIL_STD = tuple(T.DIL)
DIL_LONG = (1, 2, 4, 8, 16, 32, 4, 2)
MIRROR_COLS = V.MIRROR_COLS          # first 26 columns as tcn_lib; the 6 extras are mirror- and swap-invariant


def extra_features(half):
    xy, ball, team = half["xy"], half["ball"], half["team"]
    with np.errstate(all="ignore"):
        v = np.linalg.norm(np.diff(xy, axis=0, prepend=xy[:1]), axis=2) * FPS
        v[v > MAX_SPEED] = np.nan
        nv = (~np.isnan(v)).sum(1)
        slow1 = np.where(nv > 0, (v < 1).sum(1) / np.maximum(nv, 1), np.nan)
        slow2 = np.where(nv > 0, (v < 2).sum(1) / np.maximum(nv, 1), np.nan)
        out_p = ((np.abs(xy[..., 0]) > HALF_LEN + 1) | (np.abs(xy[..., 1]) > HALF_WID + 1)).sum(1).astype(float)
        cxh, cxa = np.nanmean(xy[:, team == 0, 0], 1), np.nanmean(xy[:, team == 1, 0], 1)
        sep = -np.sign(cxh) * np.sign(cxa)
        bx, by = ball[:, 0], ball[:, 1]
        seen = ~np.isnan(bx)
        bout = np.where(seen, ((np.abs(bx) > HALF_LEN + 0.5) | (np.abs(by) > HALF_WID + 0.5)).astype(float), np.nan)
        bline = np.where(seen, np.minimum(HALF_LEN - np.abs(bx), HALF_WID - np.abs(by)), np.nan)
        X = np.stack([slow1, slow2, out_p, sep, bout, bline], 1).astype(np.float32)
        n = len(X) // SUB * SUB
        return np.nanmean(X[:n].reshape(-1, SUB, X.shape[1]), axis=1)


def features(half, fset="f26"):
    X = T.frame_features(half)
    if fset == "f26":
        return X
    return np.concatenate([X, extra_features(half)[:len(X)]], 1)


def swap_index(nin):
    return np.r_[T.SWAP, np.arange(26, nin)]


def mirror(X):
    X = X.copy()
    X[:, MIRROR_COLS] *= -1
    return X


class Block(nn.Module):
    def __init__(self, c, d, p):
        super().__init__()
        self.c1 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.c2 = nn.Conv1d(c, c, 3, padding=d, dilation=d)
        self.n = nn.GroupNorm(8, c)
        self.p = nn.Dropout(p)

    def forward(self, x):
        return x + self.c2(self.p(F.gelu(self.n(self.c1(x)))))


class TCN21(nn.Module):
    """(B, F, T) -> (B, 4, T): 3 class logits (H, A, D) + boundary logit."""
    def __init__(self, nin=26, c=64, p=0.2, dil=DIL_STD):
        super().__init__()
        self.i = nn.Conv1d(nin, c, 1)
        self.b = nn.ModuleList(Block(c, d, p) for d in dil)
        self.o = nn.Conv1d(c, 3, 1)
        self.ob = nn.Conv1d(c, 1, 1)

    def forward(self, x):
        h = self.i(x)
        for b in self.b:
            h = b(h)
        h = F.gelu(h)
        return torch.cat([self.o(h), self.ob(h)], 1)


def train21(items, seed=0, steps=1500, crop=300, bs=64, dil=DIL_STD, nin=26, short_w=3.0, bd_lambda=V.BD_LAMBDA,
            dev=None, threads=4, lr=2e-3):
    """items: list of dict(feats=[variant arrays (T5, nin)], y=(T5,) int64 in {0,1,2}, -1 unknown, optional mask_d=True
    (team loss only, as SkillCorner in v4_lib)). Returns {net (cpu, eval), mu, sd, config}."""
    dev = dev or ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = dev == "cuda"
    if not cuda:
        torch.set_num_threads(threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    allX = np.concatenate([it["feats"][0] for it in items])
    mu, sd = np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-6
    for it in items:
        it["Z"] = [torch.tensor(T.prep(f, mu, sd)).to(dev) for f in it["feats"]]
        it["Zm"] = [torch.tensor(T.prep(mirror(f), mu, sd)).to(dev) for f in it["feats"]]
        it["Y"] = torch.tensor(it["y"]).to(dev)
        it["B"] = torch.tensor(V.boundary_target(it["y"])).to(dev)
        it["W"] = torch.tensor(V.short_weights(it["y"], short_w)).to(dev)
    size = np.array([len(it["y"]) for it in items], float)
    swap = torch.tensor(swap_index(nin)).to(dev)
    net = TCN21(nin, 64, 0.2, dil).to(dev)
    ema = [p.detach().clone() for p in net.parameters()]
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    for _ in range(steps):
        xb, yb, bb, wb, mb = [], [], [], [], []
        for _ in range(bs):
            it = items[rng.choice(len(items), p=size / size.sum())]
            v = rng.integers(len(it["Z"]))
            a = rng.integers(0, len(it["Y"]) - crop)
            Z = it["Zm"][v] if rng.random() < 0.5 else it["Z"][v]
            x, y, b, w = Z[a:a + crop], it["Y"][a:a + crop], it["B"][a:a + crop], it["W"][a:a + crop]
            if rng.random() < 0.5:
                x, y = x[:, swap], torch.where((y == 0) | (y == 1), 1 - y, y)
            xb.append(x.T), yb.append(y), bb.append(b), wb.append(w), mb.append(bool(it.get("mask_d", False)))
        xb = torch.stack(xb)
        xb = xb + 0.05 * torch.randn_like(xb)
        yb, bb, wb = torch.stack(yb), torch.stack(bb), torch.stack(wb)
        md = torch.tensor(mb, device=dev)[:, None]
        if cuda:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = net(xb)
        else:
            out = net(xb)
        out = out.float()
        lp3 = F.log_softmax(out[:, :3], 1)
        lp2 = F.log_softmax(out[:, :2], 1)
        yc = yb.clamp(min=0)
        ce = torch.where(md, -lp2.gather(1, yc.clamp(max=1)[:, None])[:, 0], -lp3.gather(1, yc[:, None])[:, 0])
        valid = (yb >= 0) & (~md | (yb < 2))
        loss = (ce * valid * wb).sum() / valid.sum().clamp(min=1)
        if bd_lambda > 0:
            loss = loss + bd_lambda * (F.binary_cross_entropy_with_logits(out[:, 3], bb, reduction="none") * wb).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for e, p in zip(ema, net.parameters()):
                e.lerp_(p.detach(), 1 - V.EMA)
    with torch.no_grad():
        for e, p in zip(ema, net.parameters()):
            p.copy_(e)
    cfg = {"net": {"nin": nin, "c": 64, "p": 0.2, "dil": list(dil)}, "steps": steps, "seed": seed, "crop": crop, "bs": bs,
           "short_w": short_w, "bd_lambda": bd_lambda, "ema": V.EMA, "lr": lr}
    return {"net": net.cpu().eval(), "mu": mu, "sd": sd, "config": cfg}


@torch.no_grad()
def predict21(model, feats, tta=True):
    """-> (lp (T5, 3), b (T5,)) at 5 Hz, CPU float32, 1 thread; mirror TTA."""
    torch.set_num_threads(1)
    outs = []
    for X in ([feats, mirror(feats)] if tta else [feats]):
        z = torch.tensor(T.prep(X, model["mu"], model["sd"])).T[None]
        o = model["net"](z)[0].T
        outs.append((F.log_softmax(o[:, :3], 1).numpy().astype(np.float64), torch.sigmoid(o[:, 3]).numpy().astype(np.float64)))
    return np.mean([o[0] for o in outs], 0), np.mean([o[1] for o in outs], 0)


def save(model, path, extra=None):
    cfg = {**model["config"], **(extra or {}), "torch": torch.__version__}
    torch.save({"state_dict": model["net"].state_dict(), "mu": torch.tensor(model["mu"]), "sd": torch.tensor(model["sd"]),
                "config": __import__("json").dumps(cfg)}, path)


def load(path):
    import json
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = json.loads(ck["config"])
    n = cfg["net"]
    net = TCN21(n["nin"], n["c"], n["p"], tuple(n["dil"]))
    net.load_state_dict(ck["state_dict"])
    return {"net": net.float().eval(), "mu": ck["mu"].numpy(), "sd": ck["sd"].numpy(), "config": cfg}
