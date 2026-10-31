"""Ball Radar style network: per-frame permutation-invariant set encoder, then a bidirectional LSTM over time.

Per frame the track slots form a set. Each slot token = (x, y, vx, vy, speed, offset and distance to the
observed ball, ball-seen flag) + a team embedding. Two self-attention blocks (no positional encoding, so
the encoder is permutation invariant over slots; empty slots are masked; a learned CLS token is always
present so an empty frame is still defined) are followed by pooling: CLS, mean over home slots, mean over
away slots. The observed ball (x, y, seen flag; zeros when unseen) joins the pooled vector. A 2-layer
BiLSTM runs over a window of frames. Heads: ball (x, y) in metres and team in possession (H, A, D).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SX, SY = 52.5, 34.0  # pitch half sizes, metres


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, keep):  # x (N, S, d); keep (N, S) bool, True = real token
        N, S, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(N, S, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None, :])
        x = x + self.proj(a.transpose(1, 2).reshape(N, S, d))
        return x + self.mlp(self.n2(x))


class BallRadar(nn.Module):
    def __init__(self, d=64, heads=4, layers=2, hidden=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.cfg = dict(d=d, heads=heads, layers=layers, hidden=hidden, lstm_layers=lstm_layers, dropout=dropout)
        self.inp = nn.Sequential(nn.Linear(9, d), nn.GELU(), nn.Linear(d, d))
        self.team = nn.Embedding(2, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.out_norm = nn.LayerNorm(d)
        self.frame = nn.Sequential(nn.Linear(3 * d + 6, hidden), nn.GELU(), nn.Dropout(dropout))
        self.lstm = nn.LSTM(hidden, hidden, lstm_layers, batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.ball_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 2))
        self.team_head = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.Linear(hidden, 3))

    def forward(self, slot, mask, team, ball, seen):
        """slot (B,T,S,4) mask (B,T,S) team (S,) or (B,S) long ball (B,T,2) seen (B,T) -> ball (B,T,2) metres, logits (B,T,3)."""
        B, T, S, _ = slot.shape
        if team.dim() == 1:
            team = team[None].expand(B, S)
        pos, vel = slot[..., :2], slot[..., 2:]
        sc = torch.tensor([SX, SY], device=slot.device)
        sn = seen.float()[:, :, None]
        rel = (pos - ball[:, :, None, :]) * sn[..., None]
        dist = rel.norm(dim=-1, keepdim=True)
        feats = torch.cat([pos / sc, vel / 8.0, vel.norm(dim=-1, keepdim=True) / 8.0, rel / 20.0, dist / 20.0,
                           sn[..., None].expand(B, T, S, 1)], dim=-1) * mask[..., None]
        x = self.inp(feats) + self.team(team)[:, None]
        x = x.reshape(B * T, S, -1)
        keep = torch.cat([torch.ones(B * T, 1, dtype=torch.bool, device=x.device), mask.reshape(B * T, S)], 1)
        x = torch.cat([self.cls.expand(B * T, 1, -1), x], 1)
        for blk in self.blocks:
            x = blk(x, keep)
        x = self.out_norm(x)
        pl, tm = x[:, 1:], team[:, None, :].expand(B, T, S).reshape(B * T, S, 1)
        m = mask.reshape(B * T, S, 1).float()
        mh, ma = m * (tm == 0), m * (tm == 1)
        mean_h = (pl * mh).sum(1) / mh.sum(1).clamp(min=1.0)
        mean_a = (pl * ma).sum(1) / ma.sum(1).clamp(min=1.0)
        z = torch.cat([x[:, 0], mean_h, mean_a], dim=-1).reshape(B, T, -1)
        # explicit nearest-player-per-team distances to the observed ball (large + flag when unseen)
        tt = team[:, None, :].expand(B, T, S)
        far = 30.0
        ok = mask & seen[:, :, None]
        dd = torch.where(ok, dist[..., 0], torch.full_like(dist[..., 0], far))
        dh = torch.where(tt == 0, dd, torch.full_like(dd, far)).amin(-1)
        da = torch.where(tt == 1, dd, torch.full_like(dd, far)).amin(-1)
        near = torch.stack([dh / 10.0, da / 10.0, ((da - dh) / 10.0).clamp(-3, 3) * sn[..., 0]], dim=-1)
        z = self.frame(torch.cat([z, ball / sc * sn, sn, near], dim=-1))
        h, _ = self.lstm(z)
        return self.ball_head(h) * sc, self.team_head(h)
