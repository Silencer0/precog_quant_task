from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class FinMambaConfig:
    lookback: int = 20
    n_levels: int = 2
    hidden_dim: int = 64
    n_heads: int = 4

    # Market-aware sparsification
    tau: float = 0.20  # kappa_t = tau * sigmoid(inception(M_t))
    min_keep_ratio: float = 0.02
    enforce_self_loops: bool = True

    # Loss weights
    ranking_eta: float = 0.10
    gib_lambda: float = 0.10


class InceptionSparsity(nn.Module):
    """Parameter-efficient inception-like block producing a scalar per sample.

    Input: market proxy tensor M of shape [B, 1, L, F].
    Output: scalar logits [B].
    """

    def __init__(self, in_channels: int = 1, mid_channels: int = 8):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.GELU(),
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(1, 3), padding=(0, 1)),
            nn.GELU(),
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 1), padding=(1, 0)),
            nn.GELU(),
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out = nn.Linear(mid_channels * 4, 1)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        x1 = self.pool(self.b1(m)).flatten(1)
        x2 = self.pool(self.b2(m)).flatten(1)
        x3 = self.pool(self.b3(m)).flatten(1)
        x4 = self.pool(self.b4(m)).flatten(1)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return self.out(x).squeeze(-1)


def _topk_edge_mask(
    weights: torch.Tensor, *, keep_ratio: torch.Tensor, enforce_self_loops: bool
) -> torch.Tensor:
    """Build a per-sample boolean edge mask by keeping top edges.

    weights: [B, N, N]
    keep_ratio: [B] in (0,1]
    returns mask: [B, N, N] bool
    """
    b, n, _ = weights.shape
    flat = weights.view(b, n * n)

    k = torch.clamp((keep_ratio * (n * n)).ceil().to(torch.int64), min=1, max=n * n)
    # Build mask by thresholding at per-row kth largest
    # torch.topk supports per-row with variable k only via loop; N=100, B small.
    mask = torch.zeros_like(flat, dtype=torch.bool)
    for i in range(b):
        ki = int(k[i].item())
        vals, idx = torch.topk(flat[i], k=ki, largest=True, sorted=False)
        mask[i, idx] = True

    mask = mask.view(b, n, n)
    if enforce_self_loops:
        eye = torch.eye(n, dtype=torch.bool, device=weights.device).unsqueeze(0)
        mask = mask | eye
    return mask


class MarketAwareGraph(nn.Module):
    """Implements Market-Aware Graph (MAG) from the FinMamba PDF.

    - Computes posterior short-term relationship Q_t from input sequences.
    - Uses a constant prior relationship D.
    - Forms A_t = Q_t @ D.
    - Computes keep ratio kappa_t = tau * sigmoid(Inception(M_t)).
    - Keeps top edges according to A_t.
    """

    def __init__(self, *, feature_dim: int, cfg: FinMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.inception = InceptionSparsity(in_channels=1)
        self.feature_dim = feature_dim

        # Prior D is provided at runtime from training window and registered as buffer.
        self.register_buffer("prior_d", torch.empty(0), persistent=False)

    def set_prior(self, prior_d: torch.Tensor) -> None:
        # prior_d: [N,N]
        self.prior_d = prior_d

    def forward(self, x_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_seq: [B,L,N,F]
        b, l, n, f = x_seq.shape
        if self.prior_d.numel() == 0:
            raise RuntimeError("prior_d not set; call set_prior() before forward")

        # Market index proxy M_t: mean over stocks, shape [B, L, F]
        m = x_seq.mean(dim=2)  # [B,L,F]
        m2 = m.unsqueeze(1)  # [B,1,L,F]
        sparsity_logits = self.inception(m2)  # [B]
        keep_ratio = self.cfg.tau * torch.sigmoid(sparsity_logits)
        keep_ratio = torch.clamp(keep_ratio, min=self.cfg.min_keep_ratio, max=1.0)

        # Posterior Q_t: cosine similarity of last-step features (simple, differentiable)
        x_last = x_seq[:, -1, :, :]  # [B,N,F]
        x_norm = torch.nn.functional.normalize(x_last, dim=-1)
        q = torch.matmul(x_norm, x_norm.transpose(1, 2))  # [B,N,N]

        # A_t = Q_t @ D
        d = self.prior_d
        if d.dim() == 2:
            d = d.unsqueeze(0).expand(b, -1, -1)
        a = torch.matmul(q, d)  # [B,N,N]

        edge_mask = _topk_edge_mask(
            a, keep_ratio=keep_ratio, enforce_self_loops=self.cfg.enforce_self_loops
        )
        return edge_mask, a


class GraphAttentionAggregator(nn.Module):
    """Multi-head attention aggregation over pruned graph (Eq. 5-6).

    Inputs:
      - s_seq: [B,N,L,F] node sequences
      - edge_mask: [B,N,N] bool
    Outputs:
      - z_seq: [B,N,L,F] neighbor representation
    """

    def __init__(self, *, f: int, n_heads: int):
        super().__init__()
        self.f = f
        self.n_heads = n_heads
        self.w = nn.Linear(f, f, bias=False)
        self.a = nn.Parameter(torch.empty(n_heads, 2 * f))
        nn.init.xavier_uniform_(self.a)
        self.leaky = nn.LeakyReLU(0.2)
        self.act = nn.GELU()

    def forward(self, s_seq: torch.Tensor, edge_mask: torch.Tensor) -> torch.Tensor:
        b, n, l, f = s_seq.shape
        s_last = s_seq[:, :, -1, :]  # [B,N,F]
        wh = self.w(s_last)  # [B,N,F]

        # Dense attention for N<=100
        wh_i = wh.unsqueeze(2).expand(b, n, n, f)
        wh_j = wh.unsqueeze(1).expand(b, n, n, f)
        cat = torch.cat([wh_i, wh_j], dim=-1)  # [B,N,N,2F]

        # head-wise scores
        scores = []
        for k in range(self.n_heads):
            ak = self.a[k].view(1, 1, 1, -1)
            e = (cat * ak).sum(dim=-1)  # [B,N,N]
            e = self.leaky(e)
            # mask
            e = e.masked_fill(~edge_mask, float("-inf"))
            alpha = torch.softmax(e, dim=-1)  # [B,N,N]
            scores.append(alpha)

        alpha = torch.stack(scores, dim=1).mean(dim=1)  # average heads: [B,N,N]

        # Aggregate full sequences with same alpha
        z_seq = torch.matmul(alpha, s_seq.view(b, n, l * f)).view(b, n, l, f)
        return self.act(z_seq)


class SelectiveSSMBlock(nn.Module):
    """A simple input-dependent SSM block (Mamba-style) in pure PyTorch.

    Implements per-timestep:
      h_t = A_t * h_{t-1} + B_t x_t
      o_t = C_t h_t
    where A_t, B_t, C_t are produced from x_t.
    """

    def __init__(self, *, d_in: int, d_state: int):
        super().__init__()
        self.d_in = d_in
        self.d_state = d_state
        self.param = nn.Linear(d_in, 3 * d_state)
        self.out = nn.Linear(d_state, d_state)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,N,L,Din]
        b, n, l, din = x.shape
        h = torch.zeros((b, n, self.d_state), device=x.device, dtype=x.dtype)
        outs = []
        for t in range(l):
            xt = x[:, :, t, :]
            p = self.param(xt)
            a, b_, c = torch.chunk(p, 3, dim=-1)
            # constrain A to (0,1) for stability
            a = torch.sigmoid(a)
            h = a * h + b_ * xt.mean(dim=-1, keepdim=True)  # cheap interaction
            ot = c * h
            outs.append(self.out(self.act(ot)).unsqueeze(2))
        return torch.cat(outs, dim=2)  # [B,N,L,D]


class MultiLevelMamba(nn.Module):
    """Multi-Level Mamba (Eq. 7-8) using SelectiveSSMBlock per level."""

    def __init__(self, *, d_in: int, cfg: FinMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.level_proj = nn.ModuleList(
            [nn.Linear(d_in, d_in) for _ in range(cfg.n_levels)]
        )
        self.ssm = nn.ModuleList(
            [
                SelectiveSSMBlock(d_in=d_in, d_state=cfg.hidden_dim)
                for _ in range(cfg.n_levels)
            ]
        )
        self.relevel = nn.ModuleList(
            [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.n_levels)]
        )
        self.final = nn.Linear(cfg.hidden_dim * cfg.n_levels, 1)

    def forward(self, p_seq: torch.Tensor) -> torch.Tensor:
        # p_seq: [B,N,L,Din]
        xs = []
        for i in range(self.cfg.n_levels):
            x = self.level_proj[i](p_seq)
            o = self.ssm[i](x)  # [B,N,L,H]
            # last timestep
            last = o[:, :, -1, :] + x[:, :, -1, :].mean(dim=-1, keepdim=True)
            last = self.relevel[i](last)
            xs.append(last)
        h = torch.cat(xs, dim=-1)
        y = self.final(h).squeeze(-1)
        return y


class FinMamba(nn.Module):
    """FinMamba model composed of MAG + GAT aggregation + MLM."""

    def __init__(self, *, feature_dim: int, cfg: FinMambaConfig):
        super().__init__()
        self.cfg = cfg
        self.mag = MarketAwareGraph(feature_dim=feature_dim, cfg=cfg)
        self.gat = GraphAttentionAggregator(f=feature_dim, n_heads=cfg.n_heads)
        self.mlm = MultiLevelMamba(d_in=2 * feature_dim, cfg=cfg)

    def set_prior(self, prior_d: torch.Tensor) -> None:
        self.mag.set_prior(prior_d)

    def forward(
        self, s_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # s_seq: [B,L,N,F]
        b, l, n, f = s_seq.shape
        edge_mask, a = self.mag(s_seq)
        s_seq_n = s_seq.permute(0, 2, 1, 3).contiguous()  # [B,N,L,F]
        z_seq = self.gat(s_seq_n, edge_mask)  # [B,N,L,F]
        p_seq = torch.cat([s_seq_n, z_seq], dim=-1)  # [B,N,L,2F]
        y = self.mlm(p_seq)  # [B,N]
        return y, z_seq, s_seq_n


def finmamba_loss(
    *,
    y_pred: torch.Tensor,
    r_true: torch.Tensor,
    z_seq: torch.Tensor,
    s_seq: torch.Tensor,
    cfg: FinMambaConfig,
) -> torch.Tensor:
    """Loss Eq. 9-11: pointwise regression + pairwise ranking + GIB."""
    # y_pred, r_true: [B,N]
    mse = torch.mean((y_pred - r_true) ** 2)

    # Pairwise hinge ranking
    yd = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)  # [B,N,N]
    rd = r_true.unsqueeze(2) - r_true.unsqueeze(1)
    hinge = torch.relu(-(yd * rd))
    rank = hinge.mean()

    # GIB loss: compare neighbor embedding z vs original s
    # mean/var over (L,F)
    z = z_seq
    s = s_seq
    z_mean = z.mean(dim=(2, 3))
    s_mean = s.mean(dim=(2, 3))
    z_var = z.var(dim=(2, 3), unbiased=False)
    s_var = s.var(dim=(2, 3), unbiased=False)
    gib = ((z_mean - s_mean) ** 2 / (z_var + s_var + 1e-8)).mean()

    return mse + cfg.ranking_eta * rank + cfg.gib_lambda * gib
