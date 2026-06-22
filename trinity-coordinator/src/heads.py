"""Coordinator head architectures from the paper (Appendix A.4).

The paper studies 4 head architectures:
    Linear          (d_h * n_a params, n_a = L+3 = 10)
    Low-rank        (U: r*d_h, V: n_a*r, ELU, fixed sigma; r=14)
    Sparse          (W: n_a*d_h, alpha: d_h, top-k with Gumbel noise)
    Block-diagonal  (B blocks, each block couples a subset of h to a subset of z)

For the Block-diagonal-10 head (one block per output logit), the projection
matrix is a vertical stacking: z_j = w_j^T h_j, where h_j is a slice of h
allocated to block j.

We also expose a generic 2-layer MLP for comparison (not in the paper, but
useful as a sanity baseline on a small feature space).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HeadConfig:
    in_dim: int            # d_h
    n_outputs: int         # n_a = L + 3
    hidden: int = 32       # only used by MLP and Low-rank
    kind: str = "linear"   # "linear" | "low_rank" | "sparse" | "block_diag" | "mlp"
    n_blocks: int = 10     # for block_diag
    sparsity_logit: float = 4.6  # sigmoid(4.6) ≈ 0.99 → keep ~99% dims (sparse)


# ------------------------------------------------------------------
# Linear head (paper Eq. 5):  z = W h, W in R^{n_a x d_h}
# ------------------------------------------------------------------

class LinearHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        self.W = nn.Parameter(torch.empty(cfg.n_outputs, cfg.in_dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.W.T

    def num_parameters(self) -> int:
        return self.W.numel()


# ------------------------------------------------------------------
# Low-rank head (paper Eq. 6-7):  u = ELU(U h),  z = V u * sigma
# ------------------------------------------------------------------

class LowRankHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        r = cfg.hidden
        d_h, n_a = cfg.in_dim, cfg.n_outputs
        self.U = nn.Parameter(torch.empty(r, d_h))
        self.V = nn.Parameter(torch.empty(n_a, r))
        # Xavier-uniform with paper's adaptive gains
        nn.init.uniform_(self.U, -math.sqrt(6.0 / (d_h + r)), math.sqrt(6.0 / (d_h + r)))
        nn.init.uniform_(self.V, -math.sqrt(18.0 / (r + n_a)), math.sqrt(18.0 / (r + n_a)))
        # fixed non-trainable scale
        self.register_buffer("sigma", torch.tensor(1.0))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        u = F.elu(h @ self.U.T)
        return (u @ self.V.T) * self.sigma

    def num_parameters(self) -> int:
        return self.U.numel() + self.V.numel()


# ------------------------------------------------------------------
# Sparse head (paper Eq. 9-11):  z = W (h * alpha)
# alpha is top-k selected via Gumbel noise during training
# ------------------------------------------------------------------

class SparseHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        d_h, n_a = cfg.in_dim, cfg.n_outputs
        self.W = nn.Parameter(torch.empty(n_a, d_h))
        self.scores = nn.Parameter(torch.zeros(d_h))   # s
        self.rho = nn.Parameter(torch.tensor(cfg.sparsity_logit))  # sparsity logit
        self.tau = 5.0  # Gumbel temperature (in [1.0, 20.0] per paper)
        nn.init.xavier_uniform_(self.W)

    def _alpha_soft(self, training: bool) -> torch.Tensor:
        d_h = self.cfg.in_dim
        s = self.scores
        if training:
            g = -torch.log(-torch.log(torch.rand_like(s).clamp_min(1e-20) + 1e-20) + 1e-20)
            tilde_s = (s + g) / self.tau
        else:
            tilde_s = s
        k = max(1, int(d_h * (1 - torch.sigmoid(self.rho)).item()))
        # soft top-k: keep k largest
        topk_vals, topk_idx = torch.topk(tilde_s, k)
        mask = torch.zeros_like(s)
        mask[topk_idx] = 1.0
        # at training time use soft mask
        return mask

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        alpha = self._alpha_soft(self.training)
        return (h * alpha) @ self.W.T

    def num_parameters(self) -> int:
        return self.W.numel() + self.scores.numel() + self.rho.numel()


# ------------------------------------------------------------------
# Block-diagonal head (paper Eq. 12):  W is block-diagonal, h split into B slices
# ------------------------------------------------------------------

class BlockDiagHead(nn.Module):
    """Block-diagonal projection: z_j = w_j^T h_j, with hidden dim split into B blocks.

    When n_blocks == n_outputs (paper's block-diagonal-10 with n_a=10), each
    output has its own block and is fully independent.
    """

    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        d_h, n_a, B = cfg.in_dim, cfg.n_outputs, cfg.n_blocks
        # distribute d_h as evenly as possible: first (d_h mod B) blocks get +1
        sizes = [d_h // B] * B
        for i in range(d_h % B):
            sizes[i] += 1
        # ensure total = d_h
        assert sum(sizes) == d_h, f"{sum(sizes)} != {d_h}"
        self.sizes = sizes
        # number of outputs per block: split n_a as evenly as possible
        out_sizes = [n_a // B] * B
        for i in range(n_a % B):
            out_sizes[i] += 1
        assert sum(out_sizes) == n_a, f"{sum(out_sizes)} != {n_a}"
        self.out_sizes = out_sizes

        self.blocks = nn.ParameterList()
        for hi, ai in zip(sizes, out_sizes):
            W = nn.Parameter(torch.empty(ai, hi))
            nn.init.xavier_uniform_(W)
            self.blocks.append(W)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # split h into B contiguous slices
        outs = []
        offset = 0
        for W, hi in zip(self.blocks, self.sizes):
            h_block = h[..., offset:offset + hi]
            outs.append(h_block @ W.T)
            offset += hi
        return torch.cat(outs, dim=-1)

    def num_parameters(self) -> int:
        return sum(W.numel() for W in self.blocks)


# ------------------------------------------------------------------
# 2-layer MLP (not in paper, but useful for the toy feature space)
# ------------------------------------------------------------------

class MLPHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        self.body = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden),
            nn.GELU(),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        self.out = nn.Linear(cfg.hidden, cfg.n_outputs)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.out(self.body(h))

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def make_head(cfg: HeadConfig) -> nn.Module:
    k = cfg.kind.lower()
    if k == "linear":
        return LinearHead(cfg)
    if k == "low_rank":
        return LowRankHead(cfg)
    if k == "sparse":
        return SparseHead(cfg)
    if k in ("block_diag", "block-diag", "block_diagonal"):
        return BlockDiagHead(cfg)
    if k in ("mlp", "mlp_head"):
        return MLPHead(cfg)
    raise ValueError(f"unknown head kind: {cfg.kind}")
