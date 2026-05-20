from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZeroInitConditionAdapter(nn.Module):
    """
    Per-point downstream adapter injected into hidden states.

    The final projection is zero-initialized, so enabling the module keeps the
    initial backbone behavior unchanged and only learns residual corrections
    during fine-tuning.
    """

    def __init__(self, cond_dim: int, hidden_dim: int, model_dim: int):
        super().__init__()
        if cond_dim <= 0 or hidden_dim <= 0 or model_dim <= 0:
            raise ValueError("Adapter dimensions must be positive.")
        self.cond_dim = int(cond_dim)
        self.hidden_dim = int(hidden_dim)
        self.model_dim = int(model_dim)
        self.net = nn.Sequential(
            nn.LayerNorm(self.cond_dim),
            nn.Linear(self.cond_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.model_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden: torch.Tensor, cond_feats: torch.Tensor, return_delta: bool = False):
        if hidden.ndim != 3 or cond_feats.ndim != 3:
            raise ValueError("Condition adapter expects [B, N, C] hidden and condition tensors.")
        if hidden.shape[:2] != cond_feats.shape[:2]:
            raise ValueError("Condition adapter requires matching batch and point dimensions.")
        if cond_feats.shape[-1] != self.cond_dim:
            raise ValueError(f"Expected condition dim {self.cond_dim}, got {cond_feats.shape[-1]}.")
        delta = self.net(cond_feats.to(device=hidden.device, dtype=hidden.dtype))
        out = hidden + delta
        if return_delta:
            return out, delta
        return out


class ZeroInitOutputResidualHead(nn.Module):
    """
    Downstream output residual:
        y = y_backbone + Delta y(hidden, point, fx)

    The final projection is zero-initialized, preserving the exact initial
    behavior of the original model when this head is enabled.
    """

    def __init__(
        self,
        hidden_dim: int,
        point_dim: int,
        feature_dim: int,
        out_dim: int,
        residual_hidden_dim: int,
    ):
        super().__init__()
        if hidden_dim <= 0 or point_dim <= 0 or feature_dim <= 0 or out_dim <= 0 or residual_hidden_dim <= 0:
            raise ValueError("Output residual head dimensions must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.point_dim = int(point_dim)
        self.feature_dim = int(feature_dim)
        self.out_dim = int(out_dim)
        self.input_dim = self.hidden_dim + self.point_dim + self.feature_dim
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, residual_hidden_dim),
            nn.GELU(),
            nn.Linear(residual_hidden_dim, residual_hidden_dim),
            nn.GELU(),
            nn.Linear(residual_hidden_dim, self.out_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        hidden: torch.Tensor,
        point_feats: torch.Tensor,
        feature_feats: torch.Tensor,
        return_delta: bool = False,
    ):
        if hidden.ndim != 3 or point_feats.ndim != 3 or feature_feats.ndim != 3:
            raise ValueError("Output residual head expects [B, N, C] tensors.")
        if hidden.shape[:2] != point_feats.shape[:2] or hidden.shape[:2] != feature_feats.shape[:2]:
            raise ValueError("Output residual head requires matching batch and point dimensions.")
        features = torch.cat(
            [
                hidden,
                point_feats.to(device=hidden.device, dtype=hidden.dtype),
                feature_feats.to(device=hidden.device, dtype=hidden.dtype),
            ],
            dim=-1,
        )
        delta = self.net(features)
        if return_delta:
            return delta
        return delta


class TokenNonLocalBlock(nn.Module):
    """
    Lightweight non-local token mixer with a sampled context set.

    The output projection is zero-initialized, so the block starts as an exact
    identity mapping and can be safely inserted into prompt-side MLPs.
    """

    def __init__(self, hidden_dim: int, reduction: int = 2, context_size: int = 128):
        super().__init__()
        if hidden_dim <= 0 or reduction <= 0 or context_size <= 0:
            raise ValueError("Non-local block dimensions must be positive.")
        self.hidden_dim = int(hidden_dim)
        self.inner_dim = max(1, self.hidden_dim // int(reduction))
        self.context_size = int(context_size)
        self.norm = nn.LayerNorm(self.hidden_dim)
        self.to_q = nn.Linear(self.hidden_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(self.hidden_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(self.hidden_dim, self.inner_dim, bias=False)
        self.to_out = nn.Linear(self.inner_dim, self.hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.to_out.bias)

    def _sample_context(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] <= self.context_size:
            return x
        idx = torch.linspace(0, x.shape[1] - 1, steps=self.context_size, device=x.device).round().long()
        return x.index_select(1, idx)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm(x)
        context = self._sample_context(x_norm)
        q = self.to_q(x_norm)
        k = self.to_k(context)
        v = self.to_v(context)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (self.inner_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        return residual + self.to_out(torch.matmul(attn, v))
