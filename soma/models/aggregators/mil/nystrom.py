"""Nystrom attention and transformer layer.

Adapted from lucidrains/nystrom-attention.

All einops operations replaced with standard PyTorch ops.
"""

from __future__ import annotations

from math import ceil

import torch
from torch import Tensor, nn


def moore_penrose_iter_pinv(x: Tensor, iters: int = 6) -> Tensor:
    """Iterative Moore-Penrose pseudo-inverse approximation.

    Args:
        x: Input matrix, shape (..., I, J).
        iters: Number of iterations.

    Returns:
        Approximate pseudo-inverse, shape (..., J, I).
    """
    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = x.transpose(-2, -1) / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=x.device).unsqueeze(0)

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttention(nn.Module):
    """Nystrom self-attention (Xiong et al., 2021).

    Approximates full self-attention using landmark-based decomposition,
    reducing complexity from O(N^2) to O(N*M) where M is the number of
    landmarks.

    Args:
        dim: Input/output dimension. Must be divisible by n_heads.
        n_heads: Number of attention heads.
        n_landmarks: Number of landmarks for approximation.
        pinv_iterations: Iterations for pseudo-inverse approximation.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        n_landmarks: int = 256,
        pinv_iterations: int = 6,
    ) -> None:
        super().__init__()
        assert dim % n_heads == 0, "dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_landmarks = n_landmarks
        self.pinv_iterations = pinv_iterations
        self.scale = self.head_dim**-0.5
        self.eps = 1e-8

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)

    def forward(
        self, x: Tensor, mask: Tensor | None = None, return_att: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            x: Input, shape (B, L, D).
            mask: Boolean mask, shape (B, L). True = valid.
            return_att: If True, also return attention weights.

        Returns:
            Output tensor (B, L, D), and optionally attention (B, H, L, L).
        """
        B, seq_len, _ = x.shape

        # Pad so sequence is divisible by n_landmarks
        remainder = seq_len % self.n_landmarks
        if remainder > 0:
            padding = self.n_landmarks - remainder
            x = nn.functional.pad(x, (0, 0, padding, 0), value=0)
            if mask is not None:
                mask = nn.functional.pad(mask, (padding, 0), value=False)

        new_seq_len = x.size(1)

        # QKV projection and reshape to multi-head
        qkv = self.qkv(x)  # (B, L', 3*D)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape: (B, L', D) → (B, H, L', head_dim)
        q = q.view(B, new_seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, new_seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, new_seq_len, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # Apply mask
        if mask is not None:
            mask_4d = mask[:, None, :, None]  # (B, 1, L', 1)
            q = q * mask_4d
            k = k * mask_4d
            v = v * mask_4d

        q = q * self.scale

        # Generate landmarks via sum-reduction
        lm = ceil(new_seq_len / self.n_landmarks)
        # Reshape to (B, H, n_landmarks, lm, head_dim) then sum over lm
        q_landmarks = q.reshape(B, self.n_heads, self.n_landmarks, lm, self.head_dim).sum(dim=3)
        k_landmarks = k.reshape(B, self.n_heads, self.n_landmarks, lm, self.head_dim).sum(dim=3)

        # Compute divisor (handle masking)
        divisor: float | Tensor = lm
        if mask is not None:
            # (B, 1, L') → (B, 1, n_landmarks, lm) → sum → (B, 1, n_landmarks)
            mask_reshaped = mask[:, None, :].float().reshape(B, 1, self.n_landmarks, lm)
            mask_landmarks_sum = mask_reshaped.sum(dim=3)  # (B, 1, n_landmarks)
            divisor = mask_landmarks_sum.unsqueeze(-1) + self.eps  # (B, 1, n_landmarks, 1)
            mask_landmarks = mask_landmarks_sum > 0  # (B, 1, n_landmarks)

        q_landmarks = q_landmarks / divisor
        k_landmarks = k_landmarks / divisor

        # Three similarity matrices
        sim1 = torch.einsum("bhid,bhjd->bhij", q, k_landmarks)  # (B, H, L', M)
        sim2 = torch.einsum("bhid,bhjd->bhij", q_landmarks, k_landmarks)  # (B, H, M, M)
        sim3 = torch.einsum("bhid,bhjd->bhij", q_landmarks, k)  # (B, H, M, L')

        # Masking
        if mask is not None:
            mask_value = -torch.finfo(q.dtype).max
            # mask: (B, 1, L'), mask_landmarks: (B, 1, M)
            mask_seq = mask[:, None, :]  # (B, 1, L')
            sim1 = sim1.masked_fill(~(mask_seq[:, :, :, None] * mask_landmarks[:, :, None, :]), mask_value)
            sim2 = sim2.masked_fill(~(mask_landmarks[:, :, :, None] * mask_landmarks[:, :, None, :]), mask_value)
            sim3 = sim3.masked_fill(~(mask_landmarks[:, :, :, None] * mask_seq[:, :, None, :]), mask_value)

        attn1 = sim1.softmax(dim=-1)
        attn2 = sim2.softmax(dim=-1)
        attn3 = sim3.softmax(dim=-1)

        attn2_inv = moore_penrose_iter_pinv(attn2, self.pinv_iterations)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)  # (B, H, L', head_dim)

        # Merge heads: (B, H, L', head_dim) → (B, L', D)
        out = out.permute(0, 2, 1, 3).reshape(B, new_seq_len, -1)
        out = self.out_proj(out)

        # Remove padding
        out = out[:, -seq_len:]

        if return_att:
            attn = attn1 @ attn2_inv @ attn3  # (B, H, L', L')
            attn = attn[:, :, -seq_len:, -seq_len:]
            return out, attn
        return out


class NystromTransformerLayer(nn.Module):
    """Pre-norm transformer layer with Nystrom attention.

    Computes: X' = X + Attn(LayerNorm(X)), optionally followed by
    X'' = X' + MLP(LayerNorm(X')).

    Args:
        dim: Input/output dimension.
        n_heads: Number of attention heads.
        n_landmarks: Number of landmarks for Nystrom attention.
        pinv_iterations: Pseudo-inverse iterations.
        dropout: Dropout rate.
        use_mlp: Whether to include a feed-forward MLP block.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        n_landmarks: int = 256,
        pinv_iterations: int = 6,
        dropout: float = 0.0,
        use_mlp: bool = False,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NystromAttention(
            dim=dim,
            n_heads=n_heads,
            n_landmarks=n_landmarks,
            pinv_iterations=pinv_iterations,
        )
        self.use_mlp = use_mlp
        if use_mlp:
            self.norm2 = nn.LayerNorm(dim)
            self.mlp = nn.Sequential(
                nn.Linear(dim, 4 * dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * dim, dim),
                nn.Dropout(dropout),
            )

    def forward(
        self, X: Tensor, mask: Tensor | None = None, return_att: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass.

        Args:
            X: Input, shape (B, L, D).
            mask: Boolean mask, shape (B, L).
            return_att: If True, also return attention weights.

        Returns:
            Output (B, L, D), and optionally attention (B, H, L, L).
        """
        attn_out = self.attn(self.norm1(X), mask=mask, return_att=return_att)
        if return_att:
            Y, att = attn_out
        else:
            Y = attn_out

        Y = X + Y

        if self.use_mlp:
            Y = Y + self.mlp(self.norm2(Y))

        if return_att:
            return Y, att
        return Y
