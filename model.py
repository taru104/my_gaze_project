"""
視線推定用軽量 MLP モデル。
アーキテクチャ: 7D → Polynomial Expansion(36D) → FC(16)+BN+ReLU → FC(16)+BN+ReLU → FC(2)
損失関数: L_total = L_MSE + gamma * L_Angular
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def poly_expand(x: torch.Tensor) -> torch.Tensor:
    """
    degree-2 多項式拡張: x (batch, 7) -> (batch, 36)
    出力順: [1, x1..x7, x1^2..x7^2, x1x2, x1x3, ..., x6x7]
    C(7+2,2) = 36 次元
    """
    n = x.shape[-1]  # 7

    bias  = torch.ones(*x.shape[:-1], 1, device=x.device, dtype=x.dtype)
    lin   = x                                                   # (batch, 7)
    quad  = x ** 2                                              # (batch, 7)
    cross = torch.cat(
        [x[..., i:i+1] * x[..., j:j+1] for i in range(n) for j in range(i+1, n)],
        dim=-1,
    )                                                           # (batch, 21)

    return torch.cat([bias, lin, quad, cross], dim=-1)          # (batch, 36)


class GazeMLP(nn.Module):
    """
    軽量 MLP 視線推定モデル。
    推論遅延目標 < 1ms (CPU).
    """

    def __init__(
        self,
        input_dim:  int = 7,
        hidden_dim: int = 16,
        output_dim: int = 2,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.output_dim = output_dim

        poly_dim = self._calc_poly_dim(input_dim)  # 36

        self.net = nn.Sequential(
            nn.Linear(poly_dim,  hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

        self._init_weights()

    @staticmethod
    def _calc_poly_dim(n: int, degree: int = 2) -> int:
        from math import comb
        return comb(n + degree, degree)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, 7) または (7,) → (batch, 2) または (2,)"""
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        out = self.net(poly_expand(x))
        return out.squeeze(0) if squeeze else out


def angular_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Angular loss: mean arccos(cosine_similarity)
    数値安定性: arccos の引数を (-1+eps, 1-eps) にクランプ
    """
    p = F.normalize(pred,   dim=-1, eps=eps)
    t = F.normalize(target, dim=-1, eps=eps)
    cos_sim = (p * t).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.acos(cos_sim).mean()


def hybrid_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    gamma:  float = 1.0,
) -> torch.Tensor:
    """
    L_total = L_MSE + gamma * L_Angular
    gamma=1.0: 2D座標MSEとAngular Lossのスケールを揃える推奨値 (Q1)
    """
    mse = F.mse_loss(pred, target)
    ang = angular_loss(pred, target)
    return mse + gamma * ang


# ──── numpy ベースの推論用ユーティリティ ────────────────────────────────

def poly_expand_numpy(x: np.ndarray) -> np.ndarray:
    """CPU numpy 版 poly_expand。推論パイプライン用。x: (7,) -> (36,)"""
    n = len(x)
    bias  = [1.0]
    lin   = x.tolist()
    quad  = (x ** 2).tolist()
    cross = [x[i] * x[j] for i in range(n) for j in range(i+1, n)]
    return np.array(bias + lin + quad + cross, dtype=np.float32)


def poly_expand_batch_numpy(X: np.ndarray) -> np.ndarray:
    """X: (N, 7) -> (N, 36)"""
    return np.stack([poly_expand_numpy(x) for x in X])
