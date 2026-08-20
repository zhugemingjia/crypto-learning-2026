"""层规范化（Layer Normalization）。"""

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """对单个样本的特征维度做规范化，适合文本这种变长序列。"""

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        # a 和 b 是可学习参数，让模型在规范化后还能恢复表达能力。
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b
