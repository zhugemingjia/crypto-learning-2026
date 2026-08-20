"""注意力机制：从单头缩点积注意力到多头注意力。"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import clones


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Dropout | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """计算注意力输出和注意力权重矩阵。

    核心公式：
        Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # 把不允许关注的位置设为很小的负数，softmax 后权重接近 0。
        scores = scores.masked_fill(mask == 0, -1e9)

    attn_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn_weights = dropout(attn_weights)

    return torch.matmul(attn_weights, value), attn_weights


class MultiHeadAttention(nn.Module):
    """把输入拆成多个“头”，并行做注意力，再融合结果。

    这样模型可以从不同子空间同时关注句子里的不同关系。
    """

    def __init__(self, d_model: int, h: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % h == 0, "d_model 必须能被头数 h 整除"

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        # 前三个线性层分别生成 Q、K、V，最后一个线性层融合多头结果。
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            # [batch, seq, seq] -> [batch, 1, seq, seq]，方便按头数广播。
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)
        query, key, value = [
            linear(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = scaled_dot_product_attention(
            query, key, value, mask, self.dropout
        )

        # [batch, h, seq, d_k] -> [batch, seq, h, d_k] -> [batch, seq, d_model]
        x = x.transpose(1, 2).contiguous().view(
            batch_size, -1, self.h * self.d_k
        )
        return self.linears[-1](x)
