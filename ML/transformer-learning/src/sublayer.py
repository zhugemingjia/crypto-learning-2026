"""残差连接 + 层规范化。"""

import torch
import torch.nn as nn

from .layer_norm import LayerNorm


class SublayerConnection(nn.Module):
    """把“子层输出”和“子层输入”相加，再做规范化。

    残差连接让梯度可以直接回流到浅层，是 Transformer 能堆叠得很深的关键。
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        # 论文原始写法：LayerNorm(x + Sublayer(x))
        return self.norm(x + self.dropout(sublayer(x)))
