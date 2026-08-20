"""输出层：把解码器向量转换为词表上的对数概率。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """线性映射 + log_softmax。"""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.linear(x), dim=-1)
