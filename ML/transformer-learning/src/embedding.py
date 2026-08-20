"""输入部分：词嵌入与位置编码。"""

import math

import torch
import torch.nn as nn


class Embedding(nn.Module):
    """把 token 索引映射为稠密向量。

    参数：
        vocab_size: 词汇表大小。
        d_model: 词向量维度，Transformer 论文中通常是 512。
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 乘以 sqrt(d_model) 是为了让嵌入向量的尺度大致稳定，
        # 避免它被后续位置编码的幅度完全淹没。
        return self.embed(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """使用正弦/余弦函数给每个位置一个唯一编码。

    Transformer 没有 RNN 那样的循环结构，所以需要显式把位置信息加进去。
    位置编码是固定计算出来的，不需要参与梯度更新。
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 60):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # 偶数位置使用 sin，奇数位置使用 cos。
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 升到 [1, max_len, d_model]，方便与 [batch_size, seq_len, d_model] 广播相加。
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
