"""编码器：由若干编码器层堆叠而成。"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .layer_norm import LayerNorm
from .sublayer import SublayerConnection
from .utils import clones


class EncoderLayer(nn.Module):
    """一个编码器层包含两个子层：多头自注意力、前馈网络。"""

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(d_model, dropout), 2)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # 子层 1：自注意力
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        # 子层 2：前馈网络
        x = self.sublayers[1](x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """堆叠 N 个编码器层，并对最终输出做一次层规范化。"""

    def __init__(self, layer: EncoderLayer, n: int):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
