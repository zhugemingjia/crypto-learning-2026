"""解码器：包含掩码自注意力、编码器-解码器注意力、前馈网络。"""

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import FeedForward
from .layer_norm import LayerNorm
from .sublayer import SublayerConnection
from .utils import clones


class DecoderLayer(nn.Module):
    """一个解码器层包含三个子层。"""

    def __init__(
        self,
        d_model: int,
        self_attn: MultiHeadAttention,
        src_attn: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(d_model, dropout), 3)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        # 子层 1：掩码自注意力，防止看到未来 token。
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # 子层 2：编码器-解码器注意力，Q 来自解码器，K/V 来自编码器。
        x = self.sublayers[1](
            x, lambda x: self.src_attn(x, memory, memory, src_mask)
        )
        # 子层 3：前馈网络
        x = self.sublayers[2](x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """堆叠 N 个解码器层，并对最终输出做一次层规范化。"""

    def __init__(self, layer: DecoderLayer, n: int):
        super().__init__()
        self.layers = clones(layer, n)
        self.norm = LayerNorm(layer.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
