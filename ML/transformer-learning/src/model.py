"""把输入嵌入、编码器、解码器和输出层组装成完整 Transformer。"""

import copy

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .decoder import Decoder, DecoderLayer
from .embedding import Embedding, PositionalEncoding
from .encoder import Encoder, EncoderLayer
from .feedforward import FeedForward
from .generator import Generator


class Transformer(nn.Module):
    """标准 Transformer 编码器-解码器模型。"""

    def __init__(
        self,
        source_embed: nn.Module,
        encoder: Encoder,
        target_embed: nn.Module,
        decoder: Decoder,
        generator: Generator,
    ):
        super().__init__()
        self.source_embed = source_embed
        self.encoder = encoder
        self.target_embed = target_embed
        self.decoder = decoder
        self.generator = generator

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        memory = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(decoder_output)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.source_embed(src), src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.decoder(self.target_embed(tgt), memory, src_mask, tgt_mask)


def make_model(
    vocab_size: int = 1000,
    d_model: int = 512,
    h: int = 8,
    d_ff: int = 2048,
    n: int = 6,
    dropout: float = 0.1,
) -> Transformer:
    """按论文默认参数创建一个 Transformer 模型。"""
    c = copy.deepcopy

    src_embed = Embedding(vocab_size, d_model)
    src_pos = PositionalEncoding(d_model, dropout)

    attention = MultiHeadAttention(d_model, h, dropout)
    feed_forward = FeedForward(d_model, d_ff, dropout)
    encoder_layer = EncoderLayer(d_model, attention, feed_forward, dropout)
    encoder = Encoder(encoder_layer, n)

    tgt_embed = c(src_embed)
    tgt_pos = c(src_pos)
    self_attn = c(attention)
    src_attn = c(attention)
    decoder_feed_forward = c(feed_forward)
    decoder_layer = DecoderLayer(
        d_model, self_attn, src_attn, decoder_feed_forward, dropout
    )
    decoder = Decoder(decoder_layer, n)
    generator = Generator(d_model, vocab_size)

    return Transformer(
        nn.Sequential(src_embed, src_pos),
        encoder,
        nn.Sequential(tgt_embed, tgt_pos),
        decoder,
        generator,
    )
