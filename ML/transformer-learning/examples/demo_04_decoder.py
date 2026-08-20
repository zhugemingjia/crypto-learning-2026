"""步骤 4：搭建并观察解码器。"""

import copy

import torch

from src import Decoder, DecoderLayer, FeedForward, MultiHeadAttention


def main() -> None:
    torch.manual_seed(0)
    d_model, d_ff, heads, n_layers = 512, 2048, 8, 6

    attention = MultiHeadAttention(d_model, heads)
    self_attn = copy.deepcopy(attention)
    src_attn = copy.deepcopy(attention)
    feed_forward = FeedForward(d_model, d_ff)

    decoder_layer = DecoderLayer(
        d_model=d_model,
        self_attn=self_attn,
        src_attn=src_attn,
        feed_forward=feed_forward,
    )
    decoder = Decoder(decoder_layer, n_layers)

    # 模拟解码器输入，以及编码器输出 memory。
    tgt = torch.randn(2, 4, d_model)
    memory = torch.randn(2, 4, d_model)
    src_mask = torch.zeros(2, 4, 4)
    tgt_mask = torch.zeros(2, 4, 4)

    output = decoder(tgt, memory, src_mask, tgt_mask)

    print("解码器结构:")
    print(decoder)
    print("\n解码器输出形状:", output.shape)
    print("输入与输出是否保持同一形状:", output.shape == tgt.shape)


if __name__ == "__main__":
    main()
