"""步骤 3：搭建并观察编码器。"""

import torch

from src import Encoder, EncoderLayer, FeedForward, MultiHeadAttention


def main() -> None:
    torch.manual_seed(0)
    d_model, d_ff, heads, n_layers = 512, 2048, 8, 6

    attention = MultiHeadAttention(d_model, heads)
    feed_forward = FeedForward(d_model, d_ff)
    encoder_layer = EncoderLayer(d_model, attention, feed_forward)
    encoder = Encoder(encoder_layer, n_layers)

    x = torch.randn(2, 4, d_model)
    mask = torch.zeros(2, 4, 4)
    output = encoder(x, mask)

    print("编码器结构:")
    print(encoder)
    print("\n编码器输出形状:", output.shape)
    print("输入与输出是否保持同一形状:", output.shape == x.shape)


if __name__ == "__main__":
    main()
