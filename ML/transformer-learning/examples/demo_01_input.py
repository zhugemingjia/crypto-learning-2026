"""步骤 1：观察词嵌入与位置编码的输出形状。"""

import torch

from src import Embedding, PositionalEncoding


def main() -> None:
    vocab_size = 1000
    d_model = 512

    # 模拟 2 个句子，每个句子 4 个 token。
    token_ids = torch.tensor(
        [
            [100, 2, 432, 300],
            [500, 800, 306, 509],
        ]
    )

    embedding = Embedding(vocab_size, d_model)
    embedded = embedding(token_ids)
    print("词嵌入输出形状:", embedded.shape)
    print("期望形状: [batch_size, seq_len, d_model] = [2, 4, 512]")

    positional_encoding = PositionalEncoding(d_model=d_model, dropout=0.1)
    encoded = positional_encoding(embedded)
    print("\n位置编码后形状:", encoded.shape)
    print("前两个位置编码的差异范数:", torch.norm(positional_encoding.pe[0, 1] - positional_encoding.pe[0, 0]).item())


if __name__ == "__main__":
    main()
