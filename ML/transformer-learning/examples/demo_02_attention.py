"""步骤 2：观察单头注意力和多头注意力的计算过程。"""

import torch

from src import MultiHeadAttention, scaled_dot_product_attention


def main() -> None:
    torch.manual_seed(0)

    batch_size, seq_len, d_model = 2, 4, 512
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)

    output, weights = scaled_dot_product_attention(query, key, value)
    print("单头注意力输出形状:", output.shape)
    print("注意力权重形状:", weights.shape)
    print("每行注意力权重之和:", weights.sum(-1))

    # 用全 0 掩码测试：所有位置都被屏蔽后，softmax 会变成均匀分布。
    mask = torch.zeros(batch_size, seq_len, seq_len)
    masked_output, masked_weights = scaled_dot_product_attention(
        query, key, value, mask=mask
    )
    print("\n全 0 掩码下权重第一行:", masked_weights[0, 0])

    multihead = MultiHeadAttention(d_model=d_model, h=8)
    multihead_output = multihead(query, key, value, mask=None)
    print("\n多头注意力输出形状:", multihead_output.shape)
    print("多头注意力矩阵形状:", multihead.attn.shape)


if __name__ == "__main__":
    main()
