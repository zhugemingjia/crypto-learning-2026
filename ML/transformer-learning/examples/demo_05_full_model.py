"""步骤 5：运行完整 Transformer 的前向传播。"""

import torch

from src import make_model


def main() -> None:
    torch.manual_seed(0)

    model = make_model()
    print("完整 Transformer 模型结构:")
    print(model)

    # 使用论文默认词表 1000，所以 token id 要小于 1000。
    src = torch.tensor([[100, 2, 432, 300], [500, 800, 306, 509]])
    tgt = torch.tensor([[3, 2, 4, 5], [2, 4, 1, 5]])
    src_mask = torch.zeros(2, 4, 4)
    tgt_mask = torch.zeros(2, 4, 4)

    output = model(src, tgt, src_mask, tgt_mask)
    print("\n完整 Transformer 输出形状:", output.shape)
    print("期望形状: [batch_size, seq_len, vocab_size] = [2, 4, 1000]")
    print("输出是对数概率，可用来计算损失或取 argmax 得到预测词。")


if __name__ == "__main__":
    main()
