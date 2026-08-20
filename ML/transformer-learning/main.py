"""项目主入口：构建完整 Transformer 并做一次示例前向传播。

运行方式：
    python main.py
"""

import torch

from src import make_model


def main() -> None:
    torch.manual_seed(0)
    model = make_model()

    print("=" * 70)
    print("Transformer 从零实现：完整前向传播演示")
    print("=" * 70)
    print(model)

    src = torch.tensor([[100, 2, 432, 300], [500, 800, 306, 509]])
    tgt = torch.tensor([[3, 2, 4, 5], [2, 4, 1, 5]])
    src_mask = torch.zeros(2, 4, 4)
    tgt_mask = torch.zeros(2, 4, 4)

    output = model(src, tgt, src_mask, tgt_mask)
    print("\n输出形状:", tuple(output.shape))
    print("输出示例（对数概率）:", output[0, 0, :5])
    print("\n建议按顺序运行 examples 下的 5 个演示脚本逐步理解。")


if __name__ == "__main__":
    main()
