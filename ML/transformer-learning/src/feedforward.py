"""前馈全连接层。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """每个位置独立地先升维、再降维。

    论文中使用 d_model -> d_ff -> d_model，中间加 ReLU。
    这一步让注意力捕获到上下文关系后，再对每个 token 做非线性特征加工。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))
