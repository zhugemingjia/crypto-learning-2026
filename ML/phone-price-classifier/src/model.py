"""
模型定义：3 层全连接网络 + Dropout
"""

import torch.nn as nn


class PhonePriceClassifier(nn.Module):
    """
    手机价格分类网络

    结构:
        Linear(20, 256) → ReLU → Dropout(0.3)
        Linear(256, 512) → ReLU → Dropout(0.3)
        Linear(512, 256) → ReLU → Dropout(0.3)
        Linear(256, num_classes)   ← 输出 logits（不接 Softmax，CrossEntropyLoss 内部包含）
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)
