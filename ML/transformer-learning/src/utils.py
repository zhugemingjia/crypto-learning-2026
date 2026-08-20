"""通用工具函数。"""

import copy

import torch.nn as nn


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """深拷贝一个模块 n 次，返回 nn.ModuleList。

    深拷贝意味着每个副本拥有独立的参数，不会共享权重。
    这个函数在编码器、解码器和多头注意力的线性层中都会反复用到。
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
