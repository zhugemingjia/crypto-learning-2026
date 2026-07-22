"""
模型定义模块

- My_RNN:  基础循环神经网络
- My_LSTM: 长短期记忆网络
- My_GRU:  门控循环单元

三个模型 API 设计一致，方便对比实验。
"""

import torch
import torch.nn as nn


class My_RNN(nn.Module):
    """RNN 模型 —— 基线

    RNN(input_size, hidden_size) → Linear → LogSoftmax
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size, hidden_size, n_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """前向传播

        Args:
            input:  [seq_len, input_size]
            hidden: [n_layers, 1, hidden_size]

        Returns:
            output: [1, output_size]  对数概率
            hidden: [n_layers, 1, hidden_size]
        """
        input = input.unsqueeze(1)          # [seq_len, 1, input_size]
        output, hidden = self.rnn(input, hidden)
        tmp_output = output[-1]              # 取最后一个时间步
        tmp_output = self.linear(tmp_output)
        return self.softmax(tmp_output), hidden

    def init_hidden(self):
        """初始化隐藏状态为零"""
        return torch.zeros(self.n_layers, 1, self.hidden_size)


class My_LSTM(nn.Module):
    """LSTM 模型

    LSTM(input_size, hidden_size) → Linear → LogSoftmax
    比 RNN 多了 cell state，能更好地捕捉长距离依赖。
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, cell):
        """前向传播

        Args:
            input:  [seq_len, input_size]
            hidden: [n_layers, 1, hidden_size]
            cell:   [n_layers, 1, hidden_size]

        Returns:
            output: [1, output_size]
            hidden, cell: 更新后的状态
        """
        input = input.unsqueeze(1)
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        tmp_output = output[-1]
        tmp_output = self.linear(tmp_output)
        return self.softmax(tmp_output), hidden, cell

    def init_hidden(self):
        """初始化 hidden 和 cell 为零"""
        hidden = torch.zeros(self.n_layers, 1, self.hidden_size)
        cell = torch.zeros(self.n_layers, 1, self.hidden_size)
        return hidden, cell


class My_GRU(nn.Module):
    """GRU 模型

    GRU(input_size, hidden_size) → Linear → LogSoftmax
    LSTM 的简化版，只有 hidden state，参数更少、训练更快。
    """

    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=False)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """前向传播

        Args:
            input:  [seq_len, input_size]
            hidden: [n_layers, 1, hidden_size]

        Returns:
            output: [1, output_size]
            hidden: [n_layers, 1, hidden_size]
        """
        input = input.unsqueeze(1)
        output, hidden = self.gru(input, hidden)
        tmp_output = output[-1]
        tmp_output = self.linear(tmp_output)
        return self.softmax(tmp_output), hidden

    def init_hidden(self):
        """初始化隐藏状态为零"""
        return torch.zeros(self.n_layers, 1, self.hidden_size)
