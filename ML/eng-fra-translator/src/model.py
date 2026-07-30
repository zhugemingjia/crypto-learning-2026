"""
模型定义：基于 GRU 的编码器 / 解码器（无 Attention + 带 Attention）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import MAX_LENGTH, device


# ============================================================
# 1. 编码器（基于 GRU）
# ============================================================
class EncoderRNN(nn.Module):
    """
    Seq2Seq 编码器：英文句子 → 隐藏状态序列

    结构: Embedding → GRU
    """

    def __init__(self, input_size: int, hidden_size: int):
        """
        :param input_size:  词汇表大小（英语单词数）
        :param hidden_size: 隐藏层维度
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 词嵌入层: [batch, seq_len] → [batch, seq_len, hidden_size]
        self.embedding = nn.Embedding(input_size, hidden_size)

        # GRU 层: batch_first=True 表示输入形状为 [batch, seq_len, features]
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

    def forward(self, input, hidden):
        """
        :param input:  [batch_size, seq_len] 单词索引序列
        :param hidden: [num_layers, batch_size, hidden_size] 初始隐藏状态
        :return: (output, hidden)
        """
        output = self.embedding(input)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self):
        """返回全零初始隐藏状态"""
        return torch.zeros(1, 1, self.hidden_size, device=device)


# ============================================================
# 2. 解码器（无 Attention）— 基础版
# ============================================================
class DecoderRNN(nn.Module):
    """
    Seq2Seq 解码器（无注意力机制）

    结构: Embedding → ReLU → GRU → Linear → LogSoftmax
    """

    def __init__(self, output_size: int, hidden_size: int):
        """
        :param output_size: 目标语言词汇表大小（法语单词数）
        :param hidden_size: 隐藏层维度
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):
        """
        :param input:  [batch_size, seq_len] 当前时间步输入
        :param hidden: [num_layers, batch_size, hidden_size]
        :return: (output, hidden)
        """
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden


# ============================================================
# 3. 解码器（带 Attention）— 核心版
# ============================================================
class AttnDecoderRNN(nn.Module):
    """
    带注意力机制的 GRU 解码器

    结构:
        Embedding → Dropout
        → Attention(concat(embedded, hidden) → Linear → softmax)
        → bmm(attn_weights, encoder_outputs)
        → concat(embedded, attn_applied) → Linear → ReLU
        → GRU → Linear → LogSoftmax
    """

    def __init__(self, output_size: int, hidden_size: int,
                 dropout_p: float = 0.1, max_length: int = MAX_LENGTH):
        """
        :param output_size: 目标语言词汇表大小
        :param hidden_size: 隐藏层维度
        :param dropout_p:   Dropout 概率
        :param max_length:  句子最大长度（用于注意力权重维度）
        """
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        # 词嵌入
        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # 注意力权重计算: concat(embedded, hidden) → attn_weights
        self.attn = nn.Linear(self.hidden_size + self.hidden_size, self.max_length)

        # 注意力融合: concat(embedded, attn_applied) → hidden_size
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, encoder_outputs):
        """
        :param input:            [batch_size, 1] 当前时间步输入词索引
        :param hidden:           [1, batch_size, hidden_size] 上一时间步隐藏状态
        :param encoder_outputs:  [max_length, hidden_size] 编码器所有时间步输出
        :return: (output, hidden, attn_weights)
        """
        # 1. 词嵌入 + Dropout
        embedded = self.embedding(input)          # [1, 1, hidden_size]
        embedded = self.dropout(embedded)

        # 2. 注意力权重
        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), dim=1)), dim=1
        )

        # 3. 注意力上下文向量
        attn_applied = torch.bmm(
            attn_weights.unsqueeze(0),
            encoder_outputs.unsqueeze(0)
        )

        # 4. 融合嵌入向量与注意力上下文
        output = torch.cat((embedded[0], attn_applied[0]), dim=1)
        output = self.attn_combine(output).unsqueeze(0)

        # 5. 激活 + GRU
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        # 6. 输出层
        output = self.softmax(self.out(output[0]))  # [1, output_size]

        return output, hidden, attn_weights
