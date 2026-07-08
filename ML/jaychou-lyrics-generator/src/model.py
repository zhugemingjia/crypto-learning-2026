"""
模型定义：词嵌入 + RNN + 输出层
"""

import torch
import torch.nn as nn


class TextGenerator(nn.Module):
    """
    周杰伦歌词生成器

    结构:
        nn.Embedding(vocab_size, 128)   # 词 -> 128 维向量
        nn.RNN(128, 256, num_layers=1)  # 单层 RNN
        nn.Linear(256, vocab_size)      # 映射回词表大小
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=False,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden):
        """
        x: (batch, seq_len) 词索引
        hidden: (num_layers, batch, hidden_dim)
        返回: (output, hidden)
            output: (batch * seq_len, vocab_size) 每个时间步每个词的预测 logits
        """
        emb = self.embedding(x)                              # (batch, seq_len, embed_dim)
        out, hidden = self.rnn(emb.transpose(0, 1), hidden)  # (seq_len, batch, hidden)
        out = self.output(out.reshape(-1, out.shape[-1]))   # (batch*seq_len, vocab_size)
        return out, hidden

    def init_hidden(self, batch_size: int, device=None):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim)
        if device is not None:
            h = h.to(device)
        return h
