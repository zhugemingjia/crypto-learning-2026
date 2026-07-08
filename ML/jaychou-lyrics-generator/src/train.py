"""
训练函数
"""

import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .dataset import build_vocab, LyricsDataset
from .model import TextGenerator


def train(
    lyrics_path: str = "./data/jaychou_lyrics.txt",
    seq_len: int = 32,
    batch_size: int = 5,
    epochs: int = 10,
    lr: float = 1e-3,
    grad_clip: float = 5.0,
    save_dir: str = "./data",
    device: str = None,
    verbose: bool = True,
):
    """
    训练主循环

    关键技巧:
        - detach() 隐藏状态（避免跨 batch 反传）
        - 梯度裁剪（防止 RNN 梯度爆炸）
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    os.makedirs(save_dir, exist_ok=True)

    # 1. 构建词表（同时保存）
    vocab_path = os.path.join(save_dir, "vocab.json")
    word_to_index, index_to_word, vocab_size, corpus_idx = build_vocab(lyrics_path, save_path=vocab_path)
    if verbose:
        print(f"词表大小: {vocab_size}  |  语料 token 总数: {len(corpus_idx)}")
        print(f"✅ 词表已保存到: {vocab_path}")

    # 2. Dataset + DataLoader
    dataset = LyricsDataset(corpus_idx, num_chars=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. 模型 + 损失 + 优化器
    model = TextGenerator(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    for epoch in range(epochs):
        start = time.time()
        iter_num = 0
        total_loss = 0.0
        hidden = model.init_hidden(batch_size, device=device)

        for x, y in loader:
            # 最后一个 batch 可能不足 batch_size
            bs = x.size(0)
            if bs != batch_size:
                hidden = model.init_hidden(bs, device=device)
            else:
                hidden = hidden.detach()

            x, y = x.to(device), y.to(device)

            # 前向
            output, hidden = model(x, hidden)
            # output: (bs*seq, vocab)  y: (bs, seq) -> (bs*seq)
            y = y.transpose(0, 1).reshape(-1).to(device)

            loss = criterion(output, y)

            # 反向
            optimizer.zero_grad()
            loss.backward()
            # 梯度裁剪（防止 RNN 爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()
            iter_num += 1

        avg_loss = total_loss / iter_num
        loss_history.append(avg_loss)
        if verbose:
            print(f"epoch {epoch + 1:2d}/{epochs}  loss={avg_loss:.4f}  time={time.time() - start:.2f}s")

    # 4. 保存模型
    model_path = os.path.join(save_dir, "lyric_model.pth")
    torch.save(model.state_dict(), model_path)
    if verbose:
        print(f"\n✅ 模型已保存到: {model_path}")

    return model, loss_history
