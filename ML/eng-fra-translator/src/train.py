"""
训练模块：单批次训练 + 完整多轮训练循环
"""

import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from .dataset import SOS_token, EOS_token, MAX_LENGTH, device
from .model import EncoderRNN, AttnDecoderRNN


# ============================================================
# 超参数
# ============================================================
my_lr = 5e-4
epochs = 20
teacher_forcing_ratio = 0.3
print_interval_num = 1000
plot_interval_num = 100


# ============================================================
# 1. 单批次训练（一个样本的 编码→解码→反向传播）
# ============================================================
def train_iters(x, y,
                my_encoder_rnn: EncoderRNN,
                my_attn_decoder_rnn: AttnDecoderRNN,
                myadam_encode: optim.Adam,
                myadam_decode: optim.Adam,
                my_crossentropy_loss: nn.NLLLoss):
    """
    完成一个样本的编码 → 解码 → 反向传播 → 参数更新

    :param x: 输入序列（英语）[batch_size=1, seq_len]
    :param y: 目标序列（法语）[batch_size=1, seq_len]
    :return: 平均损失值 (loss / y_len)
    """
    # ---- 编码阶段 ----
    encoder_hidden = my_encoder_rnn.init_hidden()
    encoder_output, encoder_hidden = my_encoder_rnn(x, encoder_hidden)

    # ---- 解码准备 ----
    # 构建固定长度编码器输出张量 [MAX_LENGTH, hidden_size]
    encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
    for idx in range(x.shape[1]):
        encoder_output_c[idx] = encoder_output[0, idx]

    decoder_hidden = encoder_hidden                    # [1, 1, hidden_size]
    input_y = torch.tensor([[SOS_token]], device=device)  # [1, 1]

    # ---- 逐时间步解码 ----
    my_loss = 0.0
    y_len = y.shape[1]
    use_teacher_forcing = random.random() < teacher_forcing_ratio

    if use_teacher_forcing:
        # Teacher Forcing: 用真实标签作为下一步输入
        for i in range(y_len):
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(
                input_y, decoder_hidden, encoder_output_c
            )
            target_y = y[0][i].view(1)
            my_loss += my_crossentropy_loss(output_y, target_y)
            input_y = y[0][i].view(1, -1)  # 下一步输入 = 真实标签
    else:
        # 无 Teacher Forcing: 用上一步预测结果作为下一步输入
        for i in range(y_len):
            output_y, decoder_hidden, attn_weights = my_attn_decoder_rnn(
                input_y, decoder_hidden, encoder_output_c
            )
            target_y = y[0][i].view(1)
            my_loss += my_crossentropy_loss(output_y, target_y)

            topv, topi = output_y.topk(1)
            if topi.squeeze().item() == EOS_token:
                break
            input_y = topi.detach()  # 下一步输入 = 预测结果

    # ---- 反向传播 & 参数更新 ----
    myadam_encode.zero_grad()
    myadam_decode.zero_grad()
    my_loss.backward()
    myadam_encode.step()
    myadam_decode.step()

    return my_loss.item() / y_len


# ============================================================
# 2. 完整训练循环
# ============================================================
def train_seq2seq(my_dataloader, english_word_n, french_word_n,
                  model_save_dir='./model', img_save_dir='./img'):
    """
    多轮、多批次训练 Seq2Seq 模型（带 Attention）

    :param my_dataloader:   DataLoader 实例
    :param english_word_n:  英语词汇表大小
    :param french_word_n:   法语词汇表大小
    :param model_save_dir:  模型保存目录
    :param img_save_dir:    图片保存目录
    :return: plot_loss_list（绘图用损失列表）
    """
    # ---- 模型初始化 ----
    my_encoder_rnn = EncoderRNN(english_word_n, 256).to(device)
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256,
                                         dropout_p=0.1, max_length=10).to(device)

    # ---- 优化器 ----
    myadam_encode = optim.Adam(my_encoder_rnn.parameters(), lr=my_lr)
    myadam_decode = optim.Adam(my_attn_decoder_rnn.parameters(), lr=my_lr)

    # ---- 损失函数 ----
    my_crossentropy_loss = nn.NLLLoss()

    # ---- 训练 ----
    plot_loss_list = []

    for epoch_idx in range(1, epochs + 1):
        print_loss_total = 0.0
        plot_loss_total = 0.0
        start_time = time.time()

        for item, (x, y) in enumerate(tqdm(my_dataloader, desc=f"Epoch {epoch_idx}"), start=1):
            # 单批次训练
            my_loss = train_iters(x, y,
                                  my_encoder_rnn, my_attn_decoder_rnn,
                                  myadam_encode, myadam_decode,
                                  my_crossentropy_loss)

            print_loss_total += my_loss
            plot_loss_total += my_loss

            # 打印训练日志
            if item % print_interval_num == 0:
                print_loss_avg = print_loss_total / print_interval_num
                print_loss_total = 0.0
                print(f"轮次: {epoch_idx}, 平均损失: {print_loss_avg:.4f}, "
                      f"耗时: {time.time() - start_time:.4f} s")

            # 记录损失用于绘图
            if item % plot_interval_num == 0:
                plot_loss_avg = plot_loss_total / plot_interval_num
                plot_loss_list.append(plot_loss_avg)
                plot_loss_total = 0.0

        # ---- 每轮结束：保存模型 ----
        torch.save(my_encoder_rnn.state_dict(),
                   f'{model_save_dir}/my_encoder_rnn_{epoch_idx}.pth')
        torch.save(my_attn_decoder_rnn.state_dict(),
                   f'{model_save_dir}/my_attn_decoder_rnn_{epoch_idx}.pth')
        print(f'模型已保存至 {model_save_dir}/my_encoder_rnn_{epoch_idx}.pth')
        print(f'--- 第 {epoch_idx} 轮训练完成 ---')

    # ---- 绘制损失曲线 ----
    plt.figure()
    plt.plot(plot_loss_list)
    plt.savefig(f"{img_save_dir}/seq2seq_loss.png")
    print(f"损失曲线已保存至 {img_save_dir}/seq2seq_loss.png")
    plt.show()

    return plot_loss_list
