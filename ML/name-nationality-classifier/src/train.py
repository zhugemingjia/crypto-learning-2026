"""
模型训练模块

- train_rnn():  训练 RNN 模型
- train_lstm(): 训练 LSTM 模型
- train_gru():  训练 GRU 模型
- demo_show_rnn_lstm_gru(): 三模型对比训练 + 可视化
"""

import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from .dataset import read_data, NameClassDataset, n_letters, category_num
from .model import My_RNN, My_LSTM, My_GRU

# 解决绘图中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ============ 超参数 ============

LR = 1e-3
EPOCHS = 3
INPUT_SIZE = n_letters       # 56（字符集大小）
HIDDEN_SIZE = 128
OUTPUT_SIZE = category_num   # 18（国家数）
N_LAYERS = 1


def _train_model(model, criterion, optimizer, name_class_dataset, epochs, model_name, save_prefix):
    """通用训练循环

    Args:
        model:             nn.Module 实例
        criterion:         损失函数
        optimizer:         优化器
        name_class_dataset: 数据集
        epochs:            训练轮数
        model_name:        模型名称（'RNN' / 'LSTM' / 'GRU'）
        save_prefix:       保存路径前缀

    Returns:
        total_loss_list: 每 100 步平均损失列表
        total_time:      总耗时（秒）
        total_acc_list:  每 100 步平均准确率列表
    """
    start_time = time.time()
    total_iter_num = 0
    total_loss = 0.0
    total_loss_list = []
    total_acc_num = 0
    total_acc_list = []

    os.makedirs('./model', exist_ok=True)

    for epoch in range(epochs):
        print(f"\n开始第 {epoch + 1} / {epochs} 轮训练...")
        train_dataloader = DataLoader(name_class_dataset, batch_size=1, shuffle=True)

        for i, (x, y) in enumerate(tqdm(train_dataloader)):
            # ---- 前向传播 ----
            if model_name == 'LSTM':
                h, c = model.init_hidden()
                output, hidden, cell = model(x[0], h, c)
            else:
                output, hidden = model(x[0], model.init_hidden())

            loss = criterion(output, y)

            # ---- 反向传播 ----
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # ---- 统计 ----
            total_iter_num += 1
            total_loss += loss.item()
            pred_tag = torch.argmax(output).item()
            total_acc_num += (1 if pred_tag == y.item() else 0)

            if total_iter_num % 100 == 0:
                total_loss_list.append(total_loss / total_iter_num)
                total_acc_list.append(total_acc_num / total_iter_num)

            if total_iter_num % 2000 == 0:
                avg_loss = total_loss / total_iter_num
                elapsed = int(time.time() - start_time)
                avg_acc = total_acc_num / total_iter_num
                print(f"轮次: {epoch + 1}, 样本数: {total_iter_num}, "
                      f"平均损失: {avg_loss:.4f}, 耗时: {elapsed}s, 准确率: {avg_acc:.4f}")

        # 每轮保存一次
        torch.save(model.state_dict(), f'{save_prefix}{epoch + 1}.bin')

    total_time = int(time.time() - start_time)
    print(f"\n{model_name} 训练完成，总耗时: {total_time}s, 总样本数: {total_iter_num}")
    return total_loss_list, total_time, total_acc_list


def train_rnn(data_path='./data/name_classfication.txt'):
    """训练 RNN 模型"""
    my_list_x, my_list_y = read_data(data_path)
    dataset = NameClassDataset(my_list_x, my_list_y)

    model = My_RNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    return _train_model(model, criterion, optimizer, dataset, EPOCHS,
                        'RNN', './model/my_rnn_name_prj')


def train_lstm(data_path='./data/name_classfication.txt'):
    """训练 LSTM 模型"""
    my_list_x, my_list_y = read_data(data_path)
    dataset = NameClassDataset(my_list_x, my_list_y)

    model = My_LSTM(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    return _train_model(model, criterion, optimizer, dataset, EPOCHS,
                        'LSTM', './model/my_lstm_name_prj')


def train_gru(data_path='./data/name_classfication.txt'):
    """训练 GRU 模型"""
    my_list_x, my_list_y = read_data(data_path)
    dataset = NameClassDataset(my_list_x, my_list_y)

    model = My_GRU(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    return _train_model(model, criterion, optimizer, dataset, EPOCHS,
                        'GRU', './model/my_gru_name_prj')


def demo_show_rnn_lstm_gru(data_path='./data/name_classfication.txt'):
    """三模型对比训练 + 可视化

    依次训练 RNN / LSTM / GRU，然后绘制三张对比图：
    1. 损失曲线
    2. 训练耗时柱状图
    3. 准确率曲线
    """
    print("=" * 60)
    print("开始三模型对比训练：RNN → LSTM → GRU")
    print("=" * 60)

    total_loss_list_rnn, total_time_rnn, total_acc_list_rnn = train_rnn(data_path)
    total_loss_list_lstm, total_time_lstm, total_acc_list_lstm = train_lstm(data_path)
    total_loss_list_gru, total_time_gru, total_acc_list_gru = train_gru(data_path)

    os.makedirs('./img', exist_ok=True)

    # ---- 图 1: 损失对比 ----
    plt.figure(0, figsize=(10, 5))
    plt.plot(total_loss_list_rnn, label='RNN', color='red')
    plt.plot(total_loss_list_lstm, label='LSTM', color='green')
    plt.plot(total_loss_list_gru, label='GRU', color='blue')
    plt.title('模型损失对比曲线')
    plt.xlabel('训练步数（每100步）')
    plt.ylabel('平均损失值')
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.legend(loc='upper right')
    plt.savefig('./img/RNN_LSTM_GRU_loss.png')
    plt.show()

    # ---- 图 2: 耗时对比 ----
    plt.figure(1, figsize=(10, 5))
    x_data = ['RNN', 'LSTM', 'GRU']
    y_data = [total_time_rnn, total_time_lstm, total_time_gru]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data, color=['red', 'green', 'blue'])
    plt.title('模型耗时对比柱状图')
    plt.ylabel('耗时（秒）')
    for i, v in enumerate(y_data):
        plt.text(i, v + 1, str(v), ha='center')
    plt.savefig('./img/RNN_LSTM_GRU_time.png')
    plt.show()

    # ---- 图 3: 准确率对比 ----
    plt.figure(2, figsize=(10, 5))
    plt.plot(total_acc_list_rnn, label='RNN', color='red')
    plt.plot(total_acc_list_lstm, label='LSTM', color='green')
    plt.plot(total_acc_list_gru, label='GRU', color='blue')
    plt.title('模型准确率对比曲线')
    plt.xlabel('训练步数（每100步）')
    plt.ylabel('平均准确率')
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.legend(loc='lower right')
    plt.savefig('./img/RNN_LSTM_GRU_acc.png')
    plt.show()

    print("\n三模型对比训练完成，图表已保存到 ./img/")
