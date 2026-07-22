"""
模型预测模块

- demo_rnn_predict():  RNN 单样本预测（Top-3）
- demo_lstm_predict(): LSTM 单样本预测（Top-3）
- demo_gru_predict():  GRU 单样本预测（Top-3）
- predict_all_models(): 三模型同时预测对比
"""

import os
import torch

from .dataset import all_letters, n_letters, categorys, category_num, lineToTensor
from .model import My_RNN, My_LSTM, My_GRU

HIDDEN_SIZE = 128
OUTPUT_SIZE = category_num  # 18
N_LAYERS = 1


def _print_top3(output, model_label, input_name):
    """打印 Top-3 预测结果"""
    topv, topi = output.topk(3, 1, True)
    print(f"\n[{model_label} 预测] 待预测人名: {input_name}")
    print("-" * 40)
    for i in range(3):
        value = topv[0][i].item()
        category_index = topi[0][i].item()
        category = categorys[category_index]
        prob = torch.exp(torch.tensor(value)).item()
        print(f"  Top {i+1}: {category:<12} 概率: {prob:.4f}  (log_prob: {value:.4f})")


def demo_rnn_predict(x, model_path='./model/my_rnn_name_prj1.bin'):
    """使用 RNN 模型预测人名的国家

    Args:
        x:          人名字符串
        model_path: 模型权重路径
    """
    x_tensor = lineToTensor(x)
    my_rnn = My_RNN(n_letters, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS)

    if os.path.exists(model_path):
        my_rnn.load_state_dict(torch.load(model_path, map_location='cpu'))
        my_rnn.eval()
    else:
        print(f"警告: 模型文件 {model_path} 不存在，使用随机初始化的模型")
        my_rnn.eval()

    with torch.no_grad():
        output, hidden = my_rnn(x_tensor, my_rnn.init_hidden())
        _print_top3(output, 'RNN', x)


def demo_lstm_predict(x, model_path='./model/my_lstm_name_prj1.bin'):
    """使用 LSTM 模型预测人名的国家"""
    x_tensor = lineToTensor(x)
    my_lstm = My_LSTM(n_letters, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS)

    if os.path.exists(model_path):
        my_lstm.load_state_dict(torch.load(model_path, map_location='cpu'))
        my_lstm.eval()
    else:
        print(f"警告: 模型文件 {model_path} 不存在，使用随机初始化的模型")
        my_lstm.eval()

    with torch.no_grad():
        h, c = my_lstm.init_hidden()
        output, hidden, cell = my_lstm(x_tensor, h, c)
        _print_top3(output, 'LSTM', x)


def demo_gru_predict(x, model_path='./model/my_gru_name_prj1.bin'):
    """使用 GRU 模型预测人名的国家"""
    x_tensor = lineToTensor(x)
    my_gru = My_GRU(n_letters, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYERS)

    if os.path.exists(model_path):
        my_gru.load_state_dict(torch.load(model_path, map_location='cpu'))
        my_gru.eval()
    else:
        print(f"警告: 模型文件 {model_path} 不存在，使用随机初始化的模型")
        my_gru.eval()

    with torch.no_grad():
        output, hidden = my_gru(x_tensor, my_gru.init_hidden())
        _print_top3(output, 'GRU', x)


def predict_all_models(x,
                       rnn_path='./model/my_rnn_name_prj1.bin',
                       lstm_path='./model/my_lstm_name_prj1.bin',
                       gru_path='./model/my_gru_name_prj1.bin'):
    """使用 RNN / LSTM / GRU 三个模型同时预测并对比

    Args:
        x:         人名字符串
        rnn_path:  RNN 权重路径
        lstm_path: LSTM 权重路径
        gru_path:  GRU 权重路径
    """
    demo_rnn_predict(x, rnn_path)
    demo_lstm_predict(x, lstm_path)
    demo_gru_predict(x, gru_path)
