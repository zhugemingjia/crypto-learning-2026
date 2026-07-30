"""
评估 & 推理模块：加载训练好的模型，进行英 → 法翻译
"""

import torch

from .dataset import SOS_token, EOS_token, MAX_LENGTH, device, normalizeString
from .model import EncoderRNN, AttnDecoderRNN


# ============================================================
# 1. 单句翻译（核心推理函数）
# ============================================================
def evaluate_seq2seq(x,
                     my_encoder_rnn: EncoderRNN,
                     my_attn_decoder_rnn: AttnDecoderRNN,
                     french_index2word: dict):
    """
    使用训练好的 Seq2Seq 模型翻译一个英语句子

    :param x:                      输入张量 [1, seq_len]
    :param my_encoder_rnn:         编码器模型
    :param my_attn_decoder_rnn:    带注意力的解码器模型
    :param french_index2word:      法语索引 → 单词映射表
    :return: (decode_words, decoder_attentions)
    """
    with torch.no_grad():
        # ---- 编码 ----
        encoder_hidden = my_encoder_rnn.init_hidden()
        encode_output, encoder_hidden = my_encoder_rnn(x, encoder_hidden)

        # 构建固定长度编码器输出
        encoder_output_c = torch.zeros(MAX_LENGTH, my_encoder_rnn.hidden_size, device=device)
        for idx in range(x.shape[1]):
            encoder_output_c[idx] = encode_output[0, idx]

        decode_hidden = encoder_hidden
        input_y = torch.tensor([[SOS_token]], device=device)

        # ---- 自回归解码 ----
        decode_words = []
        decoder_attentions = torch.zeros(MAX_LENGTH, MAX_LENGTH)

        for idx in range(MAX_LENGTH):
            output_y, decode_hidden, attn_weights = my_attn_decoder_rnn(
                input_y, decode_hidden, encoder_output_c
            )

            decoder_attentions[idx] = attn_weights

            topv, topi = output_y.topk(1)
            if topi.squeeze().item() == EOS_token:
                break
            else:
                decode_words.append(french_index2word[topi.squeeze().item()])

            input_y = topi.detach()

        return decode_words, decoder_attentions[:idx + 1]


# ============================================================
# 2. 批量评估（加载模型 + 自定义样本翻译）
# ============================================================
def seq2seq_evaluate(english_word2index: dict,
                     french_index2word: dict,
                     english_word_n: int,
                     french_word_n: int,
                     encoder_path: str = './model/my_encoder_rnn_1.pth',
                     decoder_path: str = './model/my_attn_decoder_rnn_1.pth'):
    """
    加载预训练模型并对自定义样本进行翻译评估

    :param english_word2index: 英语词汇表映射
    :param french_index2word:  法语反向映射
    :param english_word_n:     英语词汇表大小
    :param french_word_n:      法语词汇表大小
    :param encoder_path:       编码器权重路径
    :param decoder_path:       解码器权重路径
    """
    # ---- 加载编码器 ----
    my_encoder_rnn = EncoderRNN(english_word_n, hidden_size=256).to(device)
    my_encoder_rnn.load_state_dict(
        torch.load(encoder_path, map_location=device, weights_only=True), strict=False
    )
    print(f"编码器模型架构:\n{my_encoder_rnn}")

    # ---- 加载解码器 ----
    my_attn_decoder_rnn = AttnDecoderRNN(french_word_n, 256).to(device)
    my_attn_decoder_rnn.load_state_dict(
        torch.load(decoder_path, map_location=device, weights_only=True), strict=False
    )
    print(f"解码器架构:\n{my_attn_decoder_rnn}")

    # ---- 自定义测试样本 ----
    my_sample_pairs = [
        ['i m ok .', 'je vais bien .'],
        ['i m sad .', 'je suis triste .'],
        ['i m busy .', 'je suis occupe .'],
        ['she is happy .', 'elle est heureuse .'],
        ['he is a teacher .', 'il est enseignant .'],
        ['i love you .', 'je t aime .'],
    ]
    print(f"自定义测试样本: {my_sample_pairs}")

    # ---- 逐样本翻译 ----
    for index, pair in enumerate(my_sample_pairs):
        x = pair[0]  # 英语
        y = pair[1]  # 法语（参考）

        # 文本数值化
        x_clean = normalizeString(x)
        tmpx = []
        for word in x_clean.split():
            if word in english_word2index:
                tmpx.append(english_word2index[word])
            else:
                print(f"警告：单词 '{word}' 不在英语词典中，已跳过")
        if len(tmpx) == 0:
            print("错误：句子中没有可用的已知单词，跳过该样本")
            continue
        tmpx.append(EOS_token)
        tensor_x = torch.tensor(tmpx, dtype=torch.long, device=device).view(1, -1)

        # 模型预测
        decode_words, attentions = evaluate_seq2seq(
            tensor_x, my_encoder_rnn, my_attn_decoder_rnn, french_index2word
        )

        # 输出结果
        output_sentence = ' '.join(decode_words)
        print(f"输入（英文句子）: {x}")
        print(f"参考（法语句子）: {y}")
        print(f"输出（预测法语）:   {output_sentence}")
        print("___===" * 10)
