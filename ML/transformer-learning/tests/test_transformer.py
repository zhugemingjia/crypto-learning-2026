"""基础形状与数值检查，使用标准库 unittest，无需额外依赖。"""

import copy
import unittest

import torch

from src import (
    Decoder,
    DecoderLayer,
    Embedding,
    Encoder,
    EncoderLayer,
    FeedForward,
    Generator,
    MultiHeadAttention,
    PositionalEncoding,
    make_model,
)


class TestTransformerComponents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.batch = 2
        self.seq = 4
        self.d_model = 512
        self.vocab = 1000

    def test_embedding_shape(self):
        x = torch.randint(0, self.vocab, (self.batch, self.seq))
        out = Embedding(self.vocab, self.d_model)(x)
        self.assertEqual(out.shape, (self.batch, self.seq, self.d_model))

    def test_positional_encoding_keeps_sequence_shape(self):
        x = torch.randn(self.batch, self.seq, self.d_model)
        out = PositionalEncoding(self.d_model)(x)
        self.assertEqual(out.shape, x.shape)

    def test_multihead_attention_shape(self):
        x = torch.randn(self.batch, self.seq, self.d_model)
        mha = MultiHeadAttention(self.d_model, h=8)
        out = mha(x, x, x)
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(mha.attn.shape, (self.batch, 8, self.seq, self.seq))

    def test_encoder_shape(self):
        attention = MultiHeadAttention(self.d_model, 8)
        ff = FeedForward(self.d_model, 2048)
        encoder = Encoder(EncoderLayer(self.d_model, attention, ff), 2)
        x = torch.randn(self.batch, self.seq, self.d_model)
        out = encoder(x, torch.zeros(self.batch, self.seq, self.seq))
        self.assertEqual(out.shape, x.shape)

    def test_decoder_shape(self):
        attention = MultiHeadAttention(self.d_model, 8)
        ff = FeedForward(self.d_model, 2048)
        layer = DecoderLayer(
            self.d_model,
            copy.deepcopy(attention),
            copy.deepcopy(attention),
            ff,
        )
        decoder = Decoder(layer, 2)
        x = torch.randn(self.batch, self.seq, self.d_model)
        memory = torch.randn(self.batch, self.seq, self.d_model)
        out = decoder(
            x,
            memory,
            torch.zeros(self.batch, self.seq, self.seq),
            torch.zeros(self.batch, self.seq, self.seq),
        )
        self.assertEqual(out.shape, x.shape)

    def test_generator_output_shape(self):
        generator = Generator(self.d_model, self.vocab)
        x = torch.randn(self.batch, self.seq, self.d_model)
        out = generator(x)
        self.assertEqual(out.shape, (self.batch, self.seq, self.vocab))
        self.assertTrue(torch.allclose(torch.exp(out).sum(-1), torch.tensor(1.0), atol=1e-6))

    def test_full_model_forward(self):
        model = make_model()
        src = torch.randint(0, self.vocab, (self.batch, self.seq))
        tgt = torch.randint(0, self.vocab, (self.batch, self.seq))
        src_mask = torch.zeros(self.batch, self.seq, self.seq)
        tgt_mask = torch.zeros(self.batch, self.seq, self.seq)
        out = model(src, tgt, src_mask, tgt_mask)
        self.assertEqual(out.shape, (self.batch, self.seq, self.vocab))


if __name__ == "__main__":
    unittest.main()
