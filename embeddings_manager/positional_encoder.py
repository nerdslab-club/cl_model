import math
import unittest

import pytorch_lightning as pl
import torch


class SinusoidEncoder(pl.LightningModule):
    def __init__(self, embedding_dim, max_len=5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_len = max_len

    def positional_encoding(self, x):
        position = torch.arange(
            0, self.max_len, dtype=torch.float, device=x.device
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float, device=x.device)
            * (-math.log(10000.0) / self.embedding_dim)
        )
        print(div_term.shape)
        pos_enc = torch.zeros(self.max_len, self.embedding_dim, device=x.device)
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        return pos_enc

    def forward(self, x):
        pos_enc = self.positional_encoding(x)
        x = x + pos_enc[: x.size(1)]
        return x


class TestSinusoidEncoding(unittest.TestCase):
    def test_create_embedding(self):
        batch = 1
        embedding_dim = 8
        seq_len = 3
        x = torch.zeros(batch, seq_len, embedding_dim)
        encoding = SinusoidEncoder(embedding_dim).forward(x)
        expected = torch.Tensor(
            [
                [
                    [
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                    ],
                    [
                        8.4147e-01,
                        5.4030e-01,
                        9.9833e-02,
                        9.9500e-01,
                        9.9998e-03,
                        9.9995e-01,
                        1.0000e-03,
                        1.0000e00,
                    ],
                    [
                        9.0930e-01,
                        -4.1615e-01,
                        1.9867e-01,
                        9.8007e-01,
                        1.9999e-02,
                        9.9980e-01,
                        2.0000e-03,
                        1.0000e00,
                    ],
                ]
            ]
        )
        torch.testing.assert_close(encoding, expected, rtol=10e-5, atol=10e-5)

    def test_create_embedding_multi_batch(self):
        batch = 2
        embedding_dim = 8
        seq_len = 3
        x = torch.zeros(batch, seq_len, embedding_dim)
        encoding = SinusoidEncoder(embedding_dim).forward(x)
        expected = torch.Tensor(
            [
                [
                    [
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                    ],
                    [
                        8.4147e-01,
                        5.4030e-01,
                        9.9833e-02,
                        9.9500e-01,
                        9.9998e-03,
                        9.9995e-01,
                        1.0000e-03,
                        1.0000e00,
                    ],
                    [
                        9.0930e-01,
                        -4.1615e-01,
                        1.9867e-01,
                        9.8007e-01,
                        1.9999e-02,
                        9.9980e-01,
                        2.0000e-03,
                        1.0000e00,
                    ],
                ],
                [
                    [
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                        0.0000e00,
                        1.0000e00,
                    ],
                    [
                        8.4147e-01,
                        5.4030e-01,
                        9.9833e-02,
                        9.9500e-01,
                        9.9998e-03,
                        9.9995e-01,
                        1.0000e-03,
                        1.0000e00,
                    ],
                    [
                        9.0930e-01,
                        -4.1615e-01,
                        1.9867e-01,
                        9.8007e-01,
                        1.9999e-02,
                        9.9980e-01,
                        2.0000e-03,
                        1.0000e00,
                    ],
                ],
            ]
        )
        torch.testing.assert_close(encoding, expected, rtol=10e-5, atol=10e-5)


if __name__ == "__main__":
    pl.seed_everything(42)
    unittest.main()
