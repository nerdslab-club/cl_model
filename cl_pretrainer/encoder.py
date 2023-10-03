import unittest
import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from embeddings_manager.embeddings_manager import EmbeddingsManager
from multi_head_attention import MultiHeadAttention


class TransformerEncoder(nn.Module):
    def __init__(
            self,
            embeddings_manager: EmbeddingsManager,
            hidden_dim: int,
            ff_dim: int,
            num_heads: int,
            num_layers: int,
            dropout_p: float,
    ):
        super().__init__()
        self.embeddings_manager = embeddings_manager
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(p=dropout_p)
        self.encoder_blocks = nn.ModuleList(
            [
                EncoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
            self,
            batch_io_parser_output: list[list[dict]],
            task_type: str,
            src_padding_mask: torch.BoolTensor = None,
    ):
        """
        Performs one encoder forward pass given input token ids and an optional attention mask.

        N = batch size
        S = source sequence length
        E = embedding dimensionality

        :param task_type: Type of task. ie: func_to_nl_translation.
        :param batch_io_parser_output: batch of io_parser_output
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :return: The encoder's final (contextualized) token embeddings. Shape: (N, S, E)
        """
        x = self.embeddings_manager.get_batch_combined_embeddings(batch_io_parser_output, task_type) * math.sqrt(
            self.hidden_dim)  # (N, S, E)

        x = self.dropout(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block.forward(x, src_padding_mask=src_padding_mask)
        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()


class EncoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        # Dropout is also known as regularization
        # Normalizing is adding
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.FloatTensor, src_padding_mask: torch.BoolTensor = None):
        """
        Performs one encoder *block* forward pass given the previous block's output and an optional attention mask.

        N = batch size
        S = source sequence length
        E = embedding dimensionality

        :param x: Tensor containing the output of the previous encoder block. Shape: (N, S, E)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :return: Updated intermediate encoder (contextualized) token embeddings. Shape: (N, S, E)
        """
        output = self.dropout1(
            self.self_mha.forward(x, src_padding_mask=src_padding_mask)
        )
        x = self.layer_norm1(x + output)

        output = self.dropout2(self.feed_forward(x))
        x = self.layer_norm2(x + output)
        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()


class TestTransformerEncoder(unittest.TestCase):
    def test_transformer_encoder_single_sequence_batch(self):
        sentences = ["Hello my name is Joris and I was born with the name Joris."]
        hidden_dim = 768  # !! IMPORTANT CHANGING THIS WON'T WORK, AS USING MODEL FOR TOKENIZATION
        batch_size = 1
        num_heads = 8
        max_encoding_length = 16
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value
        with torch.no_grad():
            # Initialize a transformer encoder (qkv_dim is automatically set to hidden_dim // num_heads)
            encoder = TransformerEncoder(
                embeddings_manager=EmbeddingsManager(
                    batch_size=batch_size,
                    n_heads=num_heads,
                    max_sequence_length=max_encoding_length,
                    with_mask=False,
                ),
                hidden_dim=hidden_dim,
                ff_dim=2048,
                num_heads=num_heads,
                num_layers=2,
                dropout_p=0.1,
            )
            encoder._reset_parameters()
            encoder.eval()
            batch_io_parser_outputs = BatchBuilder.get_batch_io_parser_output(sentences, True, max_encoding_length)
            output = encoder.forward(batch_io_parser_outputs, task_type)
            self.assertEqual(output.shape, (batch_size, max_encoding_length, hidden_dim))
            self.assertEqual(torch.any(torch.isnan(output)), False)

    def test_transformer_encoder_multi_sequence_batch(self):
        # Create vocabulary and special token indices given a dummy batch
        sentences = [
            "Hello my name is Joris and I was born with the name Joris.",
            "A shorter sequence in the batch",
        ]
        hidden_dim = 768  # !! IMPORTANT CHANGING THIS WON'T WORK, AS USING MODEL FOR TOKENIZATION
        batch_size = 2
        num_heads = 8
        max_encoding_length = 16
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value

        # Initialize a transformer encoder (qkv_dim is automatically set to hidden_dim // num_heads)
        with torch.no_grad():
            encoder = TransformerEncoder(
                embeddings_manager=EmbeddingsManager(
                    batch_size=batch_size,
                    n_heads=num_heads,
                    max_sequence_length=max_encoding_length,
                    with_mask=False,
                ),
                hidden_dim=hidden_dim,
                ff_dim=2048,
                num_heads=num_heads,
                num_layers=2,
                dropout_p=0.1,
            )
            encoder.eval()
            batch_io_parser_outputs = BatchBuilder.get_batch_io_parser_output(sentences, True, max_encoding_length)
            src_padding_mask = BatchBuilder.construct_padding_mask(batch_io_parser_outputs)
            output = encoder.forward(batch_io_parser_outputs, task_type, src_padding_mask=src_padding_mask)
            self.assertEqual(output.shape, (batch_size, max_encoding_length, hidden_dim))
            self.assertEqual(torch.any(torch.isnan(output)), False)


if __name__ == "__main__":
    unittest.main()
