import math
import unittest
import random
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from embeddings_manager.embeddings_manager import EmbeddingsManager
from multi_head_attention import MultiHeadAttention


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        vocab_size: int,
        dropout_p: float,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.embeddings_manager = embeddings_manager
        self.dropout = nn.Dropout(p=0.1)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(hidden_dim, ff_dim, num_heads, dropout_p)
                for _ in range(num_layers)
            ]
        )

        # Projecting the hidden dimension into vocabulary size,
        # so that we can use softmax and find specific word probability.
        self.output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Note: a linear layer multiplies the input with a transpose of the weight matrix, so no need to do that here.
        # Tying weights refers to the practice of sharing the same weight matrix
        # between the embedding layer and the output (softmax) layer.
        # This means that the embedding layer and the output layer use
        # the same set of weights to transform the input and produce the output probabilities.
        # tie_output_to_embedding: Optional[bool] = True,
        # if tie_output_to_embedding:
        #     self.output_layer.weight = nn.Parameter(self.embed.weight)

    def _reset_parameters(self):
        """Perform xavier weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
        self,
        batch_io_parser_output: list[list[dict]],
        task_type: str,
        encoder_hidden_states: torch.Tensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Performs one decoder forward pass given encoder hidden states, the decoder input tokens and attention masks.
        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        V = vocabulary size

        :param task_type: Type of task. ie: func_to_nl_translation.
        :param batch_io_parser_output: batch of io_parser_output. Shape (N, T)
        :param encoder_hidden_states: The encoder's final (contextualized) token embeddings. Shape: (N, S, E)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :param future_mask: An attention mask to ignore future-tokens in the target input. Shape (T, T)
        :return: Unnormalized logits over the vocabulary for every token in the batch. Shape (N, T, V)
        """
        # (batch_size, sequence_length, hidden_dim)
        x = self.embeddings_manager.get_batch_combined_embeddings(batch_io_parser_output, task_type) * math.sqrt(
            self.hidden_dim)  # (N, T, E)
        x = self.dropout(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_hidden_states, src_padding_mask, future_mask)

        # (batch_size, sequence_length, vocab_size)
        logits = self.output_layer(x)

        # Not using softmax need to figure out why,
        # Training result degrade in quality
        # softmax_output = torch.softmax(logits, dim=1)

        return logits

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()


class DecoderBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()

        self.cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.dropout3 = nn.Dropout(p=dropout_p)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        encoder_hidden_states: torch.FloatTensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Performs one decoder *block* forward pass given final encoder hidden states, the previous block's output, and
        attention masks.

        N = batch size
        S = source sequence length
        T = target sequence length
        E = embedding dimensionality
        V = vocabulary size

        :param x: Previous decoder block's output. Shape: (N, T, E)
        :param encoder_hidden_states: The encoder's final (contextualized) token embeddings. Shape: (N, S, E)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :param future_mask: An attention mask to ignore future-tokens in the target input. Shape (T, T)
        :return: Updated, contextualized token embeddings. Shape (N, T, E)
        """

        # Self attention (with future masking during training)
        output = self.dropout1(self.self_mha.forward(x, future_mask=future_mask))
        x = self.layer_norm1(x + output)

        # Cross or encoder-decoder attention
        output = self.dropout2(
            self.cross_mha.forward(
                x,
                encoder_hidden_states=encoder_hidden_states,
                src_padding_mask=src_padding_mask,
            )
        )
        x = self.layer_norm2(x + output)

        # Feed forward layers
        output = self.dropout3(self.feed_forward(x))
        x = self.layer_norm3(x + output)
        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()


class TestTransformerDecoder(unittest.TestCase):
    def test_one_layer_transformer_decoder_inference(self):
        """
        Test two forward passes, simulating two inference steps using teacher forcing
        """
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        with torch.no_grad():
            batch_size = 2
            src_seq_len = 10
            hidden_dim = 768
            vocab_size = 2000
            num_layers = 1
            num_heads = 8
            max_decoding_length = 1 # Not used anymore
            task_type = TaskTypes.NL_TO_NL_TRANSLATION.value

            # Prepare fake encoder hidden states and padding masks
            encoder_output = torch.randn((batch_size, src_seq_len, hidden_dim))
            src_padding_mask = torch.BoolTensor(
                [[1, 1, 1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
            )

            # Initialize the decoder, perform xavier init and set to evaluation mode
            decoder = TransformerDecoder(
                embeddings_manager=EmbeddingsManager(
                    batch_size=batch_size,
                    n_heads=num_heads,
                    max_sequence_length=max_decoding_length,
                    with_mask=False,
                ),
                hidden_dim=hidden_dim,
                ff_dim=2048,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_p=0.1,
                vocab_size=vocab_size,
            )
            decoder._reset_parameters()
            decoder.eval()

            # Iteratively feed the decoder input back in using teacher forcing
            sentences = [
                "Hello my name is Joris and I was born with the name Joris.",
                "A shorter sequence in the batch",
            ]
            decoder_input = BatchBuilder.get_batch_io_parser_output(sentences, True, 1)
            future_mask = None
            for i in range(3):
                index = i + 1
                decoder_output = decoder.forward(
                    decoder_input,
                    task_type,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)
                print(f"Predicted token index: {predicted_tokens}")

                # Teacher forcing
                decoder_input = BatchBuilder.get_batch_io_parser_output(sentences, True, index + 1)
                future_mask = BatchBuilder.construct_future_mask(index + 1)

                self.assertEqual(decoder_output.shape, (batch_size, index, vocab_size))
                # softmax entropy should not be 0
                self.assertEqual(torch.any(decoder_output == 1), False)
                """
                With only one decoder layer the predicted tokens will always be the input token ids. This happens
                only when the final linear transformation is tied to the (transpose of) the embedding matrix.
                This is because the input embedding is barely transformed due to residual connections. This results in
                the highest dot product between its final "contextualized" embedding and the original embedding vector
                in the pre-softmax weight matrix (i.e. embedding matrix) - because they are still very similar.
                This can be avoided by 1) scaling up the memory states - probably because this adds sufficient random
                noise through cross-attention to the contextualised embedding to divergence from the input embedding.
                2) increasing the number of layers - again adding more and more "noise" or 3) removing the last
                residual connection after the feed forward layers. In practice, however, this is not an issue. Training
                will take care of it.
                """


if __name__ == "__main__":
    unittest.main()
