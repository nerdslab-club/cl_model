import math
import unittest
from typing import Optional

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.category_map_decoder import CategoryMapDecoder
from cl_pretrainer.common_block import CommonBlock
from cl_pretrainer.output_token_block import OutputTokenBlock
from embeddings_manager.embeddings_manager import EmbeddingsManager


class OutputTokenDecoder(nn.Module):
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
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_param = dropout_p
        self.embeddings_manager = embeddings_manager

        self.output_token_decoder_dropout = nn.Dropout(p=0.1)
        self.output_token_decoder_blocks = nn.ModuleList(self._create_output_token_decoder_blocks())

    def _create_output_token_decoder_blocks(self):
        block_list = []
        for _ in range(self.num_layers):
            block_list.append(CommonBlock(self.hidden_dim, self.num_heads, self.dropout_param))
            block_list.append(OutputTokenBlock(self.hidden_dim, self.ff_dim, self.num_heads, self.dropout_param))
        return block_list

    def _reset_parameters(self):
        """Perform xavier weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
            self,
            batch_io_parser_output: list[list[dict]],
            task_type: str,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Performs on output token decoder forward pass to calculate the E2 -> Embedding for output tokens
        N = batch size
        S = source sequence length
        E = embedding dimensionality
        V = vocabulary size

        :param task_type: Type of task. ie: func_to_nl_translation.
        :param batch_io_parser_output: batch of io_parser_output. Shape (N, S)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :param future_mask: An attention mask to ignore future-tokens in the target input. Shape (T, T)
        :return: E2 -> Embedding for output tokens. Shape (N, S, E)
        """
        # (batch_size, sequence_length, hidden_dim)
        x = self.embeddings_manager.get_batch_combined_embeddings(batch_io_parser_output, task_type) \
            * math.sqrt(self.hidden_dim)  # (N, S, E)
        x = self.output_token_decoder_dropout(x)

        for decoder_block in self.output_token_decoder_blocks:
            if isinstance(decoder_block, CommonBlock):
                x = decoder_block(x, src_padding_mask, future_mask)
            elif isinstance(decoder_block, OutputTokenBlock):
                x = decoder_block(x)

        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()


class TestCategoryMapDecoder(unittest.TestCase):

    def test_output_token_decoder(self):
        """
        Test three forward pass of output token decoder and check if output tensor shape is as expected.
        :return: None
        """
        batch_size = 2
        num_heads = 8
        max_decoding_length = 10
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 2
        dropout_p = 0.1
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value
        sentences = [
            "Hello my name is Joaa and my parents gave the name Joaa.",
            "Hello my name is Prattoy and my grandma gave the name Prattoy."
        ]
        with torch.no_grad():
            output_token_decoder = OutputTokenDecoder(
                embeddings_manager=EmbeddingsManager(
                    batch_size=batch_size,
                    n_heads=num_heads,
                    max_sequence_length=max_decoding_length,
                    with_mask=False,
                ),
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout_p=dropout_p,
            )
            output_token_decoder._reset_parameters()
            output_token_decoder.eval()

            batch_io_parser = BatchBuilder.get_batch_io_parser_output(sentences, True, 1)
            future_mask = BatchBuilder.construct_future_mask(1)
            for i in range(3):
                index = i + 1
                output_token_decoder_output = output_token_decoder.forward(
                    batch_io_parser,
                    task_type,
                    future_mask=future_mask,
                )

                # Teacher forcing
                batch_io_parser = BatchBuilder.get_batch_io_parser_output(sentences, True, index + 1)
                future_mask = BatchBuilder.construct_future_mask(index + 1)

                self.assertEqual(output_token_decoder_output.shape, (batch_size, index, hidden_dim))


if __name__ == "__main__":
    unittest.main()
