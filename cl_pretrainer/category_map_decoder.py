import math
import unittest
from typing import Optional

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.category_map_block import CategoryMapBlock
from cl_pretrainer.common_block import CommonBlock
from cl_pretrainer.pre_trainer_utils import PreTrainerUtils
from embeddings_manager.embeddings_manager import EmbeddingsManager


class CategoryMapDecoder(nn.Module):
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
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.category_map_decoder_dropout = nn.Dropout(p=0.1)
        self.category_map_decoder_blocks = nn.ModuleList(self._create_category_map_decoder_blocks())

        # Move to device
        self.category_map_decoder_dropout.to(self.device)

    def _create_category_map_decoder_blocks(self):
        block_list = []
        for _ in range(self.num_layers):
            block_list.append(CommonBlock(self.hidden_dim, self.num_heads, self.dropout_param))
            block_list.append(CategoryMapBlock(self.hidden_dim, self.ff_dim, self.num_heads, self.dropout_param))
        return block_list

    def _reset_parameters(self):
        """Perform xavier weight initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(
            self,
            batch_io_parser_output: list[list[dict]],
            task_types: list[str],
            src_padding_mask: Optional[torch.BoolTensor] = None,
            future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Performs on category map decoder forward pass to calculate the E1 -> Embedding for category
        N = batch size
        S = source sequence length
        E = embedding dimensionality
        V = vocabulary size

        :param task_types: list of Type of task. ie: [func_to_nl_translation, ...]
        :param batch_io_parser_output: batch of io_parser_output. Shape (N, S)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :param future_mask: An attention mask to ignore future-tokens in the target input. Shape (T, T)
        :return: E1 -> Embedding for category. Shape (N, S, E)
        """
        # (batch_size, sequence_length, hidden_dim)
        x, cross_attention_mask, batch_of_encoder_hidden_states = self.embeddings_manager.get_batch_combined_embeddings_with_mask(
            batch_io_parser_output,
            task_types
        )
        x = x * math.sqrt(self.hidden_dim)  # (N, S, E)

        x = self.category_map_decoder_dropout(x)

        for decoder_block in self.category_map_decoder_blocks:
            if isinstance(decoder_block, CommonBlock):
                x = decoder_block.forward(x, src_padding_mask, future_mask)
            elif isinstance(decoder_block, CategoryMapBlock):
                x = decoder_block.forward(
                    x,
                    PreTrainerUtils.create_function_param_token_infos(
                        x,
                        cross_attention_mask,
                        batch_of_encoder_hidden_states,
                        shift_right=True,
                    ),
                )

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

    def test_category_map_decoder(self):
        """
        Test three forward pass of category map decoder and check if output tensor shape is as expected.
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
            category_map_decoder = CategoryMapDecoder(
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
            category_map_decoder._reset_parameters()
            category_map_decoder.eval()

            batch_io_parser = BatchBuilder.get_batch_io_parser_output(sentences, True, 1)
            future_mask = BatchBuilder.construct_future_mask(1)
            for i in range(3):
                index = i + 1
                category_map_decoder_output = category_map_decoder.forward(
                    batch_io_parser,
                    task_type,
                    future_mask=future_mask,
                )

                # Teacher forcing
                batch_io_parser = BatchBuilder.get_batch_io_parser_output(sentences, True, index + 1)
                future_mask = BatchBuilder.construct_future_mask(index + 1)

                self.assertEqual(category_map_decoder_output.shape, (batch_size, index, hidden_dim))


if __name__ == "__main__":
    unittest.main()
