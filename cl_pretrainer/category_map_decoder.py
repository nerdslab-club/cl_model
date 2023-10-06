import math
from typing import Optional

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from cl_pretrainer.category_map_block import CategoryMapBlock
from cl_pretrainer.common_block import CommonBlock
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

        self.category_map_decoder_dropout = nn.Dropout(p=0.1)
        self.category_map_decoder_blocks = nn.ModuleList(self._create_category_map_decoder_blocks())

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
            task_type: str,
            src_padding_mask: Optional[torch.BoolTensor] = None,
            future_mask: Optional[torch.BoolTensor] = None,
    ):
        """
        Performs on category map decoder forward pass to calculate the E1 -> Embedding for category
        N = batch size
        S = source sequence length
        E = embedding dimensionality
        V = vocabulary size

        :param task_type: Type of task. ie: func_to_nl_translation.
        :param batch_io_parser_output: batch of io_parser_output. Shape (N, S)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :param future_mask: An attention mask to ignore future-tokens in the target input. Shape (T, T)
        :return: E1 -> Embedding for category. Shape (N, S, E)
        """
        # (batch_size, sequence_length, hidden_dim)
        x, m = self.embeddings_manager.get_batch_combined_embeddings_with_mask(batch_io_parser_output, task_type)
        x = x * math.sqrt(self.hidden_dim)  # (N, S, E)

        x = self.category_map_decoder_dropout(x)

        for decoder_block in self.category_map_decoder_blocks:
            if isinstance(decoder_block, CommonBlock):
                x = decoder_block.forward(x, src_padding_mask, future_mask)
            elif isinstance(decoder_block, CategoryMapBlock):
                # TODO complete CategoryMapBlock class.
                x = decoder_block.forward(x, )

        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
