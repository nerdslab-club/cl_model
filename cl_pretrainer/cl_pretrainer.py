import unittest

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from category_router.category_router import CategoryRouter
from cl_pretrainer.category_map_classification_head import CategoryMapClassificationHead
from cl_pretrainer.category_map_decoder import CategoryMapDecoder
from cl_pretrainer.output_token_decoder import OutputTokenDecoder
from embeddings_manager.embeddings_manager import EmbeddingsManager


class ClPreTrainer(nn.Module):
    def __init__(
            self,
            batch_size: int,
            hidden_dim: int,
            ff_dim: int,
            num_heads: int,
            num_layers: int,
            max_decoding_length: int,
            dropout_p: float,
            category_vocab_size: int,
            index_to_output_vocabularies: dict[int, dict]
    ):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.ff_dim = ff_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_decoding_length = max_decoding_length
        self.dropout_p = dropout_p
        self.category_vocab_size = category_vocab_size
        self.index_to_output_vocabularies = index_to_output_vocabularies

        # Creating embeddings manager instance
        self.embeddings_manager = EmbeddingsManager(
            batch_size=batch_size,
            n_heads=num_heads,
            max_sequence_length=max_decoding_length,
            with_mask=False,
        )

        # Creating category map decoder instance
        self.category_map_decoder = CategoryMapDecoder(
            embeddings_manager=self.embeddings_manager,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_p=dropout_p,
        )
        self.category_map_decoder.eval()

        # Creating category map classification head instance
        self.category_map_classification_head = CategoryMapClassificationHead(
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            dropout_p=dropout_p,
            vocab_size=category_vocab_size,
        )
        self.category_map_classification_head.eval()

        # Creating output token decoder instance
        self.output_token_decoder = OutputTokenDecoder(
            embeddings_manager=self.embeddings_manager,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_p=dropout_p,
        )
        self.output_token_decoder.eval()

        # Creating Category router instance
        self.category_router = CategoryRouter(
                index_to_output_vocabularies=self.index_to_output_vocabularies,
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                dropout_p=dropout_p,
        )
        self.category_router.eval()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()


class TestClPreTrainer(unittest.TestCase):
    def test_cl_pre_trainer(self):

        pass


if __name__ == "__main__":
    unittest.main()
