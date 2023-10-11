import unittest

import torch
from torch import nn

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.category_map_decoder import CategoryMapDecoder
from embeddings_manager.embeddings_manager import EmbeddingsManager
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder


class CategoryMapClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, vocab_size: int, dropout_p: float):
        super().__init__()

        self.category_map_classification_head_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        # Dropout is also known as regularization
        self.category_map_classification_head_dropout = nn.Dropout(p=dropout_p)
        # Normalizing layer for propagating the token values
        self.category_map_classification_head_layer_norm = nn.LayerNorm(hidden_dim)

        # Projecting the hidden dimension into vocabulary size,
        # so that we can use softmax and find specific word probability.
        self.category_map_classification_head_output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, e_one: torch.Tensor):
        # Feed forward layers
        output = self.category_map_classification_head_dropout(
            self.category_map_classification_head_feed_forward(e_one),
        )
        e_one = self.category_map_classification_head_layer_norm(e_one + output)

        # Linear layer, output shape(batch_size, sequence_length, category_vocab_size)
        logits = self.category_map_classification_head_output_layer(e_one)

        # Argmax, output shape(batch_size, sequence_length)
        category_probability = logits.argmax(dim=-1)
        return category_probability, logits

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()


class TestCategoryMapClassificationHead(unittest.TestCase):

    def test_category_map_classification_head(self):
        """
        Test three forward pass of category map decoder and check if output tensor shape is as expected.
        :return: None
        """
        batch_size = 3
        num_heads = 8
        max_decoding_length = 10
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 1
        dropout_p = 0.1
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value
        sentences = [
            "Hello my name is Joaa and my parents gave the name Joaa.",
            "Hello my name is Prattoy and my grandma gave the name Prattoy.",
            "##addition(3,2)",
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

            # Create category vocabulary builder instance
            corpus_io_parser_output = BatchBuilder.get_batch_io_parser_output(sentences, True, 16)
            category_vocab_builder = CategoryVocabBuilder(corpus_io_parser_output)
            vocab_size = len(category_vocab_builder.category_vocab_item_to_index.keys())
            print(f"Vocab size: {vocab_size}")

            # Create category map classification head instance
            category_map_classification_head = CategoryMapClassificationHead(
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                dropout_p=dropout_p,
                vocab_size=vocab_size,
            )
            for i in range(5):
                index = i + 1
                category_map_decoder_output = category_map_decoder.forward(
                    batch_io_parser,
                    task_type,
                    future_mask=future_mask,
                )

                category_map_classification_head_output = category_map_classification_head.forward(
                    category_map_decoder_output,
                )
                print(f"category map decoder input: {batch_io_parser}")
                print(f"Category map classification head output shape: {category_map_classification_head_output.shape}")
                print(f"Predicted token values:"
                      f" {category_vocab_builder.batch_decode(category_map_classification_head_output.tolist())}")

                # Teacher forcing
                batch_io_parser = BatchBuilder.get_batch_io_parser_output(sentences, True, index + 1)
                future_mask = BatchBuilder.construct_future_mask(index + 1)

                self.assertEqual(category_map_decoder_output.shape, (batch_size, index, hidden_dim))
                self.assertEqual(category_map_classification_head_output.shape, (batch_size, index))

                # check if item is in vocab range
                for item_list in category_map_classification_head_output.tolist():
                    for item in item_list:
                        self.assertTrue(0 <= item < vocab_size)


if __name__ == "__main__":
    unittest.main()
