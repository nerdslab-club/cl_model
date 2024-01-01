import unittest

import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from category_router.category_router import CategoryRouter
from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.category_map_classification_head import CategoryMapClassificationHead
from cl_pretrainer.category_map_decoder import CategoryMapDecoder
from cl_pretrainer.output_token_decoder import OutputTokenDecoder
from cl_pretrainer.pre_trainer_utils import PreTrainerUtils
from embeddings_manager.embeddings_manager import EmbeddingsManager
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


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
        """
        Test 10 complete decoding cycle for both category and output probability
        This is inference example that's why using the for loop over the sequence
        :return: No
        """
        batch_size = 3
        num_heads = 8
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 2
        dropout_p = 0.1
        max_decoding_length = 5
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value
        sentences = [
            "The quick brown fox jumps over the lazy dog in the meadow",
            "Adding 3 plus 2 equals ##addition(3,2)",
            "##average([2,3,4,5]) is the mean value of [2,3,4,5]",
        ]

        # Initialization of corpus
        corpus_io_parser_output = BatchBuilder.get_batch_io_parser_output(sentences, True, 15)

        batch_io_parser_output = BatchBuilder.get_batch_io_parser_output(sentences, True, 1)
        future_mask = BatchBuilder.construct_future_mask(1)

        # Initialize category vocabulary builder instance
        category_vocab_builder = CategoryVocabBuilder(corpus_io_parser_output)
        category_vocab_size = len(category_vocab_builder.category_vocab_item_to_index.keys())

        print(f"Output token classification head count:"
              f" {len(category_vocab_builder.index_to_output_token_classification_head_vocab_item.keys())}\n"
              f"Output token classification head category type:"
              f" {category_vocab_builder.index_to_output_token_classification_head_vocab_item}")

        # Initialize output vocabulary builder instance
        output_vocab_builder = OutputVocabBuilder(
            corpus_of_io_parser_output=corpus_io_parser_output,
            index_to_output_token_classification_head_vocab_item=
            category_vocab_builder.index_to_output_token_classification_head_vocab_item
        )
        output_vocabularies = output_vocab_builder.index_to_output_vocabularies
        print(f"Output vocabularies count: {len(output_vocabularies.keys())}")

        with torch.no_grad():
            cl_pre_trainer = ClPreTrainer(
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                max_decoding_length=max_decoding_length,
                dropout_p=dropout_p,
                category_vocab_size=category_vocab_size,
                index_to_output_vocabularies=output_vocabularies,
            )
            cl_pre_trainer.eval()

            for index in range(1, 11):
                # Staring left side for category map
                e_one = cl_pre_trainer.category_map_decoder.forward(
                    batch_io_parser_output=batch_io_parser_output,
                    task_type=task_type,
                    future_mask=future_mask,
                )

                category_probability, _ = cl_pre_trainer.category_map_classification_head.forward(e_one)
                predicted_category_map = category_vocab_builder.batch_decode(category_probability.tolist())
                print(f"Predicted category probability values:"
                      f" {predicted_category_map}")

                # Starting right side for output token
                e_two = cl_pre_trainer.output_token_decoder.forward(
                    batch_io_parser_output=batch_io_parser_output,
                    task_type=task_type,
                    future_mask=future_mask,
                )

                predicted_io_parser_output_without_token = PreTrainerUtils.convert_category_map_into_io_parser_output_without_token(
                    batch_category_map=predicted_category_map,
                )
                batch_route_ids = category_vocab_builder.batch_encoder_output_token_classification_head_vocab_items(
                    batch_io_parser_output=predicted_io_parser_output_without_token,
                )

                output_probability = cl_pre_trainer.category_router.forward(
                    e_two=e_two,
                    batch_route_ids=batch_route_ids,
                    is_hub=False,
                )
                predicted_output_token = output_vocab_builder.batch_decode_for_inference(output_probability)
                print(f"Predicted token values:"
                      f" {predicted_output_token}")

                # Teacher forcing
                batch_io_parser_output = BatchBuilder.get_batch_io_parser_output(sentences, True, index + 1)
                future_mask = BatchBuilder.construct_future_mask(index + 1)

                # Asset the shapes and predictions
                self.assertEqual(e_one.shape, (batch_size, index, hidden_dim))
                self.assertEqual(category_probability.shape, (batch_size, index))
                # check if item is in vocab range for category map classification head
                for item_list in category_probability.tolist():
                    for item in item_list:
                        self.assertTrue(0 <= item < category_vocab_size)

                self.assertEqual(e_two.shape, (batch_size, index, hidden_dim))
                self.assertEqual((len(output_probability), len(output_probability[0])), (batch_size, index))
                # check if item is in vocab range
                for item_list in output_probability:
                    for route, item in item_list:
                        current_vocabulary = output_vocabularies[route]
                        vocab_size = len(current_vocabulary[output_vocab_builder.INDEX_TO_OUTPUT].keys())
                        self.assertTrue(0 <= item < vocab_size)


if __name__ == "__main__":
    unittest.main()
