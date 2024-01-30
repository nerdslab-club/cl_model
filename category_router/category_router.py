import unittest
from typing import Any

import torch
from torch import nn

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.output_token_classification_head import OutputTokenClassificationHead
from cl_pretrainer.output_token_decoder import OutputTokenDecoder
from embeddings_manager.embeddings_manager import EmbeddingsManager
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


class CategoryRouter(nn.Module):
    ROUTE_CLASSIFICATION_HEAD = "route_classification_head"
    OUTPUT_LOGITS = "output_logits"
    OUTPUT_PROBABILITY = "output_probability"
    SOFTMAX_PROBABILITY = "softmax_probability"

    def __init__(
            self,
            index_to_output_vocabularies: dict[int, dict],
            hidden_dim: int,
            ff_dim: int,
            dropout_p: float,

    ):
        super().__init__()
        self.index_to_route = index_to_output_vocabularies

        for index, route in self.index_to_route.items():
            route_vocabulary_index_to_vocab = route[OutputVocabBuilder.INDEX_TO_OUTPUT]
            vocab_size = len(route_vocabulary_index_to_vocab.keys())

            output_classification_head = OutputTokenClassificationHead(
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                dropout_p=dropout_p,
                vocab_size=vocab_size,
                output_token_classification_vocab_item_index=index,
                output_token_classification_vocab_item=route[
                    OutputVocabBuilder.OUTPUT_TOKEN_CLASSIFICATION_HEAD_VOCAB_ITEM
                ]
            )
            route[CategoryRouter.ROUTE_CLASSIFICATION_HEAD] = output_classification_head
            self.index_to_route[index] = route

    def forward(
            self,
            e_two: torch.Tensor,
            batch_route_ids: list[list[int]],
            is_hub=False,
    ) -> dict[int, dict[str, Any]] | list[list[tuple[int, Any]]]:
        """
        Pass each 768 embeddings tensor in its own classification head to get the prediction

        :param is_hub: run in hub mode if the flag is true
        :param e_two: embeddings for output token
        :param batch_route_ids: Batch integer index of the route
        :return: batch of tuple of (route_id, output_probability) and batch logits
        """
        if is_hub:
            output_logits_map = {}
            for index, route in self.index_to_route.items():
                classification_head = route[CategoryRouter.ROUTE_CLASSIFICATION_HEAD]
                current_output_probability, current_logits = classification_head.forward(e_two)
                output_logits_map[index] = {
                    CategoryRouter.OUTPUT_LOGITS: current_logits,
                    CategoryRouter.OUTPUT_PROBABILITY: current_output_probability,
                    CategoryRouter.SOFTMAX_PROBABILITY: torch.nn.functional.softmax(current_logits, dim=-1)
                }
            return output_logits_map

        batch_result = []
        for i, route_ids in enumerate(batch_route_ids):
            sequence_result = []
            e_two_sequence = e_two[i]
            for j, route_id in enumerate(route_ids):
                e_two_item = e_two_sequence[j]
                classification_head = self.index_to_route[route_id][CategoryRouter.ROUTE_CLASSIFICATION_HEAD]
                output_probability, _ = classification_head.forward(e_two_item)
                sequence_result.append((route_id, output_probability.squeeze().item()))
            batch_result.append(sequence_result)
        return batch_result

    def load_output_classification_head(self, index: int, state_dict: dict):
        """
        Given the trained state dict of the output classification head it reload the same state
        :param index: Index of the output classification head
        :param state_dict: Trained state dict of the output classification head
        :return: None
        """
        route = self.index_to_route[index]
        output_classification_head = route[CategoryRouter.ROUTE_CLASSIFICATION_HEAD]
        output_classification_head.load_saved_model_from_state_dict(state_dict)
        route[CategoryRouter.ROUTE_CLASSIFICATION_HEAD] = output_classification_head
        self.index_to_route[index] = route

    def load_all_output_classification_head(self, all_state_dict: dict[int, dict]):
        """
        Given the trained state dict of all the output classification head it reload them all
        :param all_state_dict: Trained state dict of all the output classification heads
        :return: None
        """
        for index, _ in self.index_to_route.items():
            self.load_output_classification_head(index, all_state_dict[index])

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()


class TestCategoryRouter(unittest.TestCase):

    def test_category_router(self):
        """
        Test three forward pass of output token decoder with multiple classification head
        and check if output tensor shape is as expected.
        :return: No
        """
        batch_size = 3
        num_heads = 8
        max_decoding_length = 10
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 2
        dropout_p = 0.1
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value
        sentences = [
            "Hello my name is Joaa and my parents gave the name Joaa.",
            "Hello my name is Prattoy and my grandma gave the name Prattoy.",
            "##addition(3,2)",
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

            # Initialization
            batch_io_parser = BatchBuilder.get_batch_io_parser_output(sentences, True, 1)
            future_mask = BatchBuilder.construct_future_mask(1)
            corpus_io_parser_output = BatchBuilder.get_batch_io_parser_output(sentences, True, 4)

            # Create category vocabulary builder instance
            category_vocab_builder = CategoryVocabBuilder(corpus_io_parser_output)
            print(f"Output token classification head count:"
                  f" {len(category_vocab_builder.index_to_output_token_classification_head_vocab_item.keys())}")
            print(f"Output token classification head category type:"
                  f" {category_vocab_builder.index_to_output_token_classification_head_vocab_item}")

            # Create output vocabulary builder instance
            output_vocab_builder = OutputVocabBuilder(
                corpus_of_io_parser_output=corpus_io_parser_output,
                index_to_output_token_classification_head_vocab_item=
                category_vocab_builder.index_to_output_token_classification_head_vocab_item
            )
            output_vocabularies = output_vocab_builder.index_to_output_vocabularies
            print(f"Output vocabularies count: {len(output_vocabularies.keys())}")

            # Create category router instance
            category_router = CategoryRouter(
                index_to_output_vocabularies=output_vocabularies,
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                dropout_p=dropout_p,
            )
            router_items = category_router.index_to_route
            print(f"Output classification head count: {len(router_items.keys())}")

            for i in range(4):
                index = i + 1
                e_two = output_token_decoder.forward(
                    batch_io_parser_output=batch_io_parser,
                    task_types=task_type,
                    future_mask=future_mask,
                )

                batch_route_ids = category_vocab_builder.batch_encoder_output_token_classification_head_vocab_items(
                    batch_io_parser,
                )
                category_router_output = category_router.forward(
                    e_two=e_two,
                    batch_route_ids=batch_route_ids,
                )
                print(f"category router output shape: [{len(category_router_output)}, {len(category_router_output[0])}]")
                print(f"Predicted token values:"
                      f" {output_vocab_builder.batch_decode_for_inference(category_router_output)}")

                # Teacher forcing
                batch_io_parser = BatchBuilder.get_batch_io_parser_output(sentences, False, index + 1)
                future_mask = BatchBuilder.construct_future_mask(index + 1)

                self.assertEqual(e_two.shape, (batch_size, index, hidden_dim))
                self.assertEqual((len(category_router_output), len(category_router_output[0])), (batch_size, index))

                # check if item is in vocab range
                for item_list in category_router_output:
                    for route, item in item_list:
                        current_vocabulary = output_vocabularies[route]
                        vocab_size = len(current_vocabulary[output_vocab_builder.INDEX_TO_OUTPUT].keys())
                        self.assertTrue(0 <= item < vocab_size)


if __name__ == "__main__":
    unittest.main()
