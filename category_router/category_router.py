import torch
from torch import nn, Tensor

from cl_pretrainer.output_token_classification_head import OutputTokenClassificationHead
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


class CategoryRouter(nn.Module):
    ROUTE_CLASSIFICATION_HEAD = "route_classification_head"

    def __init__(
            self,
            output_vocab_builder: OutputVocabBuilder,
            hidden_dim: int,
            ff_dim: int,
            dropout_p: int,

    ):
        super().__init__()
        self.output_vocab_builder = output_vocab_builder
        self.index_to_route = self.output_vocab_builder.index_to_output_vocabularies

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

        self.route_to_index = {v: k for k, v in self.index_to_route.items()}

    def forward(
            self,
            e_two: torch.Tensor,
            batch_route_ids: list[list[int]],
    ) -> list[list[tuple[int, Tensor]]]:
        """
        Pass each 768 embeddings tensor in its own classification head to get the prediction

        :param e_two: embeddings for output token
        :param batch_route_ids: Batch integer index of the route
        :return: batch of tuple (route_id, output_probability)
        """
        batch_result = []
        for i, route_ids in enumerate(batch_route_ids):
            sequence_result = []
            e_two_sequence = e_two[i]
            for j, route_id in enumerate(route_ids):
                e_two_item = e_two_sequence[j]
                classification_head = self.index_to_route[route_id][CategoryRouter.ROUTE_CLASSIFICATION_HEAD]
                output_probability = classification_head.forward(e_two_item)
                sequence_result.append((route_id, output_probability))
            batch_result.append(sequence_result)
        return batch_result

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
