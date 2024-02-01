import unittest

import torch
from torch import nn

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.output_token_decoder import OutputTokenDecoder
from cl_pretrainer.rmsnorm_torch import RMSNorm
from cl_pretrainer.swiglu_activation import SwiGLU
from embeddings_manager.embeddings_manager import EmbeddingsManager
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder, OutputTokenClassificationHeadVocabItem
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


class OutputTokenClassificationHead(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            ff_dim: int,
            vocab_size: int,
            dropout_p: float,
            output_token_classification_vocab_item_index: int,
            output_token_classification_vocab_item: OutputTokenClassificationHeadVocabItem,
    ):
        super().__init__()
        self.output_token_classification_vocab_item_index = output_token_classification_vocab_item_index
        self.output_token_classification_vocab_item = output_token_classification_vocab_item
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.output_token_classification_head_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.PReLU(),
            nn.Linear(ff_dim, hidden_dim),
        ).to(self.device)

        # Dropout is also known as regularization
        self.output_token_classification_head_dropout = nn.Dropout(p=dropout_p)
        # Normalizing layer for propagating the token values
        self.output_token_classification_head_layer_norm = RMSNorm(hidden_dim)

        # Projecting the hidden dimension into vocabulary size,
        # so that we can use softmax and find specific word probability.
        self.output_token_classification_head_output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Move to device
        self.output_token_classification_head_dropout.to(self.device)
        self.output_token_classification_head_layer_norm.to(self.device)
        self.output_token_classification_head_output_layer.to(self.device)

    def forward(self, e_two: torch.Tensor):
        # Move to device
        e_two = e_two.to(self.device)
        # Feed forward layers
        output = self.output_token_classification_head_dropout(
            self.output_token_classification_head_feed_forward(e_two),
        )
        e_two = self.output_token_classification_head_layer_norm(e_two + output)

        # Linear layer, output shape(batch_size, sequence_length, output_vocab_size)
        logits = self.output_token_classification_head_output_layer(e_two)

        # Argmax, output shape(batch_size, sequence_length)
        output_probability = logits.argmax(dim=-1)
        return output_probability, logits

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()


class TestOutputTokenClassificationHead(unittest.TestCase):

    def test_output_token_classification_head(self):
        """
        Test three forward pass of output token decoder with classification head
        and check if output tensor shape is as expected.
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

            batch_io_parser = BatchBuilder.get_batch_io_parser_output(sentences, False, 1)
            future_mask = BatchBuilder.construct_future_mask(1)

            corpus_io_parser_output = BatchBuilder.get_batch_io_parser_output(sentences, False, 4)

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
            output_vocabulary = output_vocab_builder.index_to_output_vocabularies[0]
            vocab_size = len(output_vocabulary[OutputVocabBuilder.INDEX_TO_OUTPUT].keys())
            print(f"Vocab size: {vocab_size}")

            # Create output token classification head instance
            output_token_classification_head = OutputTokenClassificationHead(
                hidden_dim=hidden_dim,
                ff_dim=ff_dim,
                dropout_p=dropout_p,
                vocab_size=vocab_size,
                output_token_classification_vocab_item_index=output_vocabulary[OutputVocabBuilder.INDEX],
                output_token_classification_vocab_item=output_vocabulary[
                    OutputVocabBuilder.OUTPUT_TOKEN_CLASSIFICATION_HEAD_VOCAB_ITEM
                ]
            )
            for i in range(3):
                index = i + 1
                output_token_decoder_output = output_token_decoder.forward(
                    batch_io_parser,
                    task_type,
                    future_mask=future_mask,
                )

                output_token_classification_head_output, _ = output_token_classification_head.forward(
                    output_token_decoder_output,
                )

                print(f"Output token classification head output shape: {output_token_classification_head_output.shape}")
                output_token_classification_head_output_list = output_token_classification_head_output.tolist()
                list_of_ids = [
                    [(0, x) for x in sublist]
                    for sublist in output_token_classification_head_output_list
                ]
                print(f"Predicted token values:"
                      f" {output_vocab_builder.batch_decode_for_inference(list_of_ids)}")

                # Teacher forcing
                batch_io_parser = BatchBuilder.get_batch_io_parser_output(sentences, False, index + 1)
                future_mask = BatchBuilder.construct_future_mask(index + 1)

                self.assertEqual(output_token_decoder_output.shape, (batch_size, index, hidden_dim))
                self.assertEqual(output_token_classification_head_output.shape, (batch_size, index))

                # check if item is in vocab range
                for item_list in output_token_classification_head_output.tolist():
                    for item in item_list:
                        self.assertTrue(0 <= item < vocab_size)


if __name__ == "__main__":
    unittest.main()
