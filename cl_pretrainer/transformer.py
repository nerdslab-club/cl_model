import random
import unittest

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from cl_data.src.constants import TaskTypes, SpecialTokens
from cl_pretrainer.batch_builder import BatchBuilder
from embeddings_manager.embeddings_manager import EmbeddingsManager
from response_parser.simple_response_parser import SimpleResponseParser
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from transformer_utils import construct_future_mask
from vocabulary_builder.simple_vocabulary_builder import SimpleVocabBuilder


class Transformer(nn.Module):
    def __init__(
        self,
        batch_size: int,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        max_decoding_length: int,
        vocab_size: int,
        dropout_p: float,
    ):
        super().__init__()
        # Because the encoder embedding, and decoder embedding and decoder pre-softmax transformeation share embeddings
        # weights, initialize one here and pass it on.
        self.batch_size = batch_size
        self.embeddings_manager = EmbeddingsManager(
            batch_size=batch_size,
            n_heads=num_heads,
            max_sequence_length=max_decoding_length,
            with_mask=False,
        )
        self.encoder = TransformerEncoder(
            self.embeddings_manager, hidden_dim, ff_dim, num_heads, num_layers, dropout_p
        )
        self.decoder = TransformerDecoder(
            self.embeddings_manager,
            hidden_dim,
            ff_dim,
            num_heads,
            num_layers,
            vocab_size,
            dropout_p,
        )

        self.max_decoding_length = max_decoding_length
        self.hidden_dim = hidden_dim
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


class TestTransformer(unittest.TestCase):
    def test_transformer_inference(self):
        seed = 0
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        batch_size = 1
        hidden_dim = 768
        max_decoding_length = 10
        max_encoding_length = 16
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value

        # Create (shared) vocabulary and special token indices given a dummy corpus
        corpus = [
            "Hello my name is Joris and I was born with the name Joris.",
            "Dit is een Nederlandse zin.",
        ]

        with torch.no_grad():
            # Prepare encoder input, mask and generate output hidden states
            encoder_input = BatchBuilder.get_batch_io_parser_output(corpus, True, max_encoding_length)
            src_padding_mask = BatchBuilder.construct_padding_mask(encoder_input)

            # Prepare decoder input and mask and start decoding
            decoder_input = BatchBuilder.get_batch_io_parser_output(corpus, True, 1)
            future_mask = construct_future_mask(seq_len=1)

            # Create vocabulary
            vocabulary = SimpleVocabBuilder(encoder_input)
            en_vocab_size = len(vocabulary.vocab_item_to_index.items())
            padding_token_id = vocabulary.vocab_item_to_index[SimpleVocabBuilder.get_special_vocab_item(SpecialTokens.PADDING)]
            beginning_token_id = vocabulary.vocab_item_to_index[SimpleVocabBuilder.get_special_vocab_item(SpecialTokens.BEGINNING)]
            ending_token_id = vocabulary.vocab_item_to_index[SimpleVocabBuilder.get_special_vocab_item(SpecialTokens.ENDING)]
            print(
                f"Vocab size {en_vocab_size}\n"
                f"Padding token id {padding_token_id}\n"
                f"BOS token id {beginning_token_id}\n"
                f"EOS token id {ending_token_id}"
            )

            transformer = Transformer(
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                ff_dim=2048,
                num_heads=8,
                num_layers=6,
                max_decoding_length=max_decoding_length,
                vocab_size=en_vocab_size,
                dropout_p=0.1,
            )
            transformer.eval()

            encoder_output = transformer.encoder.forward(
                encoder_input, task_type, src_padding_mask=src_padding_mask
            )
            print(f"Encoder output shape: {encoder_output.shape}")
            self.assertEqual(torch.any(torch.isnan(encoder_output)), False)

            predicted_token_list = torch.IntTensor([[beginning_token_id], [beginning_token_id]])
            for i in range(transformer.max_decoding_length):
                index = i + 1
                decoder_output = transformer.decoder.forward(
                    decoder_input,
                    task_type,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )
                # Take the argmax over the softmax of the last token to obtain the next-token prediction
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)

                predicted_token_list = torch.cat((predicted_token_list, predicted_tokens), dim=-1)

                # Teacher forcing
                decoder_input = BatchBuilder.get_batch_io_parser_output(corpus, True, index + 1)
                future_mask = BatchBuilder.construct_future_mask(index + 1)

        print(f"Decoder output shape: {decoder_output.shape}")

        # Printing predicted tokens
        SimpleResponseParser.print_response_to_console(vocabulary.batch_decode(predicted_token_list.tolist()))
        self.assertEqual((len(decoder_input), len(decoder_input[0])), (2, transformer.max_decoding_length + 1))


if __name__ == "__main__":
    unittest.main()
