import random
import unittest
from typing import Optional

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_

from vocabulary import Vocabulary
from encoder import TransformerEncoder
from decoder import TransformerDecoder
from transformer_utils import construct_future_mask


class Transformer(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        max_decoding_length: int,
        vocab_size: int,
        padding_idx: int,
        bos_idx: int,
        dropout_p: float,
        tie_output_to_embedding: Optional[bool] = None,
    ):
        super().__init__()
        # Because the encoder embedding, and decoder embedding and decoder pre-softmax transformeation share embeddings
        # weights, initialize one here and pass it on.
        self.embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.encoder = TransformerEncoder(
            self.embed, hidden_dim, ff_dim, num_heads, num_layers, dropout_p
        )
        self.decoder = TransformerDecoder(
            self.embed,
            hidden_dim,
            ff_dim,
            num_heads,
            num_layers,
            vocab_size,
            dropout_p,
            tie_output_to_embedding,
        )

        self.padding_idx = padding_idx
        self.bos_idx = bos_idx
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

        # Create (shared) vocabulary and special token indices given a dummy corpus
        corpus = [
            "Hello my name is Joris and I was born with the name Joris.",
            "Dit is een Nederlandse zin.",
        ]
        en_vocab = Vocabulary(corpus)
        en_vocab_size = len(en_vocab.token2index.items())
        print(
            f"Vocab size {en_vocab_size}\n"
            f"Padding token id {en_vocab.token2index[en_vocab.PAD]}\n"
            f"BOS token id {en_vocab.token2index[en_vocab.BOS]}\n"
            f"EOS token id {en_vocab.token2index[en_vocab.EOS]}"
        )
        with torch.no_grad():
            transformer = Transformer(
                hidden_dim=512,
                ff_dim=2048,
                num_heads=8,
                num_layers=6,
                max_decoding_length=10,
                vocab_size=en_vocab_size,
                padding_idx=en_vocab.token2index[en_vocab.PAD],
                bos_idx=en_vocab.token2index[en_vocab.BOS],
                dropout_p=0.1,
                tie_output_to_embedding=True,
            )
            transformer.eval()

            # Prepare encoder input, mask and generate output hidden states
            encoder_input = torch.IntTensor(
                en_vocab.batch_encode(corpus, add_special_tokens=False)
            )

            src_padding_mask = encoder_input != transformer.padding_idx
            encoder_output = transformer.encoder.forward(
                encoder_input, src_padding_mask=src_padding_mask
            )
            print(f"Encoder input shape: {encoder_output.shape}")
            print(f"Encoder output shape: {encoder_output.shape}")

            self.assertEqual(torch.any(torch.isnan(encoder_output)), False)

            # Prepare decoder input and mask and start decoding
            # Initializing two array for batch
            decoder_input = torch.IntTensor(
                [[transformer.bos_idx], [transformer.bos_idx]]
            )
            future_mask = construct_future_mask(seq_len=1)
            for i in range(transformer.max_decoding_length):
                decoder_output = transformer.decoder(
                    decoder_input,
                    encoder_output,
                    src_padding_mask=src_padding_mask,
                    future_mask=future_mask,
                )
                # Take the argmax over the softmax of the last token to obtain the next-token prediction
                predicted_tokens = torch.argmax(
                    decoder_output[:, -1, :], dim=-1
                ).unsqueeze(1)
                print(f"Predicted token: {predicted_tokens}")

                # Append the prediction to the already decoded tokens and construct the new mask
                decoder_input = torch.cat((decoder_input, predicted_tokens), dim=-1)
                future_mask = construct_future_mask(decoder_input.shape[1])

        print(f"Decoder input shape: {decoder_input.shape}")
        print(f"Decoder output shape: {decoder_output.shape}")
        self.assertEqual(decoder_input.shape, (2, transformer.max_decoding_length + 1))
        # see test_one_layer_transformer_decoder_inference in decoder.py for more information. with num_layers=1 this
        # will be true.
        self.assertEqual(torch.all(decoder_input == transformer.bos_idx), False)


if __name__ == "__main__":
    unittest.main()