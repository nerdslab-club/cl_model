# inference func
# word hishabe

import unittest
from typing import List, Dict, Any
import random
from random import choices

import numpy as np
import torch
from torch import nn

from cl_pretrainer.checkpoint_manager import CheckPointManager
from lr_scheduler import NoamOpt
from transformer import Transformer
from vocabulary import Vocabulary
from transformer_utils import construct_batches


def inference(
        transformer: nn.Module,
        criterion: Any,
        batches: Dict[str, List[torch.Tensor]],
        masks: Dict[str, List[torch.Tensor]],
        vocab: Vocabulary,
        verbose_log=False,
):
    """
    Main inference loop

    :param verbose_log: Log in detailed level with tgt_output and decoder_output
    :param transformer: the transformer model
    :param criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :return: the accuracy and loss on the latest batch
    """
    transformer.train(False)

    num_iters = 0

    for i, (src_batch, src_mask, tgt_batch, tgt_mask) in enumerate(
            zip(batches["src"], masks["src"], batches["tgt"], masks["tgt"])
    ):
        encoder_output = transformer.encoder(src_batch, src_padding_mask=src_mask)  # type: ignore

        # Perform one decoder forward pass to obtain *all* next-token predictions for every index i given its
        # previous *gold standard* tokens [1,..., i] (i.e. teacher forcing) in parallel/at once.
        decoder_output = transformer.decoder(
            tgt_batch,
            encoder_output,
            src_padding_mask=src_mask,
            future_mask=tgt_mask,
        )  # type: ignore

        # Align labels with predictions: the last decoder prediction is meaningless because we have no target token
        # for it. In teacher forcing we want to force all tokens, but to force/let know decoder to learn a token,
        # it has to be provided in the decoder input. If provide in decoder input then it will produce one
        # decoder output, but this output is meaningless, as we don't have any target for that token.

        # Decoder output also don't have BOS, as BOS is added in decoder input for the first token.

        # [batch_size, sequence_length, logits]
        decoder_output = decoder_output[:, :-1, :]
        # print(f"after decoder_output {decoder_output.shape}")

        # The BOS token in the target is also not something we want to compute a loss for.
        # As it's not available in Decoder output.
        # But Padding and EOS is okay, as we will compute decoder output until max_length.
        # Which include EOS and Padding musk tokens.
        # [batch_size, sequence_length]
        tgt_batch = tgt_batch[:, 1:]
        # print(f"after tgt_batch {tgt_batch.shape}")

        # Set pad tokens in the target to -100 so they don't incur a loss
        # tgt_batch[tgt_batch == transformer.padding_idx] = -100

        # Compute the average cross-entropy loss over all next-token predictions at each index i given [1, ..., i]
        # for the entire batch. Note that the original paper uses label smoothing (I was too lazy).
        batch_loss = criterion(
            decoder_output.contiguous().permute(0, 2, 1),
            tgt_batch.contiguous().long(),
        )

        # Rough estimate of per-token accuracy in the current training batch
        batch_accuracy = (
                             torch.sum(decoder_output.argmax(dim=-1) == tgt_batch)
                         ) / torch.numel(tgt_batch)

        if verbose_log:
            decode_output = decoder_output.argmax(dim=-1)
            print(
                f"tgt batch: {tgt_batch.squeeze().tolist()}\n"
                f"decoder output: {decode_output.squeeze().tolist()}\n"
            )

        num_iters += 1
    return batch_loss, batch_accuracy


class TestTransformerInference(unittest.TestCase):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    PATH = "./saved_models/model.pth"

    def test_model_load(self):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        corpus_source = [
            "These are the tokens that will end up in our vocabulary",
            "The sun set behind the mountains, painting the sky in hues of orange and pink",
            "curious cat chased a fluttering butterfly through the lush garden",
            "She sipped her steaming cup of tea as she gazed out the window at the pouring rain",
            "The laughter of children echoed through the park on a warm summer afternoon.",
            "With a flick of his wrist, the magician made the playing cards disappear into thin air.",
        ]
        corpus_target = [
            "The sun is shining brightly in the clear blue sky.",
            "She studied hard for her exams and earned top grades.",
            "The cat chased the mouse around the house.",
            "He loves to play the guitar and sing songs.",
            "They enjoyed a delicious meal at their favorite restaurant.",
            "The book was so captivating that she couldn't put it down."
        ]
        combined_list = corpus_source + corpus_target

        vocab = Vocabulary(combined_list)
        vocab_size = len(
            list(vocab.token2index.keys())
        )  # 110 tokens including bos, eos and pad
        valid_tokens = list(vocab.token2index.keys())[3:]
        print(f"Vocabulary size: {vocab_size}")

        # Construct src-tgt aligned input batches (note: the original paper uses dynamic batching based on tokens)
        # corpus = [{"src": src, "tgt": tgt} for src, tgt in zip(corpus_source, corpus_target)]
        corpus = [{"src": "These are the tokens that will end up in our vocabulary", "tgt": "The sun is shining brightly in the clear blue sky."},
                  {"src": "These are the tokens that will end up in our vocabulary", "tgt": "The sun is shining brightly in the clear blue sky."}]
        batches, masks = construct_batches(
            corpus,
            vocab,
            batch_size=1,
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )

        print(
            f"valid token {len(valid_tokens)}\n"
            f"corpus {len(corpus)}\n"
        )

        # Initialize transformer
        transformer = Transformer(
            hidden_dim=512,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
            max_decoding_length=25,
            vocab_size=vocab_size,
            padding_idx=vocab.token2index[vocab.PAD],
            bos_idx=vocab.token2index[vocab.BOS],
            dropout_p=0.1,
            tie_output_to_embedding=True,
        ).to(device)

        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )

        # Load the model...
        checkpoint_map = CheckPointManager.load_checkpoint_map(TestTransformerInference.PATH)
        transformer.load_saved_model_from_state_dict(
            CheckPointManager.get_checkpoint_item(
                checkpoint_map,
                CheckPointManager.TRANSFORMER_STATE,
            ),
        )
        optimizer.load_state_dict(
            CheckPointManager.get_checkpoint_item(
                checkpoint_map,
                CheckPointManager.OPTIM_STATE,
            ),
        )

        print("Model loaded correctly...")

        criterion = nn.CrossEntropyLoss()

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy = inference(
            transformer,
            criterion,
            batches,
            masks,
            vocab,
            verbose_log=True,
        )

        print(f"batch loss {latest_batch_loss.item()}")
        print(f"batch accuracy {latest_batch_accuracy}")
        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)


if __name__ == "__main__":
    unittest.main()
