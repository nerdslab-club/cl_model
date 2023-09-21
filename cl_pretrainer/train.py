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


def train(
    transformer: nn.Module,
    scheduler: Any,
    criterion: Any,
    batches: Dict[str, List[torch.Tensor]],
    masks: Dict[str, List[torch.Tensor]],
    n_epochs: int,
    start_epoch=0,
    is_training=True,
    verbose_log=False,
):
    """
    Main training loop

    :param start_epoch: From which epoch training should resume.
    :param verbose_log: Log in detailed level with tgt_output and decoder_output
    :param transformer: the transformer model
    :param scheduler: the learning rate scheduler
    :param criterion: the optimization criterion (loss function)
    :param batches: aligned src and tgt batches that contain tokens ids
    :param masks: source key padding mask and target future mask for each batch
    :param n_epochs: the number of epochs to train the model for
    :param is_training: is the model used for training or inference
    :return: the accuracy and loss on the latest batch
    """
    transformer.train(is_training)
    if not is_training:
        n_epochs = 1

    num_iters = 0

    for e in range(start_epoch, start_epoch + n_epochs):
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

            if num_iters % len(batches["src"]) == 0 or not is_training:
                print(
                    f"epoch: {e}, num_iters: {num_iters}, batch_loss: {batch_loss}, batch_accuracy: {batch_accuracy}"
                )
                if verbose_log:
                    print(
                        f"tgt batch: {tgt_batch}\n"
                        f"decoder output: {decoder_output.argmax(dim=-1)}"
                    )

            # Update parameters
            if is_training:
                batch_loss.backward()
                scheduler.step()
                scheduler.optimizer.zero_grad()
            num_iters += 1
    return batch_loss, batch_accuracy


class TestTransformerTraining(unittest.TestCase):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    PATH = "./saved_models/model.pth"

    def test_train_and_save(self):
        """
        Test training by trying to (over)fit a simple copy dataset - bringing the loss to ~zero. (GPU required)
        """
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # if device.type == "cpu":
        #     print("This unit test was not run because it requires a GPU")
        #     return

        # Hyperparameters
        # synthetic_corpus_size = 6
        # n_tokens_in_batch = 10
        # corpus += [
        #     " ".join(choices(valid_tokens, k=n_tokens_in_batch))
        #     for _ in range(synthetic_corpus_size)
        # ]

        batch_size = 3
        n_epochs = 50

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        # Note: the original paper uses byte pair encodings, we simply take each word to be a token.
        corpus = [
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
        combined_list = corpus + corpus_target

        vocab = Vocabulary(combined_list)
        vocab_size = len(
            list(vocab.token2index.keys())
        )  # 110 tokens including bos, eos and pad
        valid_tokens = list(vocab.token2index.keys())[3:]
        print(f"Vocabulary size: {vocab_size}")

        # Construct src-tgt aligned input batches (note: the original paper uses dynamic batching based on tokens)
        corpus = [{"src": src, "tgt": tgt} for src, tgt in zip(corpus, corpus_target)]
        batches, masks = construct_batches(
            corpus,
            vocab,
            batch_size=batch_size,
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )

        print(
            f"valid token {len(valid_tokens)}\n"
            f"corpus {len(corpus)}\n"
            f"batch size: {batch_size} Number of item in batches {len(batches['src'])},"
            f" calculated : {len(corpus)/batch_size}"
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
        scheduler = NoamOpt(
            transformer.hidden_dim,
            factor=1,
            warmup=400,
            optimizer=optimizer,
        )
        criterion = nn.CrossEntropyLoss()

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy = train(
            transformer,
            scheduler,
            criterion,
            batches,
            masks,
            n_epochs=n_epochs,
            is_training=True,
            verbose_log=False,
        )

        print(f"batch loss {latest_batch_loss.item()}")
        print(f"batch accuracy {latest_batch_accuracy}")

        CheckPointManager.save_checkpoint_map(
            TestTransformerTraining.PATH,
            n_epochs,
            transformer,
            optimizer,
        )

        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)

    def test_model_load(self):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        batch_size = 3
        n_epochs = 2
        corpus = [
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
        combined_list = corpus + corpus_target

        vocab = Vocabulary(combined_list)
        vocab_size = len(
            list(vocab.token2index.keys())
        )  # 110 tokens including bos, eos and pad
        valid_tokens = list(vocab.token2index.keys())[3:]
        print(f"Vocabulary size: {vocab_size}")

        # Construct src-tgt aligned input batches (note: the original paper uses dynamic batching based on tokens)
        corpus = [{"src": src, "tgt": tgt} for src, tgt in zip(corpus, corpus_target)]
        batches, masks = construct_batches(
            corpus,
            vocab,
            batch_size=batch_size,
            src_lang_key="src",
            tgt_lang_key="tgt",
            device=device,
        )

        print(
            f"valid token {len(valid_tokens)}\n"
            f"corpus {len(corpus)}\n"
            f"batch size: {batch_size} Number of item in batches {len(batches['src'])},"
            f" calculated : {len(corpus) / batch_size}"
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
        checkpoint_map = CheckPointManager.load_checkpoint_map(TestTransformerTraining.PATH)
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
        start_epoch = CheckPointManager.get_checkpoint_item(
            checkpoint_map,
            CheckPointManager.EPOCH,
        )
        print("Model loaded correctly...")

        scheduler = NoamOpt(
            transformer.hidden_dim,
            factor=1,
            warmup=400,
            optimizer=optimizer,
        )
        criterion = nn.CrossEntropyLoss()

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy = train(
            transformer,
            scheduler,
            criterion,
            batches,
            masks,
            n_epochs=n_epochs,
            start_epoch=start_epoch,
            is_training=False,
            verbose_log=True,
        )

        print(f"batch loss {latest_batch_loss.item()}")
        print(f"batch accuracy {latest_batch_accuracy}")
        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)


if __name__ == "__main__":
    unittest.main()
