import unittest
from typing import List, Dict, Any
import random

import numpy as np
import torch
from torch import nn

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.checkpoint_manager import CheckPointManager
from lr_scheduler import NoamOpt
from response_parser.simple_response_parser import SimpleResponseParser
from transformer import Transformer
from vocabulary_builder.simple_vocabulary_builder import SimpleVocabBuilder


def train(
    vocabulary: SimpleVocabBuilder,
    transformer: nn.Module,
    scheduler: Any,
    criterion: Any,
    batches: Dict[str, List[List[List[dict]]]],
    masks: Dict[str, List[torch.Tensor]],
    n_epochs: int,
    task_type: str,
    start_epoch=0,
    is_training=True,
    verbose_log=False,
):
    """
    Main training loop

    :param vocabulary: Vocabulary class instance.
    :param task_type: The type of task.
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
        for i, (src_batch, src_padding_mask, tgt_batch, tgt_future_mask) in enumerate(
            zip(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY],
                masks[BatchBuilder.PADDING_MASK_KEY],
                batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY],
                masks[BatchBuilder.FUTURE_MASK_KEY])
        ):
            encoder_output = transformer.encoder(src_batch, task_type, src_padding_mask=src_padding_mask)

            # Perform one decoder forward pass to obtain *all* next-token predictions for every index i given its
            # previous *gold standard* tokens [1,..., i] (i.e. teacher forcing) in parallel/at once.
            decoder_output = transformer.decoder(
                tgt_batch,
                task_type,
                encoder_output,
                src_padding_mask=src_padding_mask,
                future_mask=tgt_future_mask,
            )  # type: ignore

            # Align labels with predictions: the last decoder prediction is meaningless because we have no target token
            # for it. In teacher forcing we want to force all tokens, but to force/let know decoder to learn a token,
            # it has to be provided in the decoder input. If provide in decoder input then it will produce one
            # decoder output, but this output is meaningless, as we don't have any target for that token.

            # Decoder output also don't have BOS, as BOS is added in decoder input for the first token.

            # [batch_size, sequence_length, logits]
            decoder_output = decoder_output[:, :-1, :]
            # print(f"after decoder_output {decoder_output.shape}")

            # convert tgt_batch into integer tokens
            tgt_batch = torch.tensor(vocabulary.batch_encoder(tgt_batch))
            # The BOS token in the target is also not something we want to compute a loss for.
            # As it's not available in Decoder output.
            # But Padding and EOS is okay, as we will compute decoder output until max_length.
            # Which include EOS and Padding musk tokens.
            # [batch_size, sequence_length]
            tgt_batch = tgt_batch[:, 1:]

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

            if num_iters % len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY]) == 0 or not is_training:
                print(
                    f"epoch: {e}, num_iters: {num_iters}, batch_loss: {batch_loss}, batch_accuracy: {batch_accuracy}"
                )
                if verbose_log:
                    # Printing predicted tokens
                    print("~~~Printing target batch~~~\n")
                    SimpleResponseParser.print_response_to_console(vocabulary.batch_decode(tgt_batch.tolist()))
                    print("~~~Printing decoder output batch~~~\n")
                    SimpleResponseParser.print_response_to_console(vocabulary.batch_decode(decoder_output.argmax(dim=-1).tolist()))

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

        batch_size = 3
        n_epochs = 35
        max_encoding_length = 20
        max_decoding_length = 10
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        corpus_source = [
            "These are the tokens that will end up in our vocabulary",
            "The sun set behind the mountains, painting the sky in hues of orange and pink",
            "curious cat chased a fluttering butterfly through the lush garden",
            "She sipped her steaming cup of tea as she gazed out the window at the pouring rain",
            "The laughter of children echoed through the park on a warm summer afternoon",
            "With a flick of his wrist, the magician made the playing cards disappear into thin air",
        ]
        corpus_target = [
            "The sun is shining brightly in the clear blue sky",
            "She studied hard for her exams and earned top grades",
            "The cat chased the mouse around the house",
            "He loves to play the guitar and sing songs",
            "They enjoyed a delicious meal at their favorite restaurant",
            "The book was so captivating that she couldn't put it down",
        ]
        combined_list = corpus_source + corpus_target

        # Creating the vocabulary
        vocab = SimpleVocabBuilder(BatchBuilder.get_batch_io_parser_output(combined_list, True, None))
        vocab_size = len(list(vocab.vocab_item_to_index.keys()))
        valid_tokens = list(vocab.vocab_item_to_index.keys())[3:]
        print(f"Vocabulary size: {vocab_size}")

        # Construct src-tgt aligned input batches (note: the original paper uses dynamic batching based on tokens)
        # Creating the batch
        corpus = [
            {BatchBuilder.SOURCE_LANGUAGE_KEY: src, BatchBuilder.TARGET_LANGUAGE_KEY: tgt} for src, tgt in
            zip(corpus_source, corpus_target)
        ]

        batches, masks = BatchBuilder.construct_batches_for_transformer(
            corpus,
            batch_size=batch_size,
            max_encoder_sequence_length=max_encoding_length,
            max_decoder_sequence_length=max_decoding_length,
        )

        print(
            f"valid token {len(valid_tokens)}\n"
            f"corpus {len(corpus)}\n"
            f"batch size: {batch_size} Number of item in batches {len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY])},"
            f" calculated : {len(corpus)/batch_size}"
        )

        # Initialize transformer
        transformer = Transformer(
            hidden_dim=768,
            batch_size=batch_size,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
            max_decoding_length=max_decoding_length,
            vocab_size=vocab_size,
            dropout_p=0.1,
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
            vocabulary=vocab,
            transformer=transformer,
            scheduler=scheduler,
            criterion=criterion,
            batches=batches,
            masks=masks,
            n_epochs=n_epochs,
            task_type=task_type,
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
        n_epochs = 1
        max_encoding_length = 20
        max_decoding_length = 10
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        corpus_source = [
            "These are the tokens that will end up in our vocabulary",
            "The sun set behind the mountains, painting the sky in hues of orange and pink",
            "curious cat chased a fluttering butterfly through the lush garden",
            "She sipped her steaming cup of tea as she gazed out the window at the pouring rain",
            "The laughter of children echoed through the park on a warm summer afternoon",
            "With a flick of his wrist, the magician made the playing cards disappear into thin air",
        ]
        corpus_target = [
            "The sun is shining brightly in the clear blue sky",
            "She studied hard for her exams and earned top grades",
            "The cat chased the mouse around the house",
            "He loves to play the guitar and sing songs",
            "They enjoyed a delicious meal at their favorite restaurant",
            "The book was so captivating that she couldn't put it down",
        ]
        combined_list = corpus_source + corpus_target

        # Creating the vocabulary
        vocab = SimpleVocabBuilder(BatchBuilder.get_batch_io_parser_output(combined_list, True, None))
        vocab_size = len(list(vocab.vocab_item_to_index.keys()))
        valid_tokens = list(vocab.vocab_item_to_index.keys())[3:]
        print(f"Vocabulary size: {vocab_size}")

        # Creating the batch
        corpus = [
            {BatchBuilder.SOURCE_LANGUAGE_KEY: src, BatchBuilder.TARGET_LANGUAGE_KEY: tgt} for src, tgt in
            zip(corpus_source, corpus_target)
        ]

        batches, masks = BatchBuilder.construct_batches_for_transformer(
            corpus,
            batch_size=batch_size,
            max_encoder_sequence_length=max_encoding_length,
            max_decoder_sequence_length=max_decoding_length,
        )

        print(
            f"valid token {len(valid_tokens)}\n"
            f"corpus {len(corpus)}\n"
            f"batch size: {batch_size} Number of item in batches {len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY])},"
            f" calculated : {len(corpus) / batch_size}"
        )

        # Initialize transformer
        transformer = Transformer(
            hidden_dim=768,
            batch_size=batch_size,
            ff_dim=2048,
            num_heads=8,
            num_layers=2,
            max_decoding_length=max_decoding_length,
            vocab_size=vocab_size,
            dropout_p=0.1,
        ).to(device)

        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        optimizer = torch.optim.Adam(
            transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )

        # Load the model...
        checkpoint_map = CheckPointManager.load_checkpoint_map(
            TestTransformerTraining.PATH
        )
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
            vocabulary=vocab,
            transformer=transformer,
            scheduler=scheduler,
            criterion=criterion,
            batches=batches,
            masks=masks,
            n_epochs=n_epochs,
            task_type=task_type,
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
