import unittest
from typing import List, Dict, Any
import random

import numpy as np
import torch
from torch import nn

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.checkpoint_manager import CheckPointManager
from response_parser.simple_response_parser import SimpleResponseParser
from transformer import Transformer
from vocabulary_builder.simple_vocabulary_builder import SimpleVocabBuilder


def inference(
        vocabulary: SimpleVocabBuilder,
        transformer: nn.Module,
        criterion: Any,
        batches: Dict[str, List[List[List[dict]]]],
        masks: Dict[str, List[torch.Tensor]],
        task_type: str,
        verbose_log=False,
):
    """
    Main inference loop

    :param task_type: The type of task ie: nl_to_nl_translation
    :param vocabulary: Vocabulary class instance.
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
            zip(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY],
                masks[BatchBuilder.PADDING_MASK_KEY],
                batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY],
                masks[BatchBuilder.FUTURE_MASK_KEY])
    ):
        encoder_output = transformer.encoder(src_batch, task_type, src_padding_mask=src_mask)
        decoder_output = transformer.decoder(
            tgt_batch,
            task_type,
            encoder_output,
            src_padding_mask=src_mask,
            future_mask=tgt_mask,
        )

        decoder_output = decoder_output[:, :-1, :]

        # convert tgt_batch into integer tokens
        tgt_batch = torch.tensor(vocabulary.batch_encoder(tgt_batch))
        tgt_batch = tgt_batch[:, 1:]

        # calculate batch loss
        batch_loss = criterion(
            decoder_output.contiguous().permute(0, 2, 1),
            tgt_batch.contiguous().long(),
        )

        # Rough estimate of per-token accuracy in the current training batch
        batch_accuracy = (torch.sum(decoder_output.argmax(dim=-1) == tgt_batch)) / torch.numel(tgt_batch)

        if verbose_log:
            decode_output = decoder_output.argmax(dim=-1)
            # Printing predicted tokens
            print("~~~Printing target batch~~~\n")
            SimpleResponseParser.print_response_to_console(vocabulary.batch_decode(tgt_batch.tolist()))
            print("~~~Printing decoder output batch~~~\n")
            SimpleResponseParser.print_response_to_console(vocabulary.batch_decode(decode_output.tolist()))

        num_iters += 1
    return batch_loss, batch_accuracy


class TestTransformerInference(unittest.TestCase):
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    PATH = "./saved_models/model.pth"

    def test_model_load_and_inference(self):
        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        batch_size = 1
        # Minimum encoding length is 16
        max_encoding_length = 20
        max_decoding_length = 8
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value

        # Construct vocabulary and create synthetic data by uniform randomly sampling tokens from it
        corpus_source = [
            "The sun is shining brightly in the clear blue sky",
            "She studied hard for her exams and earned top grades",
            "The cat chased the mouse around the house",
            "He loves to play the guitar and sing songs",
            "They enjoyed a delicious meal at their favorite restaurant",
            "The book was so captivating that she couldn't put it down",
        ]
        corpus_target = [
            "He reads books daily",
            "I like chocolate ice cream",
            "Dogs bark loudly at night",
            "She dances gracefully on stage",
            "Flowers bloom in springtime",
            "Raindrops fall gently from clouds"
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
            f"corpus {len(corpus)}"
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
            vocabulary=vocab,
            transformer=transformer,
            criterion=criterion,
            batches=batches,
            masks=masks,
            task_type=task_type,
            verbose_log=True,
        )

        print(f"batch loss {latest_batch_loss.item()}")
        print(f"batch accuracy {latest_batch_accuracy}")
        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)


if __name__ == "__main__":
    unittest.main()
