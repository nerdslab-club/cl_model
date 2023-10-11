import unittest
from typing import Dict, List, Tuple, Optional

import torch

from cl_data.io_parser.io_parser import IoParser
from cl_data.pretrain_data.next_token_sample_generator import NextTokenSamplesGenerator
from cl_data.src.constants import Constants, SpecialTokens
from cl_data.src.utility import Utility


class BatchBuilder:
    SOURCE_LANGUAGE_KEY = "inputStr"
    TARGET_LANGUAGE_KEY = "outputStr"
    ENCODER_IO_PARSER_OUTPUT_KEY = "encoderIoParserOutput"
    DECODER_IO_PARSER_OUTPUT_KEY = "decoderIoParserOutput"
    FUTURE_MASK_KEY = "futureMaskKey"
    ENCODER_PADDING_MASK_KEY = "encoderPaddingMaskKey"
    DECODER_PADDING_MASK_KEY = "decoderPaddingMaskKey"
    PADDING_MASK_KEY = "paddingMaskKey"

    @staticmethod
    def get_batch_io_parser_output(
            sentences: list[str],
            add_bos_and_eos: bool,
            max_sequence_length: int | None,
    ) -> list[list[dict]]:
        """Given list of sentences it creates list of io parser output with special tokens and paddings

        :param sentences: List of sentences.
        :param add_bos_and_eos: Flag for weather to add BOS and EOS to the Token list.
        :param max_sequence_length: Max length until which padding will be added. If None then no padding.
        :return: io parser items in a list of list.
        """
        batch_io_parser_output = []
        for sentence in sentences:
            batch_io_parser_output.append(
                BatchBuilder.get_sentence_io_parser_output(
                    sentence,
                    add_bos_and_eos,
                    max_sequence_length,
                ),
            )
        return batch_io_parser_output

    @staticmethod
    def get_sentence_io_parser_output(sentence: str, add_bos_and_eos: bool, max_sequence_length: int | None) -> list[
        dict]:
        """Given a sentence it called io parser on the sentence with special tokens and paddings

        :param sentence: A normal sentence string.
        :param add_bos_and_eos: Flag for weather to add BOS and EOS to the Token list.
        :param max_sequence_length: Max length until which padding will be added. If None then no padding.
        :return: list of dict where each dict is io parser item.
        """
        io_parser_tuples = IoParser().create_value_list_from_input(sentence)
        return Utility.create_io_map_from_io_tuple(
            input_list=io_parser_tuples,
            add_bos_and_eos=add_bos_and_eos,
            max_length=max_sequence_length
        )

    @staticmethod
    def construct_batches_for_cl_pre_trainer(
            corpus: List[str],
            batch_size: int,
            max_decoder_sequence_length: int,
            device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, list[list[list[dict]]]], Dict[str, List[torch.Tensor]]]:
        """Constructs batches given sample corpus for cl pre trainer model

        :param corpus: This is a list of sentences on which the model is to be trained. ie.
        [
           "In the kingdom of Numerosia, where equations adorned the walls and numbers held the key to understanding",
           "lived a passionate mathematician named Mia.",
           "Mia's life was dedicated to unraveling mathematical puzzles",
           "and her reputation extended far beyond the kingdom's borders.",
           "One fateful day, a mysterious messenger delivered an ornate scroll to her doorstep.",
        ]
        :param batch_size: The number of sequences in a batch
        :param max_decoder_sequence_length: Truncate/Allowed max decoder sequence length. If None then no padding.
        :param device: whether to move tensors to gpu
        :return: A tuple containing two dictionaries.
        The first represents the batches, This is batch io parser output.
        Second one represents the attention masks. This is bool Tensor.
        """
        next_token_task_corpus = NextTokenSamplesGenerator.create_next_token_batches(
            corpus,
        )
        batches: Dict[str, List] = {
            BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY: [],
            BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY: [],
        }
        masks: Dict[str, List] = {
            BatchBuilder.FUTURE_MASK_KEY: [],
            BatchBuilder.PADDING_MASK_KEY: [],
        }
        for i in range(0, len(next_token_task_corpus), batch_size):
            input_sentences = [
                pair.get(BatchBuilder.SOURCE_LANGUAGE_KEY, "")
                for pair in next_token_task_corpus[i: i + batch_size]
            ]
            src_batch = BatchBuilder.get_batch_io_parser_output(
                sentences=input_sentences,
                add_bos_and_eos=True,
                max_sequence_length=max_decoder_sequence_length,
            )

            output_sentences = [
                pair.get(BatchBuilder.TARGET_LANGUAGE_KEY, "")
                for pair in next_token_task_corpus[i: i + batch_size]
            ]
            tgt_batch = BatchBuilder.get_batch_io_parser_output(
                sentences=output_sentences,
                add_bos_and_eos=True,
                max_sequence_length=max_decoder_sequence_length,
            )
            future_mask = BatchBuilder.construct_future_mask(max_decoder_sequence_length)
            padding_mask = BatchBuilder.construct_padding_mask(src_batch)

            if device is not None:
                src_batch = src_batch.to(device)  # type: ignore
                tgt_batch = tgt_batch.to(device)  # type: ignore
                future_mask = future_mask.to(device)
                padding_mask = padding_mask.to(device)

            batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY].append(src_batch)
            batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY].append(tgt_batch)
            masks[BatchBuilder.PADDING_MASK_KEY].append(padding_mask)
            masks[BatchBuilder.FUTURE_MASK_KEY].append(future_mask)
        return batches, masks

    @staticmethod
    def construct_batches_for_transformer(
            corpus: List[Dict[str, str]],
            batch_size: int,
            max_encoder_sequence_length: int,
            max_decoder_sequence_length: int,
            device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, list[list[list[dict]]]], Dict[str, List[torch.Tensor]]]:
        """ Constructs batches given sample corpus.

        :param corpus: The input corpus is a list of aligned source and target sequences, packed in a dictionary.
        ie.
        [
            {
                    "inputStr": f"##factorial({x})",
                    "outputStr": f"The factorial of {x}"
            },
            {
                    "inputStr": f"##factorial({x})",
                    "outputStr": f"The factorial of {x}"
            }
        ]
        :param batch_size: The number of sequences in a batch
        :param max_encoder_sequence_length: Truncate/Allowed max encoder sequence length. If None then no padding.
        :param max_decoder_sequence_length: Truncate/Allowed max decoder sequence length. If None then no padding.
        :param device: whether to move tensors to gpu
        :return: A tuple containing two dictionaries.
        The first represents the batches, This is batch io parser output.
        Second one represents the attention masks. This is bool Tensor.
        We don't need decoder padding mask as when we will get EOS token then we will stop predicting.
        """
        batches: Dict[str, List] = {
            BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY: [],
            BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY: [],
        }
        masks: Dict[str, List] = {
            BatchBuilder.FUTURE_MASK_KEY: [],
            BatchBuilder.ENCODER_PADDING_MASK_KEY: [],
            BatchBuilder.DECODER_PADDING_MASK_KEY: [],
        }
        for i in range(0, len(corpus), batch_size):
            input_sentences = [pair.get(BatchBuilder.SOURCE_LANGUAGE_KEY, "") for pair in corpus[i: i + batch_size]]
            src_batch = BatchBuilder.get_batch_io_parser_output(
                sentences=input_sentences,
                add_bos_and_eos=True,
                max_sequence_length=max_encoder_sequence_length,
            )

            output_sentences = [pair.get(BatchBuilder.TARGET_LANGUAGE_KEY, "") for pair in corpus[i: i + batch_size]]
            tgt_batch = BatchBuilder.get_batch_io_parser_output(
                sentences=output_sentences,
                add_bos_and_eos=True,
                max_sequence_length=max_decoder_sequence_length,
            )

            future_mask = BatchBuilder.construct_future_mask(max_decoder_sequence_length)
            encoder_padding_mask = BatchBuilder.construct_padding_mask(src_batch)
            decoder_padding_mask = BatchBuilder.construct_padding_mask(tgt_batch)

            if device is not None:
                src_batch = src_batch.to(device)  # type: ignore
                tgt_batch = tgt_batch.to(device)  # type: ignore
                future_mask = future_mask.to(device)
                encoder_padding_mask = encoder_padding_mask.to(device)
                decoder_padding_mask = decoder_padding_mask.to(device)

            batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY].append(src_batch)
            batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY].append(tgt_batch)
            masks[BatchBuilder.ENCODER_PADDING_MASK_KEY].append(encoder_padding_mask)
            masks[BatchBuilder.DECODER_PADDING_MASK_KEY].append(decoder_padding_mask)
            masks[BatchBuilder.FUTURE_MASK_KEY].append(future_mask)
        return batches, masks

    @staticmethod
    def construct_future_mask(seq_len: int):
        """
        Construct a binary mask that contains 1's for all valid connections and 0's for all outgoing future connections.
        This mask will be applied to the attention logits in decoder self-attention such that all logits with a 0 mask
        are set to -inf.

        :param seq_len: length of the input sequence
        :return: (seq_len,seq_len) mask
        """
        subsequent_mask = torch.triu(torch.full((seq_len, seq_len), 1), diagonal=1)
        return subsequent_mask == 0

    @staticmethod
    def construct_padding_mask(batch_io_parser_output: list[list[dict]]):
        """ Create padding mask by comparing batch_io_parser_output items to <PAD> token.

        :param batch_io_parser_output: batch of io_parser_output
        :return: Bool tensor padding mask.
        """
        # Convert the list to a PyTorch tensor of integers (comparing with 1)
        padding_mask = torch.tensor(
            [[item[Constants.TOKEN] == SpecialTokens.PADDING.value
              for item in io_parser_output]
             for io_parser_output in batch_io_parser_output], dtype=torch.bool)

        return padding_mask


class TestUtils(unittest.TestCase):
    def test_get_sentence_io_parser_output(self):
        sentence = "one two three four five six seven eight"
        io_parser_output = BatchBuilder.get_sentence_io_parser_output(
            sentence,
            add_bos_and_eos=True,
            max_sequence_length=12,
        )
        expected_result = [
            {'token': '<BOS>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'}, 'position': 0},
            {'token': 'one', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 1},
            {'token': 'two', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 2},
            {'token': 'three', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 3},
            {'token': 'four', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 4},
            {'token': 'five', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 5},
            {'token': 'six', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 6},
            {'token': 'seven', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 7},
            {'token': 'eight', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 8},
            {'token': '<EOS>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'}, 'position': 9},
            {'token': '<PAD>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'}, 'position': 10},
            {'token': '<PAD>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'}, 'position': 11},
        ]
        self.assertEqual(expected_result, io_parser_output)

    def test_construct_future_mask(self):
        mask = BatchBuilder.construct_future_mask(3)
        torch.testing.assert_close(
            mask,
            torch.BoolTensor(
                [[True, False, False], [True, True, False], [True, True, True]]
            ),
        )

    def test_construct_future_mask_first_decoding_step(self):
        mask = BatchBuilder.construct_future_mask(1)
        torch.testing.assert_close(
            mask,
            torch.BoolTensor([[True]]),
        )

    def test_construct_batches_for_transformer(self):
        max_encoding_length = 16
        max_decoding_length = 16
        batch_size = 2
        corpus = [
            {
                BatchBuilder.SOURCE_LANGUAGE_KEY: "This is an english sentence.",
                BatchBuilder.TARGET_LANGUAGE_KEY: "Dit is een Nederlandse zin.",
            },
            {
                BatchBuilder.SOURCE_LANGUAGE_KEY: "The weather is nice today.",
                BatchBuilder.TARGET_LANGUAGE_KEY: "Het is lekker weer vandaag.",
            },
            {
                BatchBuilder.SOURCE_LANGUAGE_KEY: "Yesterday I drove to a city called Amsterdam in my brand new car.",
                BatchBuilder.TARGET_LANGUAGE_KEY: "Ik reed gisteren in mijn gloednieuwe auto naar Amsterdam.",
            },
            {
                BatchBuilder.SOURCE_LANGUAGE_KEY: "You can pick up your laptop at noon tomorrow.",
                BatchBuilder.TARGET_LANGUAGE_KEY: "Je kunt je laptop morgenmiddag komen ophalen.",
            },
        ]
        batches, masks = BatchBuilder.construct_batches_for_transformer(
            corpus=corpus,
            batch_size=batch_size,
            max_encoder_sequence_length=max_encoding_length,
            max_decoder_sequence_length=max_decoding_length,
        )
        # batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY].Shape[batch_item, batch_size, max_encoding_length]
        self.assertEqual((len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY]),
                          len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY][0]),
                          len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY][0][0])),
                         (len(corpus) // batch_size, batch_size, max_encoding_length))


if __name__ == "__main__":
    item = [
        {
            "token": SpecialTokens.BEGINNING.value,
            "category": {
                "type": "special",
                "subType": "word",
                "subSubType": "none"
            },
            "position": 0
        },
        {
            "token": 126,
            "category": {
                "type": "integer",
                "subType": "default",
                "subSubType": "none"
            },
            "position": 1
        },
        {
            "token": "plus",
            "category": {
                "type": "word",
                "subType": "default",
                "subSubType": "none"
            },
            "position": 2
        },
        {
            "token": 840,
            "category": {
                "type": "integer",
                "subType": "default",
                "subSubType": "none"
            },
            "position": 3
        },
        {
            "token": "equals?",
            "category": {
                "type": "word",
                "subType": "default",
                "subSubType": "none"
            },
            "position": 4
        },
        {
            "token": SpecialTokens.ENDING.value,
            "category": {
                "type": "special",
                "subType": "word",
                "subSubType": "none"
            },
            "position": 5
        },
        {
            "token": SpecialTokens.PADDING.value,
            "category": {
                "type": "special",
                "subType": "word",
                "subSubType": "none"
            },
            "position": 6
        },
    ]
    print(f"Padding mask: {BatchBuilder.construct_padding_mask([item, item])}")
