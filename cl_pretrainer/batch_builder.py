import unittest
from typing import Dict, List, Tuple, Optional

import torch

from cl_data.io_parser.io_parser import IoParser
from cl_data.io_parser.io_parser_utility import split_string_custom
from cl_data.src.constants import Constants, SpecialTokens
from cl_data.src.utility import Utility


class BatchBuilder:
    SOURCE_LANGUAGE_KEY = "inputStr"
    TARGET_LANGUAGE_KEY = "outputStr"
    ENCODER_IO_PARSER_OUTPUT_KEY = "encoderIoParserOutput"
    DECODER_IO_PARSER_OUTPUT_KEY = "decoderIoParserOutput"
    FUTURE_MASK_KEY = "futureMaskKey"
    PADDING_MASK_KEY = "paddingMaskKey"
    TASK_TYPE_KEY = "taskTypeKey"
    INITIAL_TOKEN_COUNT_KEY = "initialTokenCountKey"

    #  corpus_source = [
    #             "Laughter is contagious, spreading joy and happiness",
    #             "Waves crash on the shore, a soothing lullaby",
    #             "Stars twinkle in the vast, dark night sky",
    #             "A smile can brighten even the gloomiest day",
    #             "Time flies by when you're having fun",
    #             "Nature's beauty heals the soul and inspires awe",
    #         ]
    #         corpus_target = [
    #             "Birds sing at dawn, greeting morning.",
    #             "Coffee warms, awakening the sleepy soul.",
    #             "Moonlight shimmers on calm, still waters.",
    #             "Raindrops dance on rooftops, serenading sleep.",
    #             "Snow blankets earth in silent purity.",
    #             "A hug speaks volumes without words.",
    #         ]

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
            max_length=max_sequence_length,
            is_eos_finishing_token=False,
        )

    # TODO deprecated method delete afterward.
    # @staticmethod
    # def construct_batches_for_cl_pre_trainer(
    #         corpus: List[str],
    #         batch_size: int,
    #         max_decoder_sequence_length: int,
    #         device: Optional[torch.device] = None,
    #         is_generative_training=False,
    # ) -> Tuple[Dict[str, list[list[list[dict]]]], Dict[str, List[torch.Tensor]]]:
    #     """Constructs batches given sample corpus for cl pre trainer model
    #
    #     :param corpus: This is a list of sentences on which the model is to be trained. ie.
    #     [
    #        "In the kingdom of Numerosia, where equations adorned the walls and numbers held the key to understanding",
    #        "lived a passionate mathematician named Mia.",
    #        "Mia's life was dedicated to unraveling mathematical puzzles",
    #        "and her reputation extended far beyond the kingdom's borders.",
    #        "One fateful day, a mysterious messenger delivered an ornate scroll to her doorstep.",
    #     ]
    #     :param batch_size: The number of sequences in a batch
    #     :param max_decoder_sequence_length: Truncate/Allowed max decoder sequence length. If None then no padding.
    #     :param device: whether to move tensors to gpu
    #     :param is_generative_training: Flag for generative training like: 0123 -> 01234, 01234 -> 012345 ...
    #     :return: A tuple containing two dictionaries.
    #     The first represents the batches, This is batch io parser output.
    #     Second one represents the attention masks. This is bool Tensor.
    #     """
    #     if is_generative_training:
    #         next_token_task_corpus = BatchBuilder.create_generative_training_samples(corpus)
    #         print(f"Generative examples: {next_token_task_corpus}\n")
    #     else:
    #         next_token_task_corpus = [
    #             {BatchBuilder.SOURCE_LANGUAGE_KEY: src, BatchBuilder.TARGET_LANGUAGE_KEY: tgt} for src, tgt in
    #             zip(corpus, corpus)
    #         ]
    #     batches: Dict[str, List] = {
    #         BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY: [],
    #         BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY: [],
    #     }
    #     masks: Dict[str, List] = {
    #         BatchBuilder.FUTURE_MASK_KEY: [],
    #         BatchBuilder.PADDING_MASK_KEY: [],
    #     }
    #     for i in range(0, len(next_token_task_corpus), batch_size):
    #         input_sentences = [
    #             pair.get(BatchBuilder.SOURCE_LANGUAGE_KEY, "")
    #             for pair in next_token_task_corpus[i: i + batch_size]
    #         ]
    #         src_batch = BatchBuilder.get_batch_io_parser_output(
    #             sentences=input_sentences,
    #             add_bos_and_eos=True,
    #             max_sequence_length=max_decoder_sequence_length,
    #         )
    #
    #         output_sentences = [
    #             pair.get(BatchBuilder.TARGET_LANGUAGE_KEY, "")
    #             for pair in next_token_task_corpus[i: i + batch_size]
    #         ]
    #         tgt_batch = BatchBuilder.get_batch_io_parser_output(
    #             sentences=output_sentences,
    #             add_bos_and_eos=True,
    #             max_sequence_length=max_decoder_sequence_length,
    #         )
    #         future_mask = BatchBuilder.construct_future_mask(max_decoder_sequence_length)
    #         padding_mask = BatchBuilder.construct_padding_mask(src_batch)
    #
    #         if device is not None:
    #             src_batch = src_batch.to(device)  # type: ignore
    #             tgt_batch = tgt_batch.to(device)  # type: ignore
    #             future_mask = future_mask.to(device)
    #             padding_mask = padding_mask.to(device)
    #
    #         batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY].append(src_batch)
    #         batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY].append(tgt_batch)
    #         masks[BatchBuilder.PADDING_MASK_KEY].append(padding_mask)
    #         masks[BatchBuilder.FUTURE_MASK_KEY].append(future_mask)
    #     return batches, masks

    @staticmethod
    def construct_batches_for_cl_pre_trainer_with_data_loader(
            data_loader_output: List[dict],
            batch_size: int,
            max_decoder_sequence_length: int,
            add_bos_and_eos: bool = True,
            device: Optional[torch.device] = None,
            is_generative_training=False,
    ) -> Tuple[Dict[str, list[list[list[dict]]]], Dict[str, List[torch.Tensor]]]:
        """Constructs batches given sample corpus for cl pre trainer model

        :param add_bos_and_eos: Weather to add bos and eos or not
        :param data_loader_output: This is a list of data loader output on which the model is to be trained. ie.
        [
            {
               "inputStr": "##multiplication(107.23600916441234,784.625535184504)",
               "outputStr": "107.23600916441234 times 784.625535184504 equals?",
               "taskType": "func_to_nl_translation",
               "sentence": "##multiplication(107.23600916441234,784.625535184504) = 107.23600916441234 times 784.625535184504 equals?",
               "ioParserOutput": [
                   {
                       "token": "<BOS>",
                       "category": {
                           "type": "special",
                           "subType": "word",
                           "subSubType": "none"
                       },
                       "position": 0
                   },
                   ...
               ],
               "inputTokenCount": 4,
           }
        ]
        :param batch_size: The number of sequences in a batch
        :param max_decoder_sequence_length: Truncate/Allowed max decoder sequence length. If None then no padding.
        :param device: whether to move tensors to gpu
        :param is_generative_training: Flag for generative training like: 0123 -> 01234, 01234 -> 012345 ...
        :return: A tuple containing two dictionaries.
        The first represents the batches, This is batch io parser output.
        Second one represents the attention masks. This is bool Tensor.
        """
        if is_generative_training:
            next_token_task_corpus = BatchBuilder.create_generative_training_samples(
                data_loader_output,
                max_decoder_sequence_length,
                add_bos_and_eos,
            )
            print(f"Generative examples: {next_token_task_corpus}\n")
        else:
            next_token_task_corpus = [
                {
                    BatchBuilder.SOURCE_LANGUAGE_KEY: src[Constants.IO_PARSER_OUTPUT],
                    BatchBuilder.TARGET_LANGUAGE_KEY: tgt[Constants.IO_PARSER_OUTPUT],
                    BatchBuilder.TASK_TYPE_KEY: src[Constants.TASK_TYPE],
                    BatchBuilder.INITIAL_TOKEN_COUNT_KEY: src[Constants.INITIAL_TOKEN_COUNT],
                } for src, tgt in
                zip(data_loader_output, data_loader_output)
            ]
        batches: Dict[str, List] = {
            BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY: [],
            BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY: [],
        }
        masks: Dict[str, List] = {
            BatchBuilder.FUTURE_MASK_KEY: [],
            BatchBuilder.PADDING_MASK_KEY: [],
            BatchBuilder.TASK_TYPE_KEY: [],
            BatchBuilder.INITIAL_TOKEN_COUNT_KEY: [],
        }
        for i in range(0, len(next_token_task_corpus), batch_size):
            src_batch = [
                pair.get(BatchBuilder.SOURCE_LANGUAGE_KEY, "")
                for pair in next_token_task_corpus[i: i + batch_size]
            ]
            # src_batch = BatchBuilder.get_batch_io_parser_output(
            #     sentences=input_sentences,
            #     add_bos_and_eos=True,
            #     max_sequence_length=max_decoder_sequence_length,
            # )

            tgt_batch = [
                pair.get(BatchBuilder.TARGET_LANGUAGE_KEY, "")
                for pair in next_token_task_corpus[i: i + batch_size]
            ]
            # tgt_batch = BatchBuilder.get_batch_io_parser_output(
            #     sentences=output_sentences,
            #     add_bos_and_eos=True,
            #     max_sequence_length=max_decoder_sequence_length,
            # )
            future_mask = BatchBuilder.construct_future_mask(max_decoder_sequence_length)
            padding_mask = BatchBuilder.construct_padding_mask(src_batch)
            task_type = BatchBuilder.create_task_type_list(next_token_task_corpus[i: i + batch_size])
            # Initial Token count will only be used in inference when the batch size is only 1
            initial_token_count = next_token_task_corpus[i][BatchBuilder.INITIAL_TOKEN_COUNT_KEY]

            if device is not None:
                src_batch = src_batch.to(device)  # type: ignore
                tgt_batch = tgt_batch.to(device)  # type: ignore
                future_mask = future_mask.to(device)
                padding_mask = padding_mask.to(device)

            batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY].append(src_batch)
            batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY].append(tgt_batch)
            masks[BatchBuilder.PADDING_MASK_KEY].append(padding_mask)
            masks[BatchBuilder.FUTURE_MASK_KEY].append(future_mask)
            masks[BatchBuilder.TASK_TYPE_KEY].append(task_type)
            masks[BatchBuilder.INITIAL_TOKEN_COUNT_KEY].append(initial_token_count)
        return batches, masks

    @staticmethod
    def create_generative_training_samples(
            data_loader_output: List[dict],
            max_decoding_length: int,
            add_eos: bool,
    ) -> List[Dict[str, str]]:
        """
        After three words this function will create samples by using 1-3 words as input and 1-4 words as output.
        Which is actually next word prediction. This process will be continued until the whole sentence is covered.

        :param add_eos: Weather to add end of sentence tag or not
        :param max_decoding_length: max decoding length of the current model
        :param data_loader_output: This is a list of sentences on which the model is to be trained. ie.
        [
           {
               "inputStr": "##multiplication(107.23600916441234,784.625535184504)",
               "outputStr": "107.23600916441234 times 784.625535184504 equals?",
               "taskType": "func_to_nl_translation",
               "sentence": "##multiplication(107.23600916441234,784.625535184504) = 107.23600916441234 times 784.625535184504 equals?",
               "ioParserOutput": [
                   {
                       "token": "<BOS>",
                       "category": {
                           "type": "special",
                           "subType": "word",
                           "subSubType": "none"
                       },
                       "position": 0
                   },
                   ...
               ],
               "inputTokenCount": 4,
           }
        ]
        :return: List of dictionaries where the input key is BatchBuilder.SOURCE_LANGUAGE_KEY
        and the output key is BatchBuilder.TARGET_LANGUAGE_KEY
        """

        samples = []

        for data_loader_item in data_loader_output:
            io_parser_output = data_loader_item[Constants.IO_PARSER_OUTPUT]
            current_input_word_count = data_loader_item[Constants.INITIAL_TOKEN_COUNT]

            for i in range(len(io_parser_output) - current_input_word_count):

                # Input sequence: 1--INPUT_TOKEN_COUNT words
                input_sequence = io_parser_output[0: i + current_input_word_count]
                if input_sequence[-1][Constants.TOKEN] == SpecialTokens.ENDING.value:
                    break

                input_sequence = BatchBuilder.add_padding_and_eos(
                    input_sequence,
                    max_decoding_length,
                    add_eos,
                    is_eos_finishing_token=False,
                )

                # Output sequence: 1--INPUT_TOKEN_COUNT+1 words
                output_sequence = io_parser_output[0: i + current_input_word_count + 1]
                output_sequence = BatchBuilder.add_padding_and_eos(
                    output_sequence,
                    max_decoding_length,
                    False,
                    is_eos_finishing_token=False,
                )

                # Add the sample to the list
                sample = {
                    BatchBuilder.SOURCE_LANGUAGE_KEY: input_sequence,
                    BatchBuilder.TARGET_LANGUAGE_KEY: output_sequence,
                    BatchBuilder.TASK_TYPE_KEY: data_loader_item[Constants.TASK_TYPE],
                    BatchBuilder.INITIAL_TOKEN_COUNT_KEY: data_loader_item[Constants.INITIAL_TOKEN_COUNT],
                }
                samples.append(sample)

        return samples

    @staticmethod
    def add_padding_and_eos(
            io_parser_output: list[dict],
            max_decoding_length: int,
            add_bos_and_eos: bool,
            is_eos_finishing_token: bool = False,
    ) -> list[dict]:
        """Add padding and eos to the generated samples in case of generative training

        :param io_parser_output:
        :param max_decoding_length:
        :param add_bos_and_eos:
        :param is_eos_finishing_token:
        :return:
        """
        if add_bos_and_eos and max_decoding_length is not None and is_eos_finishing_token:
            max_decoding_length = max_decoding_length - 1
        if add_bos_and_eos and not is_eos_finishing_token:
            io_parser_output.append(
                Utility.get_special_token(len(io_parser_output), SpecialTokens.MASK_TOKEN)
            )
        if max_decoding_length is not None:
            current_length = len(io_parser_output)
            if current_length >= max_decoding_length:
                # Truncate if result length is greater than the max length
                io_parser_output = io_parser_output[:max_decoding_length]
            else:
                num_padding = max_decoding_length - current_length
                for i in range(num_padding):
                    io_parser_output.append(
                        Utility.get_special_token(current_length + i, SpecialTokens.PADDING)
                    )
        if add_bos_and_eos and is_eos_finishing_token:
            io_parser_output.append(
                Utility.get_special_token(len(io_parser_output), SpecialTokens.MASK_TOKEN)
            )
        return io_parser_output

    @staticmethod
    def get_next_token_prediction_task_corpus():
        pass

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
            BatchBuilder.PADDING_MASK_KEY: [],
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

            if device is not None:
                src_batch = src_batch.to(device)  # type: ignore
                tgt_batch = tgt_batch.to(device)  # type: ignore
                future_mask = future_mask.to(device)
                encoder_padding_mask = encoder_padding_mask.to(device)

            batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY].append(src_batch)
            batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY].append(tgt_batch)
            masks[BatchBuilder.PADDING_MASK_KEY].append(encoder_padding_mask)
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

    @staticmethod
    def create_task_type_list(batch_token_task_corpus: list[dict]):
        """Gather all task type in a list for a batch.

        :param batch_token_task_corpus: This is the next_token_task_corpus for a batch
        :return: A list of task type.
        """
        result = [item[BatchBuilder.TASK_TYPE_KEY] for item in batch_token_task_corpus]
        return result


class TestUtils(unittest.TestCase):

    def test_create_generative_training_samples(self):
        from data_loader.data_loader import DataLoader

        data_loader = DataLoader()
        batch_size = 2
        max_decoding_length = 16
        task_generator_indexes = [3]
        generator_range = 2
        number_of_batch = generator_range * len(task_generator_indexes)
        seed = 42

        data_loader_result = data_loader.create_data_loader_output(
            batch_size=batch_size,
            number_of_batch=number_of_batch,
            add_bos_and_eos=True,
            max_sequence_length=max_decoding_length,
            task_generator_indexes=task_generator_indexes,
            generator_indexes=[i for i in range(generator_range)],
            identifier=0,
            shuffle=True,
            seed=seed,
        )
        generative_samples = BatchBuilder.create_generative_training_samples(
            data_loader_result,
            max_decoding_length,
            add_eos=True
        )
        print(generative_samples)
        # This test will fail by design, please check the printed generative_samples
        expected_result = []
        self.assertEqual(generative_samples, expected_result)

    def test_get_sentence_io_parser_output(self):
        # Full length with padding, BOS and EOS
        sentence = "one two three four five six seven eight"

        # need to remove <BOS> from target, We will use this to find the loss by comparing to output
        # target = "<BOS> one two three four five six seven eight <EOS>"

        # input = "<BOS> one two three four five six seven eight <EOS>"

        # need to remove garbage from output, this is right shifted result of the input
        # output = "one two three four five six seven eight <EOS> garbage"

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
            {'token': '<EOS>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
             'position': 9},
            {'token': '<PAD>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
             'position': 10},
            {'token': '<PAD>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
             'position': 11},
        ]
        self.assertEqual(expected_result, io_parser_output)
        # Truncated length with Padding, BOS and EOS
        io_parser_output = BatchBuilder.get_sentence_io_parser_output(
            sentence,
            add_bos_and_eos=True,
            max_sequence_length=6,
        )
        expected_result = [
            {'token': '<BOS>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'}, 'position': 0},
            {'token': 'one', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 1},
            {'token': 'two', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 2},
            {'token': 'three', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 3},
            {'token': 'four', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 4},
            {'token': 'five', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 5},
        ]
        self.assertEqual(expected_result, io_parser_output)
        # Full length without Padding, BOS and EOS
        io_parser_output = BatchBuilder.get_sentence_io_parser_output(
            sentence,
            add_bos_and_eos=False,
            max_sequence_length=None,
        )
        expected_result = [
            {'token': 'one', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 0},
            {'token': 'two', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 1},
            {'token': 'three', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 2},
            {'token': 'four', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 3},
            {'token': 'five', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 4},
            {'token': 'six', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 5},
            {'token': 'seven', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 6},
            {'token': 'eight', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 7},
        ]
        self.assertEqual(expected_result, io_parser_output)
        # Truncated length without Padding, BOS and EOS
        io_parser_output = BatchBuilder.get_sentence_io_parser_output(
            sentence,
            add_bos_and_eos=False,
            max_sequence_length=3,
        )
        expected_result = [
            {'token': 'one', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 0},
            {'token': 'two', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 1},
            {'token': 'three', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 2},
        ]
        self.assertEqual(expected_result, io_parser_output)
        # Full length with Padding without BOS and EOS
        io_parser_output = BatchBuilder.get_sentence_io_parser_output(
            sentence,
            add_bos_and_eos=False,
            max_sequence_length=10,
        )
        expected_result = [
            {'token': 'one', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 0},
            {'token': 'two', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 1},
            {'token': 'three', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 2},
            {'token': 'four', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 3},
            {'token': 'five', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 4},
            {'token': 'six', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 5},
            {'token': 'seven', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 6},
            {'token': 'eight', 'category': {'type': 'word', 'subType': 'default', 'subSubType': 'none'}, 'position': 7},
            {'token': '<PAD>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'}, 'position': 8},
            {'token': '<PAD>', 'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'}, 'position': 9}]
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
