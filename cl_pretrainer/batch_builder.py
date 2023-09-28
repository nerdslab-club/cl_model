from typing import Dict, List, Tuple, Optional

import torch

from cl_data.io_parser.io_parser import IoParser
from cl_data.src.utility import Utility


class BatchBuilder:
    SOURCE_LANGUAGE_KEY = "inputStr"
    TARGET_LANGUAGE_KEY = "outputStr"

    @staticmethod
    def get_batch_io_parser_output(
            sentences: list[str],
            add_bos_and_eos: bool,
            max_sequence_length: int,
    ) -> list[list[dict]]:
        """Given list of sentences it creates list of io parser output with special tokens and paddings

        :param sentences: List of sentences.
        :param add_bos_and_eos: Flag for weather to add BOS and EOS to the Token list.
        :param max_sequence_length: Max length until which padding will be added. If None then no padding.
        :return:
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
    def get_sentence_io_parser_output(sentence: str, add_bos_and_eos: bool, max_sequence_length: int) -> list[dict]:
        """Given a sentence it called io parser on the sentence with special tokens and paddings

        :param sentence: A normal sentence string.
        :param add_bos_and_eos: Flag for weather to add BOS and EOS to the Token list.
        :param max_sequence_length: Max length until which padding will be added. If None then no padding.
        :return:
        """
        io_parser_tuples = IoParser().create_value_list_from_input(sentence)
        return Utility.create_io_map_from_io_tuple(
            input_list=io_parser_tuples,
            add_bos_and_eos=add_bos_and_eos,
            max_length=max_sequence_length
        )

    @staticmethod
    def construct_batches_for_transformer(
            corpus: List[Dict[str, str]],
            batch_size: int,
            max_encoder_sequence_length: int,
            max_decoder_sequence_length: int,
            device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, list[list[dict]]], Dict[str, List[torch.Tensor]]]:
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
        """
        batches: Dict[str, List] = {"src": [], "tgt": []}
        masks: Dict[str, List] = {"src": [], "tgt": []}
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

            # TODO fix the mask part
            # suggestion : need to get the mask from get_batch_io_parser_output function as tuple
            # src_padding_mask = src_batch != pad_token_id
            # future_mask = construct_future_mask(tgt_batch.shape[-1])

            # Move tensors to gpu; if available
            if device is not None:
                src_batch = src_batch.to(device)  # type: ignore
                tgt_batch = tgt_batch.to(device)  # type: ignore
                src_padding_mask = src_padding_mask.to(device)
                future_mask = future_mask.to(device)
            batches["src"].append(src_batch)
            batches["tgt"].append(tgt_batch)
            masks["src"].append(src_padding_mask)
            masks["tgt"].append(future_mask)
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
