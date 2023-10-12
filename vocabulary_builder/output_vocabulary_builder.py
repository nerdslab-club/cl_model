from dataclasses import dataclass
from typing import Optional

from cl_data.src.constants import Constants
from vocabulary_builder.category_vocabulary_builder import OutputTokenClassificationHeadVocabItem


@dataclass
class OutputVocabItem:
    token: any

    def __hash__(self):
        """
        Calculate a hash value based on the fields that determine equality

        :return: The calculated hash
        """
        return hash(
            (
                self.token,
            )
        )

    def __str__(self):
        return f"OutputTokenVocabItem( token={self.token} )"


class OutputVocabBuilder:
    """
    integer token -> output token vocab item
    """
    INDEX = "index"
    OUTPUT_TOKEN_CLASSIFICATION_HEAD_VOCAB_ITEM = "output_token_classification_head_vocab_item"
    INDEX_TO_OUTPUT = "index_to_output"
    OUTPUT_TO_INDEX = "output_to_index"

    def __init__(
            self,
            corpus_of_io_parser_output: Optional[list[list[dict]]],
            index_to_output_token_classification_head_vocab_item: dict,
    ):
        self.index_to_output_vocabularies = index_to_output_token_classification_head_vocab_item
        # Initializing the output vocabulary items
        for index, output_token_classification_head_vocab_item in self.index_to_output_vocabularies.items():
            output_vocabulary_item = {
                OutputVocabBuilder.INDEX: index,
                OutputVocabBuilder.OUTPUT_TOKEN_CLASSIFICATION_HEAD_VOCAB_ITEM: output_token_classification_head_vocab_item,
                OutputVocabBuilder.INDEX_TO_OUTPUT: {},
                OutputVocabBuilder.OUTPUT_TO_INDEX: {},
            }
            self.index_to_output_vocabularies[index] = output_vocabulary_item

        self.output_token_classification_head_vocab_item_to_output_vocabularies = {
            v[OutputVocabBuilder.OUTPUT_TOKEN_CLASSIFICATION_HEAD_VOCAB_ITEM]: v for k, v in
            self.index_to_output_vocabularies.items()
        }
        if not corpus_of_io_parser_output:
            return
        for io_parser_output in corpus_of_io_parser_output:
            self.add_tokens(
                self.encode_io_parser_item_into_output_vocab_item(io_parser_output)
            )

    def add_tokens(self, vocab_items: list[tuple[OutputTokenClassificationHeadVocabItem, OutputVocabItem]]) -> None:
        """
        Adds OutputVocabItem to the vocabulary

        :param vocab_items: List of tuple of output token classification head vocab items and output vocab items.
        :return: None
        """
        for classification_head_item, token in vocab_items:
            # TODO as we are converting list to string in the response parser this need to be reverted
            if isinstance(token, list):
                token = str(token)
            output_vocabulary = self.output_token_classification_head_vocab_item_to_output_vocabularies[classification_head_item]
            output_vocabulary_index = output_vocabulary[OutputVocabBuilder.INDEX]
            vocab_item_to_index = output_vocabulary[OutputVocabBuilder.OUTPUT_TO_INDEX]
            index_to_vocab_item = output_vocabulary[OutputVocabBuilder.INDEX_TO_OUTPUT]

            if token not in vocab_item_to_index:
                i = len(vocab_item_to_index.items())
                vocab_item_to_index[token] = i
                index_to_vocab_item[i] = token
                output_vocabulary[OutputVocabBuilder.OUTPUT_TO_INDEX] = vocab_item_to_index
                output_vocabulary[OutputVocabBuilder.INDEX_TO_OUTPUT] = index_to_vocab_item
                self.output_token_classification_head_vocab_item_to_output_vocabularies[classification_head_item] = output_vocabulary
                self.index_to_output_vocabularies[output_vocabulary_index] = output_vocabulary

    def encoder(self, io_parser_output: list[dict], is_only_probability=False) -> list[tuple[int, int]] | list[int]:
        """
        Tokenize io parser output -> (classification head id, output vocab token id)

        :param is_only_probability: if Ture then provide only the output vocab token id otherwise both
        :param io_parser_output: Output of the io parser with or with padding and special tokens
        :return: list of tuple of (classification head id, output vocab token id)
        """
        result = []
        vocab_items = self.encode_io_parser_item_into_output_vocab_item(io_parser_output)
        for classification_head_item, token in vocab_items:
            output_vocabulary = self.output_token_classification_head_vocab_item_to_output_vocabularies[classification_head_item]
            if is_only_probability:
                result.append(output_vocabulary[OutputVocabBuilder.OUTPUT_TO_INDEX][token])
            else:
                result.append(
                    (
                        output_vocabulary[OutputVocabBuilder.INDEX],
                        output_vocabulary[OutputVocabBuilder.OUTPUT_TO_INDEX][token]
                    ),
                )
        return result

    def batch_encoder(
        self,
        batch_io_parser_output: list[list[dict]],
        is_only_probability=False,
    ) -> list[list[tuple[int, int]]] | list[list[int]]:
        """
        Batch tokenize io parser output -> (classification head id, output vocab token id)

        :param is_only_probability: if Ture then provide only the output vocab token id otherwise both
        :param batch_io_parser_output: batch of io_parser_output
        :return: Batch of list tuple of (classification head id, output vocab token id)
        """
        batch_tokens = [
            self.encoder(io_parser_output, is_only_probability)
            for io_parser_output in batch_io_parser_output
        ]
        return batch_tokens

    def decode(self, ids: list[tuple[int, int]]) -> list[tuple[OutputTokenClassificationHeadVocabItem, OutputVocabItem]]:
        """Decode tokens id into output vocab item. (classification head id, output vocab token id) -> output vocab items

        :param ids: list of tuple of (classification head id, output vocab token id)
        :return: list of tuple of output token classification head item and output vocab items
        """
        result = []

        for classification_head_item_id, token_id in ids:
            vocabulary = self.index_to_output_vocabularies[classification_head_item_id]
            result.append(
                (
                    vocabulary[OutputVocabBuilder.INDEX],
                    vocabulary[OutputVocabBuilder.INDEX_TO_OUTPUT][token_id],
                ),
            )
        return result

    def batch_decode(self, list_of_ids: list[list[tuple[int, int]]]) -> list[list[tuple[OutputTokenClassificationHeadVocabItem, OutputVocabItem]]]:
        """Decode list of tokens into batch io parser output.
        batch (classification head id, output vocab token id) ->
        batch (output token classification head vocab item, output vocab items)

        :param list_of_ids: batch of (classification head id, output vocab token id)
        :return: batch (output token classification head vocab item, output vocab items)
        """
        batch_vocab_items = [
            self.decode(ids)
            for ids in list_of_ids
        ]
        return batch_vocab_items

    @staticmethod
    def encode_io_parser_item_into_output_vocab_item(
            io_parser_output: list[dict],
    ) -> list[tuple[OutputTokenClassificationHeadVocabItem, OutputVocabItem]]:
        """
        Converts the list of io parser item dict into list of OutputVocabItem
        with it's OutputTokenClassificationHeadVocabItem

        :param io_parser_output: Output of the io parser with or with padding and special tokens
        :return: list of output token classification head vocab item as tuple and output vocab item
        """
        tokens = []
        for io_parser_item in io_parser_output:
            token: any = io_parser_item.get(Constants.TOKEN)
            category_map: dict = io_parser_item.get(Constants.CATEGORY)
            classification_head_item = OutputTokenClassificationHeadVocabItem(
                category_type=category_map.get(Constants.CATEGORY_TYPE),
                category_subtype=category_map.get(Constants.CATEGORY_SUB_TYPE),
            )
            tokens.append((classification_head_item, token))
        return tokens
