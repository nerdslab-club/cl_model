from dataclasses import dataclass
from typing import Optional

from cl_data.src.constants import CategoryType, CategorySubType, CategorySubSubType, Constants


@dataclass
class CategoryVocabItem:
    category_type: str
    category_subtype: str
    category_sub_subtype: str

    def __hash__(self):
        """
        Calculate a hash value based on the fields that determine equality

        :return: The calculated hash
        """
        return hash(
            (
                self.category_type,
                self.category_subtype,
                self.category_sub_subtype,
            )
        )


@dataclass
class OutputTokenClassificationHeadVocabItem:
    category_type: str
    category_subtype: str

    def __hash__(self):
        """
        Calculate a hash value based on the fields that determine equality

        :return: The calculated hash
        """
        return hash(
            (
                self.category_type,
                self.category_subtype,
            )
        )


class CategoryVocabBuilder:
    """
    integer token -> Category vocab item -> Category map
    """
    def __init__(self, corpus_of_io_parser_output: Optional[list[list[dict]]]):
        self.category_vocab_item_to_index = {
            CategoryVocabBuilder.get_special_category_vocab_item(): 0,
        }
        self.index_to_category_vocab_item = {v: k for k, v in self.category_vocab_item_to_index.items()}
        if not corpus_of_io_parser_output:
            return
        for io_parser_output in corpus_of_io_parser_output:
            self.add_tokens(
                self.encode_io_parser_item_into_category_vocab_item(io_parser_output)
            )

    def add_tokens(self, category_vocab_items: list[CategoryVocabItem]) -> None:
        """
        Adds CategoryVocabItem to the vocabulary

        :param category_vocab_items: List of category vocab items.
        :return: None
        """
        for token in category_vocab_items:
            if token not in self.category_vocab_item_to_index:
                i = len(self.category_vocab_item_to_index.items())
                self.category_vocab_item_to_index[token] = i
                self.index_to_category_vocab_item[i] = token

    def encoder(self, io_parser_output: list[dict]) -> list[int]:
        """
        Tokenize io parser output -> category vocab items -> integer token

        :param io_parser_output: Output of the io parser with or with padding and special tokens.
        :return: list of integer tokens.
        """
        vocab_items = self.encode_io_parser_item_into_category_vocab_item(io_parser_output)
        return [self.category_vocab_item_to_index[vocab_item] for vocab_item in vocab_items]

    def batch_encoder(
            self, batch_io_parser_output: list[list[dict]]
    ) -> list[list[int]]:
        """
        Batch tokenize io parser output -> category vocab items -> integer token

        :param batch_io_parser_output: batch of io_parser_output
        :return: Batch of list of integer tokens.
        """
        batch_tokens = [
            self.encoder(io_parser_output)
            for io_parser_output in batch_io_parser_output
        ]
        return batch_tokens

    def decode(self, tokens: list[int]) -> list[dict]:
        """Decode tokens into io parser output. integer token -> category vocab items -> category map

        :param tokens: list of integer tokens.
        :return: category map list
        """
        vocab_items = [self.index_to_category_vocab_item[token] for token in tokens]
        return self.decoder_category_vocab_item_into_category_map(vocab_items)

    def batch_decode(self, list_of_tokens: list[list[int]]) -> list[list[dict]]:
        """Decode list of tokens into batch category map.
        batch integer token -> batch category vocab items -> batch category map

        :param list_of_tokens: batch of integer tensor.
        :return: batch category map.
        """
        batch_category_map = []
        for tokens in list_of_tokens:
            vocab_items = [self.index_to_category_vocab_item[token] for token in tokens]
            batch_category_map.append(self.decoder_category_vocab_item_into_category_map(vocab_items))
        return batch_category_map

    @staticmethod
    def encode_io_parser_item_into_category_vocab_item(
            io_parser_output: list[dict],
    ) -> list[CategoryVocabItem]:
        """
        Converts the list of io parser item dict into list of category vocab item

        :param io_parser_output: Output of the io parser with or with padding and special tokens.
        :return: category vocab items
        """
        tokens = []
        for io_parser_item in io_parser_output:
            category_map: dict = io_parser_item.get(Constants.CATEGORY)
            tokens.append(
                CategoryVocabItem(
                    category_type=category_map.get(Constants.CATEGORY_TYPE),
                    category_subtype=category_map.get(Constants.CATEGORY_SUB_TYPE),
                    category_sub_subtype=category_map.get(Constants.CATEGORY_SUB_SUB_TYPE),
                ),
            )
        return tokens

    @staticmethod
    def decoder_category_vocab_item_into_category_map(
            category_vocab_items: list[CategoryVocabItem],
    ) -> list[dict]:
        """
        Convert category vocab items into io parser output

        :param category_vocab_items: list of category vocab item
        :return: category map list
        """
        category_maps = []
        for i, vocab_item in enumerate(category_vocab_items):
            category_maps.append(
                {
                    Constants.CATEGORY_TYPE: vocab_item.category_type,
                    Constants.CATEGORY_SUB_TYPE: vocab_item.category_subtype,
                    Constants.CATEGORY_SUB_SUB_TYPE: vocab_item.category_sub_subtype,
                }
            )
        return category_maps

    @staticmethod
    def get_special_category_vocab_item():
        """
        Get special tokens vocab item

        :return: A category vocab item
        """
        return CategoryVocabItem(
            CategoryType.SPECIAL.value,
            CategorySubType.WORD.value,
            CategorySubSubType.NONE.value,
        )
