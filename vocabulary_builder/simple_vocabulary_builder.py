import unittest
from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from cl_data.src.constants import (
    SpecialTokens,
    CategoryType,
    CategorySubType,
    CategorySubSubType,
    Constants,
)
from cl_pretrainer.batch_builder import BatchBuilder


@dataclass
class VocabItem:
    token: any
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
                self.token,
                self.category_type,
                self.category_subtype,
                self.category_sub_subtype,
            )
        )


class SimpleVocabBuilder:
    def __init__(self, corpus_of_io_parser_output: Optional[list[list[dict]]]):
        self.vocab_item_to_index = {
            SimpleVocabBuilder.get_special_vocab_item(SpecialTokens.BEGINNING): 0,
            SimpleVocabBuilder.get_special_vocab_item(SpecialTokens.ENDING): 1,
            SimpleVocabBuilder.get_special_vocab_item(SpecialTokens.PADDING): 2,
        }
        self.index_to_vocab_item = {v: k for k, v in self.vocab_item_to_index.items()}
        if not corpus_of_io_parser_output:
            return
        for io_parser_output in corpus_of_io_parser_output:
            self.add_tokens(
                self.encode_io_parser_item_into_vocab_item(io_parser_output)
            )

    def add_tokens(self, vocab_items: list[VocabItem]) -> None:
        """
        Adds VocabItem to the vocabulary

        :param vocab_items: List of vocab items.
        :return: None
        """
        for token in vocab_items:
            if token not in self.vocab_item_to_index:
                i = len(self.vocab_item_to_index.items())
                self.vocab_item_to_index[token] = i
                self.index_to_vocab_item[i] = token

    def encoder(self, io_parser_output: list[dict]) -> list[int]:
        """
        Tokenize io parser output -> vocab items -> integer token

        :param io_parser_output: Output of the io parser with or with padding and special tokens.
        :return: list of integer tokens.
        """
        vocab_items = self.encode_io_parser_item_into_vocab_item(io_parser_output)
        return [self.vocab_item_to_index[vocab_item] for vocab_item in vocab_items]

    def batch_encoder(
        self, batch_io_parser_output: list[list[dict]]
    ) -> list[list[int]]:
        """
        Batch tokenize io parser output -> vocab items -> integer token

        :param batch_io_parser_output: batch of io_parser_output
        :return: Batch of list of integer tokens.
        """
        batch_tokens = [
            self.encoder(io_parser_output)
            for io_parser_output in batch_io_parser_output
        ]
        return batch_tokens

    def decode(self, tokens: list[int]) -> list[dict]:
        """Decode tokens into io parser output. integer token -> vocab items -> io parser output

        :param tokens: list of integer tokens.
        :return: io parser output.
        """
        vocab_items = [self.index_to_vocab_item[token] for token in tokens]
        return self.decoder_vocab_item_into_io_parser_output(vocab_items)

    def batch_decode(self, list_of_tokens: list[list[int]]) -> list[list[dict]]:
        """Decode list of tokens into batch io parser output.
        batch integer token -> batch vocab items -> batch io parser output

        :param list_of_tokens: batch of integer tensor.
        :return: batch io parser output.
        """
        batch_io_parser_output = []
        for tokens in list_of_tokens:
            vocab_items = [self.index_to_vocab_item[token] for token in tokens]
            batch_io_parser_output.append(self.decoder_vocab_item_into_io_parser_output(vocab_items))
        return batch_io_parser_output

    @staticmethod
    def encode_io_parser_item_into_vocab_item(
        io_parser_output: list[dict],
    ) -> list[VocabItem]:
        """
        Converts the list of io parser item dict into list of VocabItem

        :param io_parser_output: Output of the io parser with or with padding and special tokens.
        :return: vocab items
        """
        tokens = []
        for io_parser_item in io_parser_output:
            category_map: dict = io_parser_item.get(Constants.CATEGORY)
            tokens.append(
                VocabItem(
                    token=io_parser_item.get(Constants.TOKEN),
                    category_type=category_map.get(Constants.CATEGORY_TYPE),
                    category_subtype=category_map.get(Constants.CATEGORY_SUB_TYPE),
                    category_sub_subtype=category_map.get(
                        Constants.CATEGORY_SUB_SUB_TYPE
                    ),
                ),
            )
        return tokens

    @staticmethod
    def batch_encode_io_parser_item_into_vocab_item(
        batch_of_io_parser_output: list[list[dict]],
    ) -> list[list[VocabItem]]:
        """Convert batch io parser output into batch of vocab item list

        :param batch_of_io_parser_output: batch of io_parser_output
        :return: batch of vocab items
        """
        list_of_tokens = [
            SimpleVocabBuilder.encode_io_parser_item_into_vocab_item(io_parser_output)
            for io_parser_output in batch_of_io_parser_output
        ]
        return list_of_tokens

    @staticmethod
    def decoder_vocab_item_into_io_parser_output(
        vocab_items: list[VocabItem],
    ) -> list[dict]:
        """
        Convert vocab items into io parser output
        :param vocab_items: list of vocab item
        :return: io parser output
        """
        io_parser_output = []
        for i, vocab_item in enumerate(vocab_items):
            io_parser_output.append(
                {
                    Constants.TOKEN: vocab_item.token,
                    Constants.CATEGORY: {
                        Constants.CATEGORY_TYPE: vocab_item.category_type,
                        Constants.CATEGORY_SUB_TYPE: vocab_item.category_subtype,
                        Constants.CATEGORY_SUB_SUB_TYPE: vocab_item.category_sub_subtype,
                    },
                    Constants.POSITION: i,
                }
            )
        return io_parser_output

    @staticmethod
    def get_special_vocab_item(special_token: SpecialTokens):
        """
        Get special tokens vocab item based on given special token.

        :param special_token: any item from SpecialTokens enum
        :return: A vocab item
        """
        if special_token == SpecialTokens.PADDING:
            return VocabItem(
                SpecialTokens.PADDING.value,
                CategoryType.SPECIAL.value,
                CategorySubType.WORD.value,
                CategorySubSubType.NONE.value,
            )
        elif special_token == SpecialTokens.BEGINNING:
            return VocabItem(
                SpecialTokens.BEGINNING.value,
                CategoryType.SPECIAL.value,
                CategorySubType.WORD.value,
                CategorySubSubType.NONE.value,
            )
        elif special_token == SpecialTokens.ENDING:
            return VocabItem(
                SpecialTokens.ENDING.value,
                CategoryType.SPECIAL.value,
                CategorySubType.WORD.value,
                CategorySubSubType.NONE.value,
            )
        elif special_token == SpecialTokens.MASK_TOKEN:
            return VocabItem(
                SpecialTokens.MASK_TOKEN.value,
                CategoryType.SPECIAL.value,
                CategorySubType.WORD.value,
                CategorySubSubType.NONE.value,
            )
        elif special_token == SpecialTokens.SEPARATOR_TOKEN:
            return VocabItem(
                SpecialTokens.SEPARATOR_TOKEN.value,
                CategoryType.SPECIAL.value,
                CategorySubType.WORD.value,
                CategorySubSubType.NONE.value,
            )


class TestVocabulary(unittest.TestCase):
    def test_init_vocab(self):
        input_sentences = ["Hello my name is Joris and I was born with the name Joris."]
        corpus_io_parser_output = BatchBuilder.get_batch_io_parser_output(input_sentences, True, 16)
        vocab = SimpleVocabBuilder(corpus_io_parser_output)
        print(vocab.vocab_item_to_index)
        self.assertEqual(vocab.index_to_vocab_item[2], SimpleVocabBuilder.get_special_vocab_item(SpecialTokens.PADDING))
        self.assertEqual(vocab.index_to_vocab_item[0], SimpleVocabBuilder.get_special_vocab_item(SpecialTokens.BEGINNING))
        self.assertEqual(vocab.index_to_vocab_item[1], SimpleVocabBuilder.get_special_vocab_item(SpecialTokens.ENDING))
        self.assertEqual(len(vocab.vocab_item_to_index.items()), 15)

    def test_encode(self):
        input_sequence = "Hello my name is Joris and I was born with the name Joris"
        corpus = BatchBuilder.get_batch_io_parser_output([input_sequence], True, 16)
        io_parser_output = BatchBuilder.get_sentence_io_parser_output(input_sequence, True, 18)

        output = SimpleVocabBuilder(corpus).encoder(io_parser_output)
        self.assertEqual(
            output, [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 7, 2, 2, 2, 1]
        )

    def test_decode(self):
        input_sequence = "Hello my name is Joris and I was born with the name Joris"
        corpus = BatchBuilder.get_batch_io_parser_output([input_sequence], True, 16)
        decoded_output = SimpleVocabBuilder(corpus).decode([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 5, 7, 2, 2, 2, 1])
        io_parser_output = BatchBuilder.get_sentence_io_parser_output(input_sequence, True, 18)
        self.assertEqual(decoded_output, io_parser_output)


if __name__ == "__main__":
    vocab_item_one = VocabItem(10, "bool", "default", "param_one")
    vocab_item_two = VocabItem(10, "bool", "default", "param_one")
    vocab_item_three = VocabItem(10, "bool", "default", "param_two")

    print(vocab_item_one == vocab_item_two)
    print(vocab_item_one == vocab_item_three)
    unittest.main()

