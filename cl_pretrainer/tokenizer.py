import unittest

import torch
from torch import Tensor

from cl_data.src.constants import Constants, SpecialTokens
from data_loader.data_loader import DataLoader
from vocabulary_builder.category_vocabulary_builder import OutputTokenClassificationHeadVocabItem, CategoryVocabBuilder
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder, OutputVocabItem


class Tokenizer:

    @staticmethod
    def get_token_vocab_size(index_to_output_vocabularies: dict[int, dict]) -> int:
        """All type of token are counted and added to get total token count.

        :param index_to_output_vocabularies: This is the map which is in output vocabulary builder
        :return: Total token count
        """
        total_vocab_size = 2  # Adding "nOtMyToKeN" and <MUSK> vocab index here
        for index, output_vocabulary in index_to_output_vocabularies.items():
            current_count = len(output_vocabulary[OutputVocabBuilder.INDEX_TO_OUTPUT].items()) - 1
            # print(f"Head index {index} vocab count: {current_count}")
            total_vocab_size += current_count
        return total_vocab_size

    @staticmethod
    def get_index_of_token(
            output_vocab_builder: OutputVocabBuilder,
            io_parser_output_item: dict,
    ) -> int:
        """Get the index number of a specific token

        :param output_vocab_builder: This is the output vocab builder instance
        :param io_parser_output_item: This is the item of which the index is to be determined
        :return: The index number of a specific token
        """
        # Get head index
        category_map: dict = io_parser_output_item.get(Constants.CATEGORY)
        output_token_classification_head_vocab_item: OutputTokenClassificationHeadVocabItem = \
            OutputTokenClassificationHeadVocabItem(
                category_type=category_map.get(Constants.CATEGORY_TYPE),
                category_subtype=category_map.get(Constants.CATEGORY_SUB_TYPE),
            )
        current_vocabulary = output_vocab_builder.output_token_classification_head_vocab_item_to_output_vocabularies \
            .get(output_token_classification_head_vocab_item)
        head_index = current_vocabulary[OutputVocabBuilder.INDEX]

        # Get token index
        token: any = io_parser_output_item[Constants.TOKEN]
        output_vocab_item: OutputVocabItem = OutputVocabItem(token)

        if output_vocab_item == OutputVocabItem(Constants.NOT_MY_TOKEN):
            return 0
        elif output_vocab_item == OutputVocabItem(SpecialTokens.MASK_TOKEN.value):
            return 1

        token_index = current_vocabulary[OutputVocabBuilder.OUTPUT_TO_INDEX][output_vocab_item]
        trailing_index = 1
        for index in range(head_index + 1):
            if index == head_index:
                break
            current_vocabulary = output_vocab_builder.index_to_output_vocabularies[index]
            current_vocab_size = len(current_vocabulary[OutputVocabBuilder.INDEX_TO_OUTPUT].keys()) - 1
            # print(f'Head index is: {index} with vocab of: {current_vocab_size}')
            trailing_index += current_vocab_size

        # print(f'Token index is: {token_index}')
        return trailing_index + token_index

    @staticmethod
    def get_sentence_input_tensor(output_vocab_builder: OutputVocabBuilder, io_parser_output: list[dict]) -> Tensor:
        """Create word index tensor for sentence

        :param output_vocab_builder: This is the output vocab builder instance
        :param io_parser_output:
        [
             {
                "token":126,
                "category":{
                   "type":"integer",
                   "subType":"default",
                   "subSubType":"none"
                },
                "position":0
             },
             ...
         ]
        :return: word index tensor for sentence
        """
        input_indices = [
            Tokenizer.get_index_of_token(output_vocab_builder, io_parser_output_item)
            for io_parser_output_item in io_parser_output
        ]
        # Convert to PyTorch tensor
        input_indices_tensor = torch.Tensor(input_indices)
        return input_indices_tensor

    @staticmethod
    def get_batch_input_tensor(output_vocab_builder: OutputVocabBuilder,
                               batch_io_parser_output: list[list[dict]]) -> Tensor:
        """Create word index tensor for batch of sentence

        :param output_vocab_builder: This is the output vocab builder instance
        :param batch_io_parser_output: Batch of io_parser_output
        :return: Word index tensor for batch of sentence
        """
        input_tensors = []
        for io_parser_output in batch_io_parser_output:
            input_tensor = Tokenizer.get_sentence_input_tensor(output_vocab_builder, io_parser_output)
            input_tensors.append(input_tensor)
        batch_input_tensor = torch.stack(input_tensors)
        return batch_input_tensor


class TestTokenizer(unittest.TestCase):

    def test_tokenizer_get_token_vocab_size(self) -> tuple:
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        batch_size = 1 if device == torch.device("cpu") else 4
        task_generator_indexes = [3]
        generator_range = 1 if device == torch.device("cpu") else 10

        max_decoding_length = 16
        number_of_batch = generator_range * len(task_generator_indexes)
        seed = 42
        add_bos_and_eos = True

        data_loader = DataLoader()
        data_loader_result = data_loader.create_data_loader_output(
            batch_size=batch_size,
            number_of_batch=number_of_batch,
            add_bos_and_eos=add_bos_and_eos,
            max_sequence_length=max_decoding_length,
            task_generator_indexes=task_generator_indexes,
            generator_indexes=[i for i in range(generator_range)],
            identifier=0,
            shuffle=True,
            seed=seed,
        )
        print(data_loader_result)
        corpus_io_parser_output = [item[Constants.IO_PARSER_OUTPUT] for item in data_loader_result]
        # Initialize category vocabulary builder instance
        category_vocab_builder = CategoryVocabBuilder(corpus_io_parser_output)

        print(f"Output token classification head count:"
              f" {len(category_vocab_builder.index_to_output_token_classification_head_vocab_item.keys())}\n")

        # Initialize output vocabulary builder instance
        output_vocab_builder = OutputVocabBuilder(
            corpus_of_io_parser_output=corpus_io_parser_output,
            index_to_output_token_classification_head_vocab_item=
            category_vocab_builder.index_to_output_token_classification_head_vocab_item
        )

        index_to_output_vocabularies = output_vocab_builder.index_to_output_vocabularies
        print(f"index_to_output_vocabularies: {index_to_output_vocabularies}")
        token_vocab_size = Tokenizer.get_token_vocab_size(index_to_output_vocabularies)
        print(f"Token vocab size: {token_vocab_size}")
        return output_vocab_builder, corpus_io_parser_output

    def test_get_index_of_token(self):
        output_vocab_builder, corpus_io_parser_output = self.test_tokenizer_get_token_vocab_size()
        current_token = corpus_io_parser_output[0][1]
        print(f'Current token is: {current_token}')

        value = Tokenizer.get_index_of_token(output_vocab_builder, io_parser_output_item=current_token)
        print(f'Current index is: {value}')


if __name__ == "__main__":
    unittest.main()
