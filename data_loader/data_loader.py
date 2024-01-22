import random
import unittest

from cl_data.src.constants import Constants
from cl_data.src.random_value_generator import RandomValueGenerator
from cl_pretrainer.batch_builder import BatchBuilder
from data_generator import DataGenerator


class DataLoader:
    def __init__(self):
        self.data_generator = DataGenerator()
        # 'equal', 'answer'
        self.catalyst_list = ['=']

    def create_sentence_using_input_and_output(self, example: dict):
        random_index = RandomValueGenerator.generate_random_integer(0, len(self.catalyst_list))
        chosen_catalyst = self.catalyst_list[random_index]

        # Generate input and output strings
        input_str = example['inputStr']
        output_str = example['outputStr']

        # Combine using the chosen catalyst
        combined_str = f"{input_str} {chosen_catalyst} {output_str}"
        example['sentence'] = combined_str
        return combined_str

    def create_data_loader_output(
            self,
            count: int,
            task_generator_index: int | None,
            generator_index: int | None,
            identifier: int | None,
            add_bos_and_eos: bool,
            max_sequence_length: int | None,
    ) -> list[dict]:
        data_loader_output = self.data_generator.generate_batch_of(count, task_generator_index, generator_index, identifier)
        for data in data_loader_output:
            self.create_sentence_using_input_and_output(data)
            data['io_parser_output'] = BatchBuilder.get_sentence_io_parser_output(
                data['sentence'],
                add_bos_and_eos,
                max_sequence_length,
            )
            print(data['sentence'])
            print(data['io_parser_output'])
            for io_parser_item in data['io_parser_output']:
                token: any = io_parser_item[Constants.TOKEN]
                position: int = io_parser_item[Constants.POSITION]
                if token in self.catalyst_list:
                    data['input_token_count'] = position
        print(data_loader_output)
        return data_loader_output

        # {
        #     "io_parser_output": [{}],
        #     "input_token_count": 9,
        #     "task_type": ""
        # }


class DataLoaderTest(unittest.TestCase):

    def test_create_sentence_using_input_and_output(self):
        exampl_dict = {
            "inputStr": f"Adding {5} to {4}",
            "outputStr": f"##addition(5,4)",
        }
        data_loader = DataLoader()
        result = data_loader.create_sentence_using_input_and_output(exampl_dict)
        self.assertEqual(result, 'Adding 5 to 4 = ##addition(5,4)')


    def test_create_data_loader_output(self):
        data_loader = DataLoader()
        data_loader.create_data_loader_output(4, True, 24)


if __name__ == "__main__":
    unittest.main()
