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
        """ This function will concat the input str and output str of the generated data,
        so that they can be used in the generative curious learner pre trainer model.

        :param example: Is the output of DataGenerator().generate_batch_of() functions output
        :return: The result after the concatenation of input and output string in the same dict
        """
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
            add_bos_and_eos: bool,
            max_sequence_length: int | None,
            task_generator_index: int | None = None,
            generator_index: int | None = None,
            identifier: int | None = None
    ) -> list[dict]:
        """This function will concat the inputStr and outputStr and create a sentence on which the generative
        Curious learner model is to be trained, It will also include the input_token_count to the dictionary
        which is to be used as a dynamic value during training for the input.

        :param count: The number of example that is needed.
        :param add_bos_and_eos: Flag for weather to add BOS and EOS to the Token list.
        :param max_sequence_length: Max length until which padding will be added. If None then no padding.
        :param task_generator_index: It can be between 0-3, which indicated the task-type generator index
        :param generator_index: This is the example function generator index,
               which we can be between 0-get_length_of_sample_generators(self)
        :param identifier: This is the example of the function, it can be any number,
               but for a specific number same example will be retrieved each time.
        :return: The samples with the taskType, sentence, io_parser_output & input_token_count in a map.
        """
        data_loader_output = self.data_generator.generate_batch_of(
            count,
            task_generator_index,
            generator_index,
            identifier,
        )
        for data in data_loader_output:
            self.create_sentence_using_input_and_output(data)
            data['io_parser_output'] = BatchBuilder.get_sentence_io_parser_output(
                data['sentence'],
                add_bos_and_eos,
                max_sequence_length,
            )
            for io_parser_item in data['io_parser_output']:
                token: any = io_parser_item[Constants.TOKEN]
                position: int = io_parser_item[Constants.POSITION]
                if token in self.catalyst_list:
                    data['input_token_count'] = position
        return data_loader_output


class DataLoaderTest(unittest.TestCase):

    def test_create_sentence_using_input_and_output(self):
        exampl_dict = {
            "inputStr": "3988 / 2482",
            "outputStr": "The result of dividing 3988 by 2482 is ##division(3988,2482)",
            "taskType": "nl_to_nl_translation",
        }
        data_loader = DataLoader()
        result = data_loader.create_sentence_using_input_and_output(exampl_dict)
        self.assertEqual(result, '3988 / 2482 = The result of dividing 3988 by 2482 is ##division(3988,2482)')

    def test_create_data_loader_output(self):
        data_loader = DataLoader()
        result = data_loader.create_data_loader_output(
            count=1,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_index=2,
            generator_index=2,
            identifier=3
        )
        print(result)
        expected_result = [
            {
                "inputStr": "##multiplication(107.23600916441234,784.625535184504)",
                "outputStr": "107.23600916441234 times 784.625535184504 equals?",
                "taskType": "func_to_nl_translation",
                "sentence": "##multiplication(107.23600916441234,784.625535184504) = 107.23600916441234 times 784.625535184504 equals?",
                "io_parser_output": [
                    {
                        "token": "<BOS>",
                        "category": {
                            "type": "special",
                            "subType": "word",
                            "subSubType": "none"
                        },
                        "position": 0
                    },
                    {
                        "token": "<function MathFunctions.multiplication at 0x1168b3880>",
                        "category": {
                            "type": "function",
                            "subType": "float",
                            "subSubType": "execute"
                        },
                        "position": 1
                    },
                    {
                        "token": 107.23600916441234,
                        "category": {
                            "type": "float",
                            "subType": "default",
                            "subSubType": "param_one"
                        },
                        "position": 2
                    },
                    {
                        "token": 784.625535184504,
                        "category": {
                            "type": "float",
                            "subType": "default",
                            "subSubType": "param_last"
                        },
                        "position": 3
                    },
                    {
                        "token": "=",
                        "category": {
                            "type": "word",
                            "subType": "default",
                            "subSubType": "none"
                        },
                        "position": 4
                    },
                    {
                        "token": 107.23600916441234,
                        "category": {
                            "type": "float",
                            "subType": "default",
                            "subSubType": "none"
                        },
                        "position": 5
                    },
                    {
                        "token": "times",
                        "category": {
                            "type": "word",
                            "subType": "default",
                            "subSubType": "none"
                        },
                        "position": 6
                    },
                    {
                        "token": 784.625535184504,
                        "category": {
                            "type": "float",
                            "subType": "default",
                            "subSubType": "none"
                        },
                        "position": 7
                    },
                    {
                        "token": "equals?",
                        "category": {
                            "type": "word",
                            "subType": "default",
                            "subSubType": "none"
                        },
                        "position": 8
                    },
                    {
                        "token": "<EOS>",
                        "category": {
                            "type": "special",
                            "subType": "word",
                            "subSubType": "none"
                        },
                        "position": 9
                    },
                    {
                        "token": "<PAD>",
                        "category": {
                            "type": "special",
                            "subType": "word",
                            "subSubType": "none"
                        },
                        "position": 10
                    },
                    {
                        "token": "<PAD>",
                        "category": {
                            "type": "special",
                            "subType": "word",
                            "subSubType": "none"
                        },
                        "position": 11
                    },
                    {
                        "token": "<PAD>",
                        "category": {
                            "type": "special",
                            "subType": "word",
                            "subSubType": "none"
                        },
                        "position": 12
                    },
                    {
                        "token": "<PAD>",
                        "category": {
                            "type": "special",
                            "subType": "word",
                            "subSubType": "none"
                        },
                        "position": 13
                    },
                    {
                        "token": "<PAD>",
                        "category": {
                            "type": "special",
                            "subType": "word",
                            "subSubType": "none"
                        },
                        "position": 14
                    },
                    {
                        "token": "<PAD>",
                        "category": {
                            "type": "special",
                            "subType": "word",
                            "subSubType": "none"
                        },
                        "position": 15
                    }
                ],
                "input_token_count": 4
            }
        ]
        self.assertEqual(result[0]['input_token_count'], expected_result[0]['input_token_count'])


if __name__ == "__main__":
    unittest.main()
