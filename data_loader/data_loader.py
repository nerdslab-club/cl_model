import random
import unittest
from sklearn.model_selection import KFold
from cl_data.io_parser.io_parser_utility import split_string_custom
from cl_data.src.constants import Constants
from cl_data.src.random_value_generator import RandomValueGenerator
from cl_pretrainer.batch_builder import BatchBuilder
from data_loader.data_generator import DataGenerator


class DataLoader:

    TRAIN = "train_dataset"
    VALIDATION = "validation_dataset"
    TEST = "test_dataset"

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
        example[Constants.SENTENCE] = combined_str
        return combined_str

    def create_data_loader_output(
            self,
            batch_size: int,
            number_of_batch: int,
            param_variation: int = 1,
            add_bos_and_eos: bool = True,
            max_sequence_length: int | None = 24,
            task_generator_indexes: list[int] | None = None,
            generator_indexes: list[int] | None = None,
            identifier: int | None = None,
            shuffle: bool = False,
            seed: int = 42,
    ) -> list[dict]:
        """This function will concat the inputStr and outputStr and create a sentence on which the generative
        Curious learner model is to be trained, It will also include the input_token_count to the dictionary
        which is to be used as a dynamic value during training for the input.

        :param param_variation: for range in param_variation new samples will be added.
        :param seed: seed for generating random indexes
        :param shuffle: shuffle the list before return is true.
        :param batch_size: Size of the batch
        :param number_of_batch: The number of batch we need. batch_size * number_of_batch == count
        :param add_bos_and_eos: Flag for weather to add BOS and EOS to the Token list.
        :param max_sequence_length: Max length until which padding will be added. If None then no padding.
        :param task_generator_indexes: It can be between 0-3, which indicated the task-type generator index
        :param generator_indexes: This is the example function generator index,
               which we can be between 0-get_length_of_sample_generators(self)
        :param identifier: This is the example of the function, it can be any number,
               but for a specific number same example will be retrieved each time.
        :return: The samples with the taskType, sentence, io_parser_output & input_token_count in a map.
        """
        data_loader_output = self.data_generator.generate_batch_of(
            batch_size=batch_size,
            number_of_batch=number_of_batch,
            task_generator_indexes=task_generator_indexes,
            generator_indexes=generator_indexes,
            identifier=identifier,
            shuffle=shuffle,
            seed=seed,
            param_variation=param_variation,
        )
        for data in data_loader_output:
            self.create_sentence_using_input_and_output(data)
            data[Constants.IO_PARSER_OUTPUT] = BatchBuilder.get_sentence_io_parser_output(
                data[Constants.SENTENCE],
                add_bos_and_eos,
                max_sequence_length,
            )
            data[Constants.INPUT_TOKEN_COUNT] = len(split_string_custom(data['inputStr'])) + 1
            for io_parser_item in data[Constants.IO_PARSER_OUTPUT]:
                token: any = io_parser_item[Constants.TOKEN]
                position: int = io_parser_item[Constants.POSITION]
                if token in self.catalyst_list:
                    data[Constants.INITIAL_TOKEN_COUNT] = position + 1
        return data_loader_output

    def create_test_train_dataset(
            self,
            batch_size: int,
            number_of_batch: int,
            param_variation: int = 1,
            add_bos_and_eos: bool = True,
            max_sequence_length: int | None = 24,
            task_generator_indexes: list[int] | None = None,
            generator_indexes: list[int] | None = None,
            identifier: int | None = None,
            shuffle: bool = False,
            seed: int = 42,
            train_ratio: float = 0.9,
    ) -> dict[str, list[dict]]:
        data = self.create_data_loader_output(
            batch_size=batch_size,
            number_of_batch=number_of_batch,
            param_variation=param_variation,
            add_bos_and_eos=add_bos_and_eos,
            max_sequence_length=max_sequence_length,
            task_generator_indexes=task_generator_indexes,
            generator_indexes=generator_indexes,
            identifier=identifier,
            shuffle=False,
            seed=seed,
        )
        # Calculate the split index
        split_index = int(len(data) * train_ratio)

        # Split the data
        train_data = data[:split_index]
        test_data = data[split_index:]
        if shuffle:
            random.seed(seed)
            random.shuffle(train_data)
            random.shuffle(test_data)
        return {DataLoader.TRAIN: train_data, DataLoader.TEST: test_data}

    @staticmethod
    def create_k_fold_validation_dataset(
            train_data: list[dict],
            k: int = 5,
    ):
        kf = KFold(n_splits=k, shuffle=True)
        fold_indices = kf.split(train_data)

        fold_data = []
        for train_index, val_index in fold_indices:
            train_fold = [train_data[i] for i in train_index]
            val_fold = [train_data[i] for i in val_index]
            fold_data.append({DataLoader.TRAIN: train_fold, DataLoader.VALIDATION: val_fold})

        return fold_data

    @staticmethod
    def split_train_validation_test(data: list[dict], shuffle: bool, seed: int) -> dict[str, list[dict]]:
        train_split_index = int(len(data) * 0.8)
        validation_split_index = int(len(data) * 0.9)

        train_data = data[:train_split_index]
        validation_data = data[train_split_index: validation_split_index]
        test_data = data[validation_split_index:]
        if shuffle:
            random.seed(seed)
            random.shuffle(train_data)
            random.shuffle(test_data)
            random.shuffle(validation_data)
        return {DataLoader.TRAIN: train_data, DataLoader.TEST: test_data, DataLoader.VALIDATION: validation_data}


class DataLoaderTest(unittest.TestCase):
    def test_create_k_fold_validation_dataset(self):
        data_loader = DataLoader()
        result = data_loader.create_test_train_dataset(
            batch_size=4,
            number_of_batch=40,
            param_variation=100,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[0, 1, 2, 3],
            generator_indexes=[i for i in range(10)],
            identifier=0,
            shuffle=False,
            seed=42
        )
        self.assertEqual(len(result[DataLoader.TRAIN]), 16000 - 1600)
        k_fold_data = DataLoader.create_k_fold_validation_dataset(result[DataLoader.TRAIN], 10)
        self.assertEqual(len(k_fold_data), 10)
        for i, fold in enumerate(k_fold_data):
            self.assertEqual(len(fold[DataLoader.TRAIN]), 16000-1600-1440)
            self.assertEqual(len(fold[DataLoader.VALIDATION]), 1440)

    def test_create_test_train_dataset(self):
        data_loader = DataLoader()
        result = data_loader.create_test_train_dataset(
            batch_size=4,
            number_of_batch=40,
            param_variation=100,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[0, 1, 2, 3],
            generator_indexes=[i for i in range(10)],
            identifier=0,
            shuffle=False,
            seed=42
        )
        self.assertEqual(len(result[DataLoader.TRAIN]), 16000 - 1600)
        self.assertEqual(len(result[DataLoader.TEST]), 1600)

    def test_same_dataset_without_shaffle_for_create_test_train_dataset(self):
        data_loader = DataLoader()
        result = data_loader.create_test_train_dataset(
            batch_size=4,
            number_of_batch=40,
            param_variation=10,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[0, 1, 2, 3],
            generator_indexes=[i for i in range(10)],
            identifier=0,
            shuffle=False,
            seed=42
        )
        result_one = data_loader.create_test_train_dataset(
            batch_size=4,
            number_of_batch=40,
            param_variation=10,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[0, 1, 2, 3],
            generator_indexes=[i for i in range(10)],
            identifier=0,
            shuffle=False,
            seed=42
        )
        self.assertEqual(result[DataLoader.TRAIN], result_one[DataLoader.TRAIN])
        self.assertEqual(result[DataLoader.TEST], result_one[DataLoader.TEST])

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
            batch_size=1,
            number_of_batch=1,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[2],
            generator_indexes=[2],
            identifier=3,
            shuffle=True,
        )
        print(result)
        expected_result = [
            {
                "inputStr": "##multiplication(107.23600916441234,784.625535184504)",
                "outputStr": "107.23600916441234 times 784.625535184504 equals?",
                "taskType": "func_to_nl_translation",
                "sentence": "##multiplication(107.23600916441234,784.625535184504) = 107.23600916441234 times 784.625535184504 equals?",
                "inputTokenCount": 2
            }
        ]
        self.assertEqual(result[0][Constants.INPUT_TOKEN_COUNT], expected_result[0][Constants.INPUT_TOKEN_COUNT])

    def test_create_data_loader_output_all(self):
        data_loader = DataLoader()
        result = data_loader.create_data_loader_output(
            batch_size=4,
            number_of_batch=40,
            param_variation=100,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[0, 1, 2, 3],
            generator_indexes=[i for i in range(10)],
            identifier=0,
            shuffle=False,
            seed=42
        )
        print(result)
        self.assertEqual(len(result), 16000)

    def test_same_data_without_shaffle(self):
        data_loader = DataLoader()
        result = data_loader.create_data_loader_output(
            batch_size=4,
            number_of_batch=40,
            param_variation=4,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[0, 1, 2, 3],
            generator_indexes=[i for i in range(10)],
            identifier=0,
            shuffle=False,
            seed=42
        )
        result_one = data_loader.create_data_loader_output(
            batch_size=4,
            number_of_batch=40,
            param_variation=4,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[0, 1, 2, 3],
            generator_indexes=[i for i in range(10)],
            identifier=0,
            shuffle=False,
            seed=42
        )
        self.assertEqual(len(result), len(result_one))
        self.assertEqual(result, result_one)

    def test_create_data_loader_output_shuffle(self):
        data_loader = DataLoader()
        result = data_loader.create_data_loader_output(
            batch_size=2,
            number_of_batch=8,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[0, 1, 2, 3],
            generator_indexes=[0, 1],
            identifier=0,
            shuffle=True,
            seed=42
        )
        print(f'Result one: \n {result}')
        result_one = data_loader.create_data_loader_output(
            batch_size=2,
            number_of_batch=8,
            add_bos_and_eos=True,
            max_sequence_length=16,
            task_generator_indexes=[0, 1, 2, 3],
            generator_indexes=[0, 1],
            identifier=0,
            shuffle=True,
            seed=42
        )
        print(f'Result two: \n {result_one}')
        self.assertEqual(result, result_one)


if __name__ == "__main__":
    unittest.main()
