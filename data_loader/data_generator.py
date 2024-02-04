import random
import unittest

from cl_data.func_func.f2f_sample_generator import F2FSamples
from cl_data.func_nl.f2n_sample_generator import F2NSamples
from cl_data.nl_func.n2f_sample_generator import N2FSamples
from cl_data.nlf_nlf.nlf2nlf_sample_generator import Nl2NlSamples
from cl_data.src.random_value_generator import RandomValueGenerator


class DataGenerator:
    def __init__(self):
        self.available_generator: list = [
            Nl2NlSamples(),
            F2FSamples(),
            F2NSamples(),
            N2FSamples()
        ]

    def generate_batch_of(
            self,
            batch_size: int,
            number_of_batch: int,
            task_generator_indexes: list[int] | None = None,
            generator_indexes: list[int] | None = None,
            identifier: int | None = None,
            shuffle: bool = False,
            seed: int = 42,
    ) -> list[dict]:
        """
        This function generates sample for training either randomly or in a paginated way if task_generator_index,
        generator_index and identifier is provided

        :param seed: seed for generating random indexes
        :param shuffle: shuffle the list before return is true.
        :param batch_size: Size of the batch
        :param number_of_batch: The number of batch we need.
        count = batch_size * number_of_batch
        and number_of_batch = len(task_generator_indexes) * len(generator_indexes)
        :param task_generator_indexes: It can be between 0-3, which indicated the task-type generator index
        :param generator_indexes: This is the example function generator index,
               which we can be between 0-get_length_of_sample_generators(self), better if consecutive
        :param identifier: This is the example of the function, it can be any number,
               but for a specific number same example will be retrieved each time.
        :return: The sample with the task type in a map.
        """
        if task_generator_indexes is not None and generator_indexes is not None:
            assert all(0 <= x <= 3 for x in task_generator_indexes), "task_generator_indexes values should be between 0 and 3."
            generator_indexes = [item for item in generator_indexes for _ in range(len(task_generator_indexes))]

        samples = []
        # Adding same generator index len(task_generator_indexes) times,
        # so that each generator index is used for each task
        for i in range(number_of_batch):
            if task_generator_indexes is not None:
                random_index = task_generator_indexes[i % len(task_generator_indexes)]
            else:
                random_index = RandomValueGenerator.generate_random_integer(0, 4)

            current_task_sample_generator = self.available_generator[random_index]
            current_generator_index = None
            if task_generator_indexes is not None and generator_indexes is not None:
                current_generator_index = generator_indexes[i % len(
                    generator_indexes)] % current_task_sample_generator.get_length_of_sample_generators()
            samples_from_this_generator = current_task_sample_generator.get_next_random_sample(
                batch_size=batch_size,
                generator_index=current_generator_index,
                identifier=identifier,
            )
            samples.extend(samples_from_this_generator)

        if shuffle:
            random.seed(seed)
            random.shuffle(samples)
        return samples


class DataGeneratorTest(unittest.TestCase):

    def test_generate_batch_of_for_random_samples(self):
        data_generator = DataGenerator()
        generated_batch = data_generator.generate_batch_of(4, 3)
        print(generated_batch)
        self.assertEqual(len(generated_batch), 12)

    def test_generate_batch_of_for_identifiable_samples(self):
        data_generator = DataGenerator()
        # number_of_batch = len(task_generator_indexes) * len(generator_indexes) but not a hard constraint
        generated_batch = data_generator.generate_batch_of(
            batch_size=4,
            number_of_batch=6,
            task_generator_indexes=[0, 1, 2],
            generator_indexes=[0, 1],
            identifier=0,
            shuffle=True,
        )
        print(generated_batch)
        self.assertEqual(len(generated_batch), 24)


if __name__ == "__main__":
    unittest.main()
