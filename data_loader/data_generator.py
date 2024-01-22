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
            count: int,
            task_generator_index: int | None,
            generator_index: int | None,
            identifier: int | None) -> list[dict]:
        """
        This function generate samples for training either randomly or in a paginated way if task_generator_index,
        generator_index and identifier is provided

        :param count: The number of sample that is asked for.
        :param task_generator_index: It can be between 0-3, which indicated the task-type generator index
        :param generator_index: This is the example function generator index,
               which we can be between 0-get_length_of_sample_generators(self)
        :param identifier: This is the example of the function, it can be any number,
               but for a specific number same example will be retrieved each time.
        :return: The sample with the task type in a map.
        """
        assert count % 4 == 0, "Count must be divisible by four..."
        samples = []
        for i in range(0, 4):
            if task_generator_index is not None:
                random_index = task_generator_index
            else:
                random_index = RandomValueGenerator.generate_random_integer(0, 4)
            current_task_sample_generator = self.available_generator[random_index]
            samples_from_this_generator = current_task_sample_generator.get_next_random_sample(
                count // 4,
                generator_index,
                identifier,
            )
            samples.extend(samples_from_this_generator)
        return samples


class DataGeneratorTest(unittest.TestCase):

    def test_generate_batch_of(self):
        data_generator = DataGenerator()
        generated_batch = data_generator.generate_batch_of(12);
        print(generated_batch)
        self.assertEqual(len(generated_batch), 12)


if __name__ == "__main__":
    unittest.main()
