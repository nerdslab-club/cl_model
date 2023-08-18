import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

from cl_data.src.constants import (
    PretrainTasks,
    TaskTypes,
    CategoryType,
    CategorySubType,
    CategorySubSubType,
)


class CategoryAndTaskEncoder:
    def __init__(
        self,
        category_map: dict,
        task_type: PretrainTasks | TaskTypes,
        desired_length=768,
        sample_rate=1024,
    ):
        self.desired_length = desired_length
        self.sample_rate = sample_rate

        # Frequencies and amplitudes
        self.base_frequencies = [100, 200, 300, 400]
        self.amplitudes = [1.0, 0.8, 0.6, 0.4]

        # Modulations
        self.modulations = [5, 9, 15, 21, 28, 36, 45, 55, 66, 78, 91]

        # Initialize signals
        self.signals = []
        for freq, amp in zip(self.base_frequencies, self.amplitudes):
            signal = amp * np.sin(
                2 * np.pi * freq * np.linspace(0, 1, self.desired_length)
            )
            self.signals.append(signal)

        # Modulate the signals based on category_map and task_type
        self.modulate_signals(category_map, task_type)

        # Combine the modulated signals
        self.combined_signal = np.sum(self.signals, axis=0)
        self.combined_signal /= np.max(np.abs(self.combined_signal))

    def categorical_encoding(self, x):
        pass

    def forward(self, x):
        pass

    def get_category_signal(self, category: CategoryType) -> Tensor:
        modulation_freq = self.__get_category_modulation_frequency(category)

        combined_freq = self.base_frequencies[0] + modulation_freq
        # Check if the sampling rate meets Nyquist-Shannon sampling theorem
        if self.sample_rate < 2 * combined_freq:
            raise ValueError("Sampling rate is too low to capture the signal.")

        # Generate time values
        t = torch.arange(0, self.desired_length) / self.sample_rate

        # Generate modulated sine signal
        signal = self.amplitudes[0] * torch.sin(2 * np.pi * combined_freq * t)
        return signal

    def __get_sub_category_signal(self, sub_category: CategorySubType) -> Tensor:
        modulation_freq = self.__get_sub_category_modulation_frequency(sub_category)

        combined_freq = self.base_frequencies[1] + modulation_freq
        # Check if the sampling rate meets Nyquist-Shannon sampling theorem
        if self.sample_rate < 2 * combined_freq:
            raise ValueError("Sampling rate is too low to capture the signal.")

        # Generate time values
        t = torch.arange(0, self.desired_length) / self.sample_rate

        # Generate modulated sine signal
        signal = self.amplitudes[1] * torch.cos(2 * np.pi * combined_freq * t)
        return signal

    def __get_sub_sub_category_signal(
        self, sub_sub_category: CategorySubSubType
    ) -> Tensor:
        modulation_freq = self.__get_sub_sub_category_modulation_frequency(
            sub_sub_category
        )

        combined_freq = self.base_frequencies[2] + modulation_freq
        # Check if the sampling rate meets Nyquist-Shannon sampling theorem
        if self.sample_rate < 2 * combined_freq:
            raise ValueError("Sampling rate is too low to capture the signal.")

        # Generate time values
        t = torch.arange(0, self.desired_length) / self.sample_rate

        # Generate modulated sine signal
        signal = self.amplitudes[2] * torch.sin(2 * np.pi * combined_freq * t)
        return signal

    def __get_task_signal(self, task_type: TaskTypes | PretrainTasks) -> Tensor:
        modulation_freq = self.__get_task_modulation_frequency(task_type)

        combined_freq = self.base_frequencies[3] + modulation_freq
        # Check if the sampling rate meets Nyquist-Shannon sampling theorem
        if self.sample_rate < 2 * combined_freq:
            raise ValueError("Sampling rate is too low to capture the signal.")

        # Generate time values
        t = torch.arange(0, self.desired_length) / self.sample_rate

        # Generate modulated sine signal
        signal = self.amplitudes[3] * torch.cos(2 * np.pi * combined_freq * t)
        return signal

    def __get_category_modulation_frequency(self, category: CategoryType):
        if category == CategoryType.FUNCTION:
            return self.modulations[0]
        elif category == CategoryType.WORD:
            return self.modulations[1]
        elif category == CategoryType.INTEGER:
            return self.modulations[2]
        elif category == CategoryType.FLOAT:
            return self.modulations[3]
        elif category == CategoryType.LIST:
            return self.modulations[4]
        elif category == CategoryType.BOOL:
            return self.modulations[5]
        elif category == CategoryType.SPECIAL:
            return self.modulations[6]

    def __get_sub_category_modulation_frequency(self, sub_category: CategorySubType):
        if sub_category == CategorySubType.DEFAULT:
            return self.modulations[0]
        elif sub_category == CategorySubType.PLACEHOLDER:
            return self.modulations[1]
        elif sub_category == CategorySubType.RETURN_VALUE:
            return self.modulations[2]
        elif sub_category == CategorySubType.WORD:
            return self.modulations[3]
        elif sub_category == CategorySubType.INTEGER:
            return self.modulations[4]
        elif sub_category == CategorySubType.FLOAT:
            return self.modulations[5]
        elif sub_category == CategorySubType.LIST:
            return self.modulations[6]
        elif sub_category == CategorySubType.BOOL:
            return self.modulations[7]

    def __get_sub_sub_category_modulation_frequency(
        self, sub_sub_category: CategorySubSubType
    ):
        if sub_sub_category == CategorySubSubType.PARAM_ONE:
            return self.modulations[0]
        elif sub_sub_category == CategorySubSubType.PARAM_TWO:
            return self.modulations[1]
        elif sub_sub_category == CategorySubSubType.PARAM_THREE:
            return self.modulations[2]
        elif sub_sub_category == CategorySubSubType.PARAM_FOUR:
            return self.modulations[3]
        elif sub_sub_category == CategorySubSubType.PARAM_FIVE:
            return self.modulations[4]
        elif sub_sub_category == CategorySubSubType.PARAM_LAST:
            return self.modulations[5]
        elif sub_sub_category == CategorySubSubType.NONE:
            return self.modulations[6]
        elif sub_sub_category == CategorySubSubType.PLACEHOLDER:
            return self.modulations[7]
        elif sub_sub_category == CategorySubSubType.EXECUTE:
            return self.modulations[8]
        elif sub_sub_category == CategorySubSubType.REPRESENT:
            return self.modulations[9]

    def __get_task_modulation_frequency(self, task: TaskTypes | PretrainTasks):
        if task == TaskTypes.FUNC_TO_FUNC_TRANSLATION:
            return self.modulations[0]
        elif task == TaskTypes.FUNC_TO_NL_TRANSLATION:
            return self.modulations[1]
        elif task == TaskTypes.NL_TO_FUNC_TRANSLATION:
            return self.modulations[2]
        elif task == TaskTypes.NL_TO_NL_TRANSLATION:
            return self.modulations[3]
        elif task == PretrainTasks.MASKED_TOKEN_PREDICTION:
            return self.modulations[4]
        elif task == PretrainTasks.NEXT_TOKEN_PREDICTION:
            return self.modulations[5]

    def modulate_signals(self, category_map, task_type):
        pass
        # Apply modulations based on category_map and task_type
        # for category, modulation in zip([CategoryType, CategorySubType, CategorySubSubType, TaskType],
        #                                 self.modulations):
        #     enum_value = category_map.get(category.name, 0)
        #     modulation_signal = np.sin(2 * np.pi * modulation * np.linspace(0, 1, self.desired_length))
        #     self.signals[enum_value] += modulation_signal

    def plot_combined_signal(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, self.desired_length), self.combined_signal)
        plt.title("Combined Modulated Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    @staticmethod
    def plot_provider_signal(desired_length: int, provided_signal: list):
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, desired_length), provided_signal)
        plt.title("Provider Modulated Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    @staticmethod
    def plot_provider_signal_fft(sample_rate: int, provider_signal: Tensor):
        signal_np = provider_signal.numpy()
        fft_result = np.fft.fft(signal_np)
        # Calculate the corresponding frequencies
        frequencies = np.fft.fftfreq(signal_np.size, d=1 / sample_rate)
        # Plot the FFT magnitude
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies, np.abs(fft_result))
        plt.title("FFT Magnitude")
        plt.xlabel("Frequency")
        plt.ylabel("Magnitude")
        plt.grid()
        plt.show()

    @staticmethod
    def print_combined_signal(provider_signal):
        print(provider_signal)


if __name__ == "__main__":
    category_and_task_encoder = CategoryAndTaskEncoder(
        {}, TaskTypes.NL_TO_FUNC_TRANSLATION
    )
    signal = category_and_task_encoder.get_category_signal(
        CategoryType.FUNCTION,
    )
    CategoryAndTaskEncoder.plot_provider_signal(768, signal)
    CategoryAndTaskEncoder.plot_provider_signal_fft(1024, signal)
