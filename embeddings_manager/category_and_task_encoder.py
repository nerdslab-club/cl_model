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
    Constants,
)


class CategoryAndTaskEncoder:
    def __init__(
        self,
        desired_length=768,
        sample_rate=1024,
        d_type=torch.float,
        device="cpu",
    ):
        self.desired_length = desired_length
        self.sample_rate = sample_rate

        # Frequencies and amplitudes
        self.base_frequencies = [100, 200, 300, 400]
        self.amplitudes = [1.0, 0.8, 0.6, 0.4]

        # Modulations
        self.modulations = [5, 9, 15, 21, 28, 36, 45, 55, 66, 78, 91]

        self.d_type = d_type
        self.device = device

        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


    def categorical_encoding(
        self,
        category_map: dict,
        task_type: str,
    ):
        category_signal = self.__get_category_signal(
            category_map[Constants.CATEGORY_TYPE],
        ).to(self.device)
        sub_category_signal = self.__get_sub_category_signal(
            category_map[Constants.CATEGORY_SUB_TYPE],
        ).to(self.device)
        sub_sub_category_signal = self.__get_sub_sub_category_signal(
            category_map[Constants.CATEGORY_SUB_SUB_TYPE],
        ).to(self.device)
        task_signal = self.__get_task_signal(task_type).to(self.device)
        combined_signal = (
            category_signal
            + sub_category_signal
            + sub_sub_category_signal
            + task_signal
        )

        # Normalize the tensor by dividing by its maximum absolute value
        combined_signal /= torch.max(torch.abs(combined_signal))
        return combined_signal

    def forward(self, token_embedding: Tensor, category_map: dict, task_type: str):
        categorical_embedding = self.categorical_encoding(category_map, task_type)
        token_embedding = (
            token_embedding + categorical_embedding[: token_embedding.size(0)]
        )
        return token_embedding

    def __get_category_signal(
        self,
        category: str,
    ) -> Tensor:
        modulation_freq = self.__get_category_modulation_frequency(category)

        combined_freq = self.base_frequencies[0] + modulation_freq
        # Check if the sampling rate meets Nyquist-Shannon sampling theorem
        if self.sample_rate < 2 * combined_freq:
            raise ValueError("Sampling rate is too low to capture the signal.")

        # Generate time values
        t = (
            torch.arange(0, self.desired_length, dtype=self.d_type, device=self.device)
            / self.sample_rate
        )

        # Generate modulated sine signal
        signal = self.amplitudes[0] * torch.sin(2 * np.pi * combined_freq * t)
        return signal

    def __get_sub_category_signal(self, sub_category: str) -> Tensor:
        modulation_freq = self.__get_sub_category_modulation_frequency(sub_category)

        combined_freq = self.base_frequencies[1] + modulation_freq
        # Check if the sampling rate meets Nyquist-Shannon sampling theorem
        if self.sample_rate < 2 * combined_freq:
            raise ValueError("Sampling rate is too low to capture the signal.")

        # Generate time values
        t = (
            torch.arange(0, self.desired_length, dtype=self.d_type, device=self.device)
            / self.sample_rate
        )

        # Generate modulated sine signal
        signal = self.amplitudes[1] * torch.cos(2 * np.pi * combined_freq * t)
        return signal

    def __get_sub_sub_category_signal(self, sub_sub_category: str) -> Tensor:
        modulation_freq = self.__get_sub_sub_category_modulation_frequency(
            sub_sub_category
        )

        combined_freq = self.base_frequencies[2] + modulation_freq
        # Check if the sampling rate meets Nyquist-Shannon sampling theorem
        if self.sample_rate < 2 * combined_freq:
            raise ValueError("Sampling rate is too low to capture the signal.")

        # Generate time values
        t = (
            torch.arange(0, self.desired_length, dtype=self.d_type, device=self.device)
            / self.sample_rate
        )

        # Generate modulated sine signal
        signal = self.amplitudes[2] * torch.sin(2 * np.pi * combined_freq * t)
        return signal

    def __get_task_signal(self, task_type: str) -> Tensor:
        modulation_freq = self.__get_task_modulation_frequency(task_type)

        combined_freq = self.base_frequencies[3] + modulation_freq
        # Check if the sampling rate meets Nyquist-Shannon sampling theorem
        if self.sample_rate < 2 * combined_freq:
            raise ValueError("Sampling rate is too low to capture the signal.")

        # Generate time values
        t = (
            torch.arange(0, self.desired_length, dtype=self.d_type, device=self.device)
            / self.sample_rate
        )

        # Generate modulated sine signal
        signal = self.amplitudes[3] * torch.cos(2 * np.pi * combined_freq * t)
        return signal

    def __get_category_modulation_frequency(self, category: str):
        if category == CategoryType.FUNCTION.value:
            return self.modulations[0]
        elif category == CategoryType.WORD.value:
            return self.modulations[1]
        elif category == CategoryType.INTEGER.value:
            return self.modulations[2]
        elif category == CategoryType.FLOAT.value:
            return self.modulations[3]
        elif category == CategoryType.LIST.value:
            return self.modulations[4]
        elif category == CategoryType.BOOL.value:
            return self.modulations[5]
        elif category == CategoryType.SPECIAL.value:
            return self.modulations[6]

    def __get_sub_category_modulation_frequency(self, sub_category: str):
        if sub_category == CategorySubType.DEFAULT.value:
            return self.modulations[0]
        elif sub_category == CategorySubType.PLACEHOLDER.value:
            return self.modulations[1]
        elif sub_category == CategorySubType.RETURN_VALUE.value:
            return self.modulations[2]
        elif sub_category == CategorySubType.WORD.value:
            return self.modulations[3]
        elif sub_category == CategorySubType.INTEGER.value:
            return self.modulations[4]
        elif sub_category == CategorySubType.FLOAT.value:
            return self.modulations[5]
        elif sub_category == CategorySubType.LIST.value:
            return self.modulations[6]
        elif sub_category == CategorySubType.BOOL.value:
            return self.modulations[7]

    def __get_sub_sub_category_modulation_frequency(self, sub_sub_category: str):
        if sub_sub_category == CategorySubSubType.PARAM_ONE.value:
            return self.modulations[0]
        elif sub_sub_category == CategorySubSubType.PARAM_TWO.value:
            return self.modulations[1]
        elif sub_sub_category == CategorySubSubType.PARAM_THREE.value:
            return self.modulations[2]
        elif sub_sub_category == CategorySubSubType.PARAM_FOUR.value:
            return self.modulations[3]
        elif sub_sub_category == CategorySubSubType.PARAM_FIVE.value:
            return self.modulations[4]
        elif sub_sub_category == CategorySubSubType.PARAM_LAST.value:
            return self.modulations[5]
        elif sub_sub_category == CategorySubSubType.NONE.value:
            return self.modulations[6]
        elif sub_sub_category == CategorySubSubType.PLACEHOLDER.value:
            return self.modulations[7]
        elif sub_sub_category == CategorySubSubType.EXECUTE.value:
            return self.modulations[8]
        elif sub_sub_category == CategorySubSubType.REPRESENT.value:
            return self.modulations[9]

    def __get_task_modulation_frequency(self, task: str):
        if task == TaskTypes.FUNC_TO_FUNC_TRANSLATION.value:
            return self.modulations[0]
        elif task == TaskTypes.FUNC_TO_NL_TRANSLATION.value:
            return self.modulations[1]
        elif task == TaskTypes.NL_TO_FUNC_TRANSLATION.value:
            return self.modulations[2]
        elif task == TaskTypes.NL_TO_NL_TRANSLATION.value:
            return self.modulations[3]
        elif task == PretrainTasks.MASKED_TOKEN_PREDICTION.value:
            return self.modulations[4]
        elif task == PretrainTasks.NEXT_TOKEN_PREDICTION.value:
            return self.modulations[5]

    @staticmethod
    def get_combined_embedding(
        token_embedding: Tensor, categorical_embedding: Tensor
    ) -> Tensor:
        # Moved to CPU or Cuda
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        token_embedding.to(device)
        categorical_embedding.to(device)

        return token_embedding + categorical_embedding[: token_embedding.size(0)]

    @staticmethod
    def plot_provider_signal(desired_length: int, provided_signal: Tensor):
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, desired_length), provided_signal)
        plt.title("Provider Modulated Signal")
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.grid()
        plt.show()

    @staticmethod
    def frequency_encoding(categorical_embedding: Tensor) -> Tensor:
        signal_np = categorical_embedding.numpy()
        fft_result = np.abs(np.fft.fft(signal_np))
        frequency_embedding = torch.tensor(fft_result)
        return frequency_embedding

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
    category_and_task_encoder = CategoryAndTaskEncoder()
    category_map = {"type": "function", "subType": "bool", "subSubType": "none"}
    task_type = TaskTypes.FUNC_TO_NL_TRANSLATION.value
    token_embedding = torch.rand(768)
    print(f"token embedding shape: {token_embedding.shape}")

    categorical_embedding = category_and_task_encoder.categorical_encoding(
        category_map, task_type
    )
    frequency_embedding = CategoryAndTaskEncoder.frequency_encoding(
        categorical_embedding
    )
    print(f"frequency embedding shape: {frequency_embedding.shape}")
    CategoryAndTaskEncoder.plot_provider_signal(768, frequency_embedding)

    combined_signal = category_and_task_encoder.forward(
        token_embedding, category_map, task_type
    )
    print(f"combined signal shape: {combined_signal.shape}")
    CategoryAndTaskEncoder.plot_provider_signal(768, combined_signal)

    # CategoryAndTaskEncoder.plot_provider_signal(768, frequency_embedding)
    CategoryAndTaskEncoder.plot_provider_signal_fft(1024, combined_signal)
