import numpy as np
import matplotlib.pyplot as plt

from cl_data.src.constants import PretrainTasks, TaskTypes, CategoryType, CategorySubType, CategorySubSubType


class CategoryAndTaskEncoder:
    def __init__(self, category_map: dict, task_type: PretrainTasks | TaskTypes, desired_length=768, sample_rate=1024):
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
            signal = amp * np.sin(2 * np.pi * freq * np.linspace(0, 1, self.desired_length))
            self.signals.append(signal)

        # Modulate the signals based on category_map and task_type
        self.modulate_signals(category_map, task_type)

        # Combine the modulated signals
        self.combined_signal = np.sum(self.signals, axis=0)
        self.combined_signal /= np.max(np.abs(self.combined_signal))

    def categorical_encoding(self, x):
        pass

    def __get_category_signal(self, category: CategoryType):
        pass

    def __get_sub_category_signal(self, sub_category: CategorySubType):
        pass

    def _get_sub_sub_category_signal(self, sub_sub_category: CategorySubSubType):
        pass

    def forward(self, x):
        pass

    def modulate_signals(self, category_map, task_type):
        # Apply modulations based on category_map and task_type
        for category, modulation in zip([CategoryType, CategorySubType, CategorySubSubType, TaskType],
                                        self.modulations):
            enum_value = category_map.get(category.name, 0)
            modulation_signal = np.sin(2 * np.pi * modulation * np.linspace(0, 1, self.desired_length))
            self.signals[enum_value] += modulation_signal

    def print_combined_signal(self):
        print(self.combined_signal)

    def plot_combined_signal(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, self.desired_length), self.combined_signal)
        plt.title('Combined Modulated Signal')
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.show()
