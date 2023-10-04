import math

import torch
from torch import Tensor


class ALiBiBiEncoder:

    @staticmethod
    def get_slopes(n_heads: int):
        n = 2 ** math.floor(math.log2(n_heads))
        m_0 = 2.0 ** (-8.0 / n)
        m = torch.pow(m_0, torch.arange(1, 1 + n, step=n / n_heads))
        return m

    @staticmethod
    def generate_reverse_distance_matrix(dimension: int, left_negative: True):
        matrix = torch.zeros((dimension, dimension), dtype=torch.float)
        multiplier = -1 if left_negative else 1

        for i in range(dimension):
            for j in range(dimension):
                if i == j:
                    matrix[i, j] = dimension
                elif i < j:
                    matrix[i, j] = dimension - abs(i - j)
                elif i > j:
                    matrix[i, j] = (dimension - abs(i - j)) * multiplier

        return matrix

    @staticmethod
    def generate_distance_matrix(dim, with_mask=False):
        matrix = torch.zeros(dim, dim)

        for i in range(dim):
            for j in range(dim):
                if i == j:
                    matrix[i, j] = 0
                elif i < j:
                    matrix[i, j] = 0 if with_mask else (j - i)
                else:
                    matrix[i, j] = i - j
        return matrix

    @staticmethod
    @torch.no_grad()
    def get_alibi_biases(
        batch_size: int,
        n_heads: int,
        sequence_length: int,
        with_mask=False,
        device="cpu",
    ) -> Tensor:
        """This bias will be subtracted from the real attention score
        No need to use positional information in cross-attention
        Link: https://github.com/ofirpress/attention_with_linear_biases/issues/5
        :param batch_size: Batch size that is used for training.
        :param n_heads: Number of head in multi head attention.
        :param sequence_length: Max supported sequence length in a sentence.
        :param with_mask: Hide future words with mask or not.
        :param device: device type.
        :return: Bias tensor which will be subtracted from attention score.
        """
        m = ALiBiBiEncoder.get_slopes(n_heads).to(device)
        distance_matrix = ALiBiBiEncoder.generate_distance_matrix(
            sequence_length, with_mask
        )

        # Multiply them pair-wise to get the AliBi bias matrix
        bias = distance_matrix[None, :, :] * m[:, None, None]

        # Broadcasting the last layer to match the batch_size
        bias_expanded_to_batch_size = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)

        return bias_expanded_to_batch_size


def _test_alibi():
    slopes = ALiBiBiEncoder.get_slopes(8)
    print(f"slopes list for MHA {slopes}")

    distance_matrix = ALiBiBiEncoder.generate_distance_matrix(5)
    print(f"distance matrix shape {distance_matrix.shape} \n {distance_matrix}")

    bias = ALiBiBiEncoder.get_alibi_biases(
        batch_size=2, n_heads=8, sequence_length=5, with_mask=False
    )
    print(f"bias shape {bias.shape} \n {bias}")


if __name__ == "__main__":
    _test_alibi()
