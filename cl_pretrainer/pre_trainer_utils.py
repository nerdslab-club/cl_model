import torch
from torch import Tensor

from cl_data.src.constants import Constants


class PreTrainerUtils:
    @staticmethod
    def create_function_param_token_infos(
            x: Tensor,
            batch_function_param_mask: Tensor,
            batch_encoder_hidden_states: list[list[Tensor]],
    ) -> list[dict]:
        """
        convert batch function param mask and batch encoder hidden states into funtion param token infos. ie
         {
            "start": (r,c),
            "end": (r,c),
            "encoder_hidden_state": Tensor, # Embeddings of the function in question found using initial function encoder
            "token": Tensor, # Embedding of category map token of function params.
        };
        :param x: Tensor containing the output of the previous decoder block. Shape: (N, S, E)
        :param batch_function_param_mask: ie. [[False, True, Ture, False], [False, True, Ture, False]]
        :param batch_encoder_hidden_states: list[Tensor, Tensor, ...]
        :return: funtion param token infos.
        """
        batch_size = batch_function_param_mask.size(0)
        sequence_length = batch_function_param_mask.size(1)

        result = []
        current_map = {}
        current_tensor = None
        current_encoder_hidden_state_index = 0
        # Iterate through the batch_function_param_mask tensor
        for i in range(batch_size):
            for j in range(sequence_length):
                if batch_function_param_mask[i][j]:
                    if current_tensor is None:
                        current_map["start"] = (i, j)
                        current_map["encoder_hidden_state"] = batch_encoder_hidden_states[i][current_encoder_hidden_state_index]
                        current_tensor = x[i][j].unsqueeze(0)
                    else:
                        current_tensor = torch.cat((current_tensor, x[i][j].unsqueeze(0)), dim=0)
                elif current_tensor is not None:
                    current_encoder_hidden_state_index += 1
                    current_map["end"] = (i, j)
                    current_map["token_tensors"] = current_tensor
                    result.append(current_map)
                    current_tensor = None
                    current_map = {}

            current_encoder_hidden_state_index = 0
            if current_tensor is not None:
                current_map["end"] = (i, sequence_length)
                current_map["token_tensors"] = current_tensor
                result.append(current_map)
                current_tensor = None
                current_map = {}

        return result

    @staticmethod
    def convert_category_map_into_io_parser_output_without_token(
            batch_category_map: list[list[dict[str, str]]],
    ) -> list[list[dict[str, any]]]:
        """
        Convert batch of category map into batch of io parser output without the token.
        :param batch_category_map: batch of category map
        :return: batch io parser output without the token
        """
        batch_io_parser_output_without_token = []
        for category_maps in batch_category_map:
            sequence_io_parser_output_without_token = []
            for category_map in category_maps:
                io_parser_output_without_token = {
                    Constants.CATEGORY: category_map,
                }
                sequence_io_parser_output_without_token.append(io_parser_output_without_token)
            batch_io_parser_output_without_token.append(sequence_io_parser_output_without_token)
        return batch_io_parser_output_without_token


if __name__ == "__main__":
    embedding_length = 4

    # Create random tensors for x and mask_tensor
    mask_tensor = torch.tensor([[False, False, True, True, True], [False, False, False, True, False]])
    x = torch.randn(mask_tensor.size(0), mask_tensor.size(1), embedding_length, dtype=torch.float32)

    # Create a batch of encoder hidden states
    batch_encoder_hidden_states = [[torch.randn(10, embedding_length)], [torch.randn(10, embedding_length)]]

    output = PreTrainerUtils.create_function_param_token_infos(x, mask_tensor, batch_encoder_hidden_states)
    print(output)
