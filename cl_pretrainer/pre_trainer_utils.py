import torch
from torch import Tensor

from cl_data.src.constants import Constants


class PreTrainerUtils:
    @staticmethod
    def create_function_param_token_infos(
            x: Tensor,
            batch_function_param_mask: Tensor,
            batch_encoder_hidden_states: list[list[Tensor]],
            shift_right=False,
    ) -> list[dict]:
        """
        convert batch function param mask and batch encoder hidden states into funtion param token infos. ie
         {
            "start": (r,c),
            "end": (r,c),
            "encoder_hidden_state": Tensor, # Embeddings of the function in question found using initial function encoder
            "token": Tensor, # Embedding of category map token of function params.
        };
        :param shift_right:
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
                    if j == 0:
                        raise Exception("create_function_param_token_infos, First token can't be a param token...")

                    if current_tensor is None:
                        current_map["start"] = (i, j)
                        current_map["encoder_hidden_state"] = batch_encoder_hidden_states[i][current_encoder_hidden_state_index]
                        current_tensor = x[i][j-1].unsqueeze(0) if shift_right else x[i][j].unsqueeze(0)
                    else:
                        current_tensor = torch.cat(
                            (current_tensor, x[i][j-1].unsqueeze(0) if shift_right else x[i][j].unsqueeze(0)),
                            dim=0,
                        )
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

    @staticmethod
    def create_tgt_tensor_for_output_classification_head(
            output_classification_head_index: int,
            tgt_batch_probability: list[list[tuple[int, int]]],
    ) -> Tensor:
        """
        Create tgt tensor probability by adding not my token to tokens that are not for current index
        Also convert the tgt batch probability to tensor.

        :param output_classification_head_index:
        :param tgt_batch_probability: Batch of list of tuple (classification head id, output vocab token id)
        :return: tgt output probability tensor for the given output classification head
        """
        batch_result = []
        for tgt_sequence_probability in tgt_batch_probability:
            sequence_result = []
            for route_id, vocab_item_index in tgt_sequence_probability:
                if route_id == output_classification_head_index:
                    sequence_result.append(vocab_item_index)
                else:
                    sequence_result.append(Constants.NOT_MY_TOKEN_INDEX)
            batch_result.append(sequence_result)
        return torch.tensor(batch_result)

    # OUTPUT_LOGITS = "output_logits"
    # TARGET_OUTPUT_PROBABILITY = "target_output_probability"
    # IS_ITEM_PRESENT = "is_item_present"
    # @staticmethod
    # def create_initial_output_losses_map(output_classification_head_index_list: list) -> dict[int, dict]:
    #     initial_output_losses_map = {}
    #     for key in output_classification_head_index_list:
    #         initial_output_losses_map[key] = {
    #             PreTrainerUtils.OUTPUT_LOGITS: None,
    #             PreTrainerUtils.TARGET_OUTPUT_PROBABILITY: [],
    #             PreTrainerUtils.IS_ITEM_PRESENT: False,
    #         }
    #
    #     return initial_output_losses_map


if __name__ == "__main__":
    embedding_length = 4

    # Create random tensors for x and mask_tensor
    mask_tensor = torch.tensor([[False, False, True, True, True], [False, False, False, True, False]])
    x = torch.randn(mask_tensor.size(0), mask_tensor.size(1), embedding_length, dtype=torch.float32)

    # Create a batch of encoder hidden states
    batch_encoder_hidden_states = [[torch.randn(10, embedding_length)], [torch.randn(10, embedding_length), torch.randn(10, embedding_length)]]

    output = PreTrainerUtils.create_function_param_token_infos(x, mask_tensor, batch_encoder_hidden_states, shift_right=True)
    print(output)
