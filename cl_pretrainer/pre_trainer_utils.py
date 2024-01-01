import torch
from torch import Tensor, nn

from cl_data.src.constants import Constants
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder
from vocabulary_builder.category_vocabulary_builder import OutputTokenClassificationHeadVocabItem
from vocabulary_builder.output_vocabulary_builder import OutputVocabItem


class PreTrainerUtils:
    NOT_MY_TOKEN_LOSS_WEIGHT = 0.3

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
                        current_map["encoder_hidden_state"] = batch_encoder_hidden_states[i][
                            current_encoder_hidden_state_index]
                        current_tensor = x[i][j - 1].unsqueeze(0) if shift_right else x[i][j].unsqueeze(0)
                    else:
                        current_tensor = torch.cat(
                            (current_tensor, x[i][j - 1].unsqueeze(0) if shift_right else x[i][j].unsqueeze(0)),
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
    def recreate_io_parser_output_switch(
            predicted_category_map: list[list[dict[str, str]]],
            predicted_output_token: list[list[tuple[OutputTokenClassificationHeadVocabItem, OutputVocabItem]]],
            start_from=0,
    ) -> list[list[dict[str, any]]]:
        """
        Recreate the batch of io parser output from predicted category map and output token
        :param start_from: where to start the counting of the position
        :param predicted_category_map: batch of category map
        :param predicted_output_token: batch of output token
        :return: batch io parser output
        """
        batch_io_parser_output = []
        for category_maps, output_tokens in zip(predicted_category_map, predicted_output_token):
            sequence_io_parser_output = []
            position = start_from
            for category_map, output_token in zip(category_maps, output_tokens):
                io_parser_output = {
                    Constants.CATEGORY: category_map,
                    Constants.POSITION: position,
                    Constants.TOKEN: output_token[1]
                }
                sequence_io_parser_output.append(io_parser_output)
                position = position + 1
            batch_io_parser_output.append(sequence_io_parser_output)
        return batch_io_parser_output

    @staticmethod
    def recreate_io_parser_output_hub(
            predicted_category_map: list[list[dict[str, str]]],
            predicted_tokens_map: dict[OutputTokenClassificationHeadVocabItem, dict[any, list[list[OutputVocabItem]] | int]],
            start_from=0,
    ) -> list[list[dict[str, any]]]:
        """
        Recreate the batch of io parser output from predicted category map and predicted token with each classification
        head output tokens.
        :param predicted_category_map: batch of category map
        :param predicted_tokens_map: batch of output token for each output classification head
        :param start_from: where to start the counting of the position
        :return: batch io parser output
        """
        batch_io_parser_output = []
        for i, category_maps in enumerate(predicted_category_map):
            sequence_io_parser_output = []
            position = start_from
            for j, category_map in enumerate(category_maps):
                output_token_classification_head_vocab_item = OutputTokenClassificationHeadVocabItem(
                        category_type=category_map.get(Constants.CATEGORY_TYPE),
                        category_subtype=category_map.get(Constants.CATEGORY_SUB_TYPE),
                    )
                current_token_batch = predicted_tokens_map[output_token_classification_head_vocab_item][OutputVocabBuilder.PREDICTED_TOKEN_KEY]
                current_token_sequence = current_token_batch[i]
                io_parser_output = {
                    Constants.CATEGORY: category_map,
                    Constants.POSITION: position,
                    Constants.TOKEN: current_token_sequence[j]
                }
                sequence_io_parser_output.append(io_parser_output)
                position = position + 1
            batch_io_parser_output.append(sequence_io_parser_output)
        return batch_io_parser_output

    @staticmethod
    def add_prediction_to_truncated_list(
            predicted_io_parser_output: list[list[dict[str, any]]],
            truncated_src_batch: list[list[dict[str, any]]],
    ) -> list[list[dict[str, any]]]:
        for pred_list, src_list in zip(predicted_io_parser_output, truncated_src_batch):
            if pred_list:
                last_dict_in_pred = pred_list[-1]
                src_list.append(last_dict_in_pred)
        return truncated_src_batch

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

    @staticmethod
    def get_category_criterion(category_index_to_count: dict[int, int]) -> any:
        return nn.CrossEntropyLoss(
            label_smoothing=0.01,
            weight=PreTrainerUtils.calculate_loss_weight(category_index_to_count)
        )

    @staticmethod
    def calculate_loss_weight(index_to_count: dict[int, int], is_output_classification_head=False) -> Tensor:
        total_sample = sum(value for value in index_to_count.values())
        sorted_items = sorted(index_to_count.items(), key=lambda item: item[0])
        # Calculate class weights
        weights = torch.zeros(len(sorted_items))
        for class_index, count in sorted_items:
            weight = total_sample / count
            weights[class_index] = weight

        if is_output_classification_head:
            weights[Constants.NOT_MY_TOKEN_INDEX] = PreTrainerUtils.NOT_MY_TOKEN_LOSS_WEIGHT
        return weights

    @staticmethod
    def get_output_criterion_map(index_to_output_vocabularies: dict) -> dict[int, any]:
        losses_map = {}
        for index, output_vocabulary in index_to_output_vocabularies.items():
            current_index_to_count = output_vocabulary[OutputVocabBuilder.INDEX_TO_COUNT]
            current_loss = nn.CrossEntropyLoss(
                label_smoothing=0.01,
                weight=PreTrainerUtils.calculate_loss_weight(current_index_to_count, is_output_classification_head=True)
            )
            losses_map[index] = current_loss
        return losses_map


if __name__ == "__main__":
    embedding_length = 4

    # Create random tensors for x and mask_tensor
    mask_tensor = torch.tensor([[False, False, True, True, True], [False, False, False, True, False]])
    x = torch.randn(mask_tensor.size(0), mask_tensor.size(1), embedding_length, dtype=torch.float32)

    # Create a batch of encoder hidden states
    batch_encoder_hidden_states = [[torch.randn(10, embedding_length)],
                                   [torch.randn(10, embedding_length), torch.randn(10, embedding_length)]]

    output = PreTrainerUtils.create_function_param_token_infos(x, mask_tensor, batch_encoder_hidden_states,
                                                               shift_right=True)
    print(output)
