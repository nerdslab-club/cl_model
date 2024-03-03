# complete this first
import copy
import unittest
from typing import List, Dict, Any

import torch
from torch import nn

from category_router.category_router import CategoryRouter
from cl_data.src.constants import Constants, SpecialTokens
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.cl_pre_trainer import ClPreTrainer
from cl_pretrainer.pre_trainer_checkpoint_manager import ClPreTrainerCheckPointManager
from cl_pretrainer.pre_trainer_utils import PreTrainerUtils
from data_loader.data_loader import DataLoader
from evaluation_metric.bleu import get_n_gram_weights, calculate_corpus_bleu_score
from evaluation_metric.perplexity import get_target_tokens_probability, calculate_batch_perplexity
from response_parser.response_parser import ResponseParser
from response_parser.simple_response_parser import SimpleResponseParser
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


def cl_pre_trainer_inference_hub(
        model: nn.Module,
        category_vocab_builder: CategoryVocabBuilder,
        output_vocab_builder: OutputVocabBuilder,
        batches: Dict[str, List[List[List[dict]]]],
        masks: Dict[str, List[torch.Tensor]],
        max_decoding_length: int,
):
    target_batches = []
    predicted_batches = []
    output_logits_map_batches = []
    model.train(False)
    num_iters = 0
    for i, (src_batch, padding_mask, tgt_batch, future_mask, task_types, initial_token_count) in enumerate(
            zip(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY],
                masks[BatchBuilder.PADDING_MASK_KEY],
                batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY],
                masks[BatchBuilder.FUTURE_MASK_KEY],
                masks[BatchBuilder.TASK_TYPE_KEY],
                masks[BatchBuilder.INITIAL_TOKEN_COUNT_KEY],
                )
    ):
        # Initially we need at least 4 words for predicting the next word
        current_sequence_length = initial_token_count
        truncated_src_batch = [sequence_list[:current_sequence_length] for sequence_list in src_batch]
        truncated_src_batch_input = copy.deepcopy(truncated_src_batch)
        truncated_src_batch_input = [BatchBuilder.add_padding_and_eos(
            sequence_list,
            max_decoding_length,
            add_bos_and_eos=True,
            is_eos_finishing_token=False,
        ) for sequence_list in truncated_src_batch_input]
        # truncated_future_mask = BatchBuilder.construct_future_mask(current_sequence_length)

        output_logits_map = {}
        # range will be max_decoding_length - current_sequence_length
        # (We can add +1 but that will produce the garbage token)
        for index in range(max_decoding_length - current_sequence_length):
            try:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~ Compute category probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                e_one = model.category_map_decoder.forward(
                    batch_io_parser_output=truncated_src_batch_input,
                    task_types=task_types,
                    # future_mask=truncated_future_mask,
                )

                category_probability, category_logits = model.category_map_classification_head.forward(e_one)
                predicted_category_map = category_vocab_builder.batch_decode(category_probability.tolist())
                # print(f"Predicted category probability values:"
                #       f" {predicted_category_map}")

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~ Compute output token probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                e_two = model.output_token_decoder.forward(
                    batch_io_parser_output=truncated_src_batch_input,
                    task_types=task_types,
                    # future_mask=truncated_future_mask,
                )

                predicted_io_parser_output_without_token = PreTrainerUtils.convert_category_map_into_io_parser_output_without_token(
                    batch_category_map=predicted_category_map,
                )
                batch_route_ids = category_vocab_builder.batch_encoder_output_token_classification_head_vocab_items(
                    batch_io_parser_output=predicted_io_parser_output_without_token,
                )
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~  is_hub=True ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                # SO WE ARE ALSO USING output_vocab_builder.batch_decode_for_training
                output_logits_map = model.category_router.forward(
                    e_two=e_two,
                    batch_route_ids=batch_route_ids,
                    is_hub=True,
                )

                predicted_tokens_map = {}
                for output_classification_head_index, output_logits_item in output_logits_map.items():
                    current_head_output_probability = output_logits_item[CategoryRouter.OUTPUT_PROBABILITY]
                    # current_head_output_probability = current_head_output_probability[:, :-1]
                    current_head_predicted_output_token = output_vocab_builder.batch_decode_for_training(
                        output_classification_head_index,
                        current_head_output_probability.tolist(),
                    )
                    # print(f"Predicted token values for index: {output_classification_head_index} is \n"
                    #       f"{current_head_predicted_output_token}")

                    # Get the output token classification vocab item from index
                    current_output_token_classification_head_vocab_item = \
                        output_vocab_builder.index_to_output_vocabularies[output_classification_head_index][
                            OutputVocabBuilder.OUTPUT_TOKEN_CLASSIFICATION_HEAD_VOCAB_ITEM]

                    # Add item to predicted tokens map using the output token classification head vocab item as key
                    predicted_tokens_map[current_output_token_classification_head_vocab_item] = {
                        OutputVocabBuilder.PREDICTED_TOKEN_KEY: current_head_predicted_output_token,
                        OutputVocabBuilder.INDEX: output_classification_head_index,
                    }
                # print("\n")

                # Removed the teacher forcing and added the prediction to the src batch
                predicted_io_parser_output = PreTrainerUtils.recreate_io_parser_output_hub(predicted_category_map,
                                                                                           predicted_tokens_map,
                                                                                           start_from=1)

                truncated_src_batch = PreTrainerUtils.add_prediction_to_truncated_list(
                    predicted_io_parser_output,
                    truncated_src_batch,
                    current_sequence_length - 1 + index,
                )
                truncated_src_batch_input = copy.deepcopy(truncated_src_batch)
                truncated_src_batch_input = [BatchBuilder.add_padding_and_eos(
                    sequence_list,
                    max_decoding_length,
                    add_bos_and_eos=True,
                    is_eos_finishing_token=False,
                ) for sequence_list in truncated_src_batch_input]

                if truncated_src_batch[0][-1][Constants.TOKEN] == SpecialTokens.ENDING.value:
                    break
                # truncated_future_mask = BatchBuilder.construct_future_mask(current_sequence_length + index + 1)

            except Exception as e:
                print(f"An error occurred for batch: {i} word: {index} error: {e}")

        # Removing <BOS> from both tgt and predicted sentences
        tgt_batch = [sequence_list[1:len(truncated_src_batch[i])] for i, sequence_list in enumerate(tgt_batch)]
        truncated_src_batch = [sequence_list[1:] for sequence_list in truncated_src_batch]
        target_batches.append(tgt_batch)
        predicted_batches.append(truncated_src_batch)
        output_logits_map_batches.append(output_logits_map)
        num_iters += 1
        if num_iters == 16:
            break
        # break

    calculate_bleu_score(target_batches, predicted_batches)
    calculate_perplexity_score(target_batches, output_logits_map_batches, output_vocab_builder)
    print_response(predicted_batches)
    print("DONE")


def print_response(predicted_batches: List[List[List[dict]]]):
    for index, batch in enumerate(predicted_batches):
        parsed_response_list = ResponseParser.parse_corpus_io_parser_output(batch)
        print(f"For batch: {index}\n Parser response list is: {parsed_response_list} ")


def calculate_perplexity_score(
        target_batches: List[List[List[dict]]],
        output_logits_map_batches: List[dict[int, dict[str, Any]]],
        output_vocab_builder: OutputVocabBuilder,
):
    for batch_index, (target_batch, output_logits_map) in enumerate(zip(target_batches, output_logits_map_batches)):
        target_batch_extracted_token = PreTrainerUtils.extract_tokens(target_batch)
        batch_predicted_probabilities = get_target_tokens_probability(
            target_batch,
            output_logits_map,
            output_vocab_builder,
        )
        perplexity_score = calculate_batch_perplexity(
            target_batch_extracted_token,
            batch_predicted_probabilities
        )
        print(f"Perplexity Score of the {batch_index} th corpus is: {perplexity_score}")


def calculate_bleu_score(target_batches: List[List[List[dict]]], predicted_batches: List[List[List[dict]]]):
    for batch_index, (target_batch, predicted_batch) in enumerate(zip(target_batches, predicted_batches)):
        target_batch_extracted_token = PreTrainerUtils.extract_tokens(target_batch)
        predicted_batch_extracted_token = PreTrainerUtils.extract_tokens(predicted_batch)

        bleu_score = calculate_corpus_bleu_score(
            target_batch_extracted_token,
            predicted_batch_extracted_token,
            bleu_weights=get_n_gram_weights(2),
        )
        # Printing the raw response
        print("Target batch: \n")
        deep_copied_tgt_list = copy.deepcopy(target_batch)
        SimpleResponseParser.print_response_to_console(deep_copied_tgt_list)
        print("Predicted batch: \n")
        deep_copied_predicted_list = copy.deepcopy(predicted_batch)
        SimpleResponseParser.print_response_to_console(deep_copied_predicted_list)
        print(f"BLEU Score of the {batch_index} th corpus is: {bleu_score}")


class TestClPreTrainerInference(unittest.TestCase):
    PATH = "./saved_models/cl_pre_trainer_generative_best_v4.pth"
    accepted_loss_threshold = 0.09
    accepted_accuracy_threshold = 0.99

    def test_cl_pre_trainer_model_load_and_inference(self):
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f'Selected Device: {device}')

        # # Hyperparameters
        # batch_size = 2 if device == torch.device("cpu") else 4
        # num_heads = 8
        # hidden_dim = 768
        # ff_dim = 2048
        # num_layers = 2
        # dropout_p = 0.1
        # max_decoding_length = 16
        # task_generator_indexes = [3]
        # generator_range = 2 if device == torch.device("cpu") else 10
        # number_of_batch = generator_range * len(task_generator_indexes)
        # seed = 42
        # add_bos_and_eos = True

        # Hyperparameters
        batch_size = 4 if device == torch.device("cpu") else 4
        num_heads = 8
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 6
        dropout_p = 0.1
        max_decoding_length = 24
        task_generator_indexes = [0, 1, 2]
        generator_range = 20 if device == torch.device("cpu") else 20
        number_of_batch = generator_range * len(task_generator_indexes)
        seed = 42
        add_bos_and_eos = True
        training_batch_size = 1

        # Initializing the data loader
        data_loader = DataLoader()

        data_loader_result = data_loader.create_data_loader_output(
            batch_size=batch_size,
            number_of_batch=number_of_batch,
            add_bos_and_eos=add_bos_and_eos,
            max_sequence_length=max_decoding_length,
            task_generator_indexes=task_generator_indexes,
            generator_indexes=[i for i in range(generator_range)],
            identifier=0,
            shuffle=True,
            seed=seed,
        )
        # print(data_loader_result)
        batch_size = training_batch_size

        corpus_io_parser_output = [item[Constants.IO_PARSER_OUTPUT] for item in data_loader_result]

        # Initialize category vocabulary builder instance
        category_vocab_builder = CategoryVocabBuilder(corpus_io_parser_output)
        category_vocab_size = len(category_vocab_builder.category_vocab_item_to_index.keys())

        print(f"Output token classification head count:"
              f" {len(category_vocab_builder.index_to_output_token_classification_head_vocab_item.keys())}\n"
              f"Output token classification head category type:"
              f" {category_vocab_builder.index_to_output_token_classification_head_vocab_item}")

        # Initialize output vocabulary builder instance
        output_vocab_builder = OutputVocabBuilder(
            corpus_of_io_parser_output=corpus_io_parser_output,
            index_to_output_token_classification_head_vocab_item=
            category_vocab_builder.index_to_output_token_classification_head_vocab_item
        )
        output_vocabularies = output_vocab_builder.index_to_output_vocabularies
        print(f"Output vocabularies count: {len(output_vocabularies.keys())}")

        # Creating the batch and masks
        batches, masks = BatchBuilder.construct_batches_for_cl_pre_trainer_with_data_loader(
            data_loader_result,
            batch_size=batch_size,
            max_decoder_sequence_length=max_decoding_length,
            is_generative_training=False,
            add_bos_and_eos=add_bos_and_eos,
        )
        # Initializing the CL pre trainer
        cl_pre_trainer = ClPreTrainer(
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_decoding_length=max_decoding_length,
            dropout_p=dropout_p,
            category_vocab_size=category_vocab_size,
            output_vocab_builder=output_vocab_builder,
            use_our_tokenizer=True,
        )
        cl_pre_trainer.eval()

        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        optimizer = torch.optim.Adam(
            cl_pre_trainer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9
        )

        # Load the model...
        checkpoint_map = ClPreTrainerCheckPointManager.load_checkpoint_map(
            TestClPreTrainerInference.PATH
        )
        # Load CL-Pre-Trainer states
        cl_pre_trainer.load_saved_model_from_state_dict(
            ClPreTrainerCheckPointManager.get_checkpoint_item(
                checkpoint_map,
                ClPreTrainerCheckPointManager.CL_PRE_TRAINER_STATE,
            ),
        )
        # Load Optimizer states
        optimizer.load_state_dict(
            ClPreTrainerCheckPointManager.get_checkpoint_item(
                checkpoint_map,
                ClPreTrainerCheckPointManager.OPTIM_STATE,
            ),
        )
        start_epoch = ClPreTrainerCheckPointManager.get_checkpoint_item(
            checkpoint_map,
            ClPreTrainerCheckPointManager.EPOCH,
        )

        # Load Category map Output token classification heads state
        cl_pre_trainer.category_router.load_all_output_classification_head(
            ClPreTrainerCheckPointManager.get_checkpoint_item(
                checkpoint_map,
                ClPreTrainerCheckPointManager.OUTPUT_TOKEN_CLASSIFICATION_HEADS_STATE,
            ),
        )

        print("Model loaded correctly...")

        # Call the inference method without teacher forcing
        cl_pre_trainer_inference_hub(
            model=cl_pre_trainer,
            category_vocab_builder=category_vocab_builder,
            output_vocab_builder=output_vocab_builder,
            batches=batches,
            masks=masks,
            max_decoding_length=max_decoding_length,
        )


if __name__ == "__main__":
    unittest.main()
