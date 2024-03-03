# this method is less likely to work. Need more exploration
import unittest
from typing import List, Dict

import torch
from torch import nn

from cl_data.src.constants import Constants
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.cl_pre_trainer import ClPreTrainer
from cl_pretrainer.pre_trainer_checkpoint_manager import ClPreTrainerCheckPointManager
from cl_pretrainer.pre_trainer_utils import PreTrainerUtils
from data_loader.data_loader import DataLoader
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


def cl_pre_trainer_inference(
        model: nn.Module,
        category_vocab_builder: CategoryVocabBuilder,
        output_vocab_builder: OutputVocabBuilder,
        batches: Dict[str, List[List[List[dict]]]],
        masks: Dict[str, List[torch.Tensor]],
        max_decoding_length: int,
):
    model.train(False)
    num_iters = 0
    for i, (src_batch, padding_mask, tgt_batch, future_mask, task_types) in enumerate(
            zip(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY],
                masks[BatchBuilder.PADDING_MASK_KEY],
                batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY],
                masks[BatchBuilder.FUTURE_MASK_KEY],
                masks[BatchBuilder.TASK_TYPE_KEY],
                )
    ):
        # Initially we need at least 4 words for predicting the next word
        current_sequence_length = 4
        truncated_src_batch = [sequence_list[:current_sequence_length] for sequence_list in src_batch]
        truncated_future_mask = BatchBuilder.construct_future_mask(current_sequence_length)

        # range will be max_decoding_length - current_sequence_length
        # (We can add +1 but that will produce the garbage token)
        for index in range(max_decoding_length - current_sequence_length):
            # ~~~~~~~~~~~~~~~~~~~~~~~~~ Compute category probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            e_one = model.category_map_decoder.forward(
                batch_io_parser_output=truncated_src_batch,
                task_types=task_types,
                future_mask=truncated_future_mask,
            )

            category_probability, category_logits = model.category_map_classification_head.forward(e_one)
            predicted_category_map = category_vocab_builder.batch_decode(category_probability.tolist())
            # print(f"Predicted category probability values:"
            #       f" {predicted_category_map}")

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~ Compute output token probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            e_two = model.output_token_decoder.forward(
                batch_io_parser_output=truncated_src_batch,
                task_types=task_types,
                future_mask=truncated_future_mask,
            )

            predicted_io_parser_output_without_token = PreTrainerUtils.convert_category_map_into_io_parser_output_without_token(
                batch_category_map=predicted_category_map,
            )
            batch_route_ids = category_vocab_builder.batch_encoder_output_token_classification_head_vocab_items(
                batch_io_parser_output=predicted_io_parser_output_without_token,
            )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~  is_hub=False ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # SO WE ARE ALSO USING output_vocab_builder.batch_decode_for_inference
            output_probability = model.category_router.forward(
                e_two=e_two,
                batch_route_ids=batch_route_ids,
                is_hub=False,
            )

            predicted_output_token = output_vocab_builder.batch_decode_for_inference(output_probability)
            print(f"Predicted token values:"
                  f" {predicted_output_token}")

            # Removed the teacher forcing and added the prediction to the src batch
            predicted_io_parser_output = PreTrainerUtils.recreate_io_parser_output_switch(predicted_category_map, predicted_output_token, start_from=1)
            truncated_src_batch = PreTrainerUtils.add_prediction_to_truncated_list(predicted_io_parser_output, truncated_src_batch)
            truncated_future_mask = BatchBuilder.construct_future_mask(current_sequence_length + index + 1)

        tgt_batch = [sequence_list[1:] for sequence_list in tgt_batch]
        truncated_src_batch = [sequence_list[1:] for sequence_list in truncated_src_batch]
        print(f"Target batch: {tgt_batch}")
        print(f"Predicted batch: {truncated_src_batch}")
        num_iters += 1

    print("DONE")


class TestClPreTrainerInference(unittest.TestCase):
    PATH = "./saved_models/cl_pre_trainer.pth"
    accepted_loss_threshold = 0.09
    accepted_accuracy_threshold = 0.99

    def test_cl_pre_trainer_model_load_and_inference(self):
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        batch_size = 2
        num_heads = 8
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 2
        dropout_p = 0.1
        max_decoding_length = 16

        # Initializing the data loader
        data_loader = DataLoader()
        data_loader_result = data_loader.create_data_loader_output(
            batch_size=batch_size,
            number_of_batch=4,
            add_bos_and_eos=True,
            max_sequence_length=max_decoding_length,
            task_generator_indexes=[2, 3],
            generator_indexes=[0],
            identifier=0,
            shuffle=True,
        )
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
        cl_pre_trainer_inference(
            model=cl_pre_trainer,
            category_vocab_builder=category_vocab_builder,
            output_vocab_builder=output_vocab_builder,
            batches=batches,
            masks=masks,
            max_decoding_length=max_decoding_length,
        )


if __name__ == "__main__":
    unittest.main()
