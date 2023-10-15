import unittest
from typing import List, Dict, Any

import torch
from torch import nn

from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.cl_pre_trainer import ClPreTrainer
from cl_pretrainer.lr_scheduler import NoamOpt
from cl_pretrainer.pre_trainer_checkpoint_manager import ClPreTrainerCheckPointManager
from cl_pretrainer.pre_trainer_train import CURRENT_BATCH_OUTPUT_ACCURACY, CURRENT_BATCH_OUTPUT_LOSS
from cl_pretrainer.pre_trainer_utils import PreTrainerUtils
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


def cl_pre_trainer_inference(
        model: nn.Module,
        category_vocab_builder: CategoryVocabBuilder,
        output_vocab_builder: OutputVocabBuilder,
        batches: Dict[str, List[List[List[dict]]]],
        masks: Dict[str, List[torch.Tensor]],
        task_type: str,
        criterion: Any,
        max_decoding_length: int,
        start_epoch=0,
):
    model.train(False)
    num_iters = 0

    for i, (src_batch, tgt_batch, future_mask) in enumerate(
            zip(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY],
                batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY],
                masks[BatchBuilder.FUTURE_MASK_KEY])):
        current_sequence_length = 4
        truncated_src_batch = [sequence_list[:current_sequence_length] for sequence_list in src_batch]
        truncated_future_mask = BatchBuilder.construct_future_mask(current_sequence_length)
        for index in range(1): #,max_decoding_length+1
            # Staring left side for category map
            e_one = model.category_map_decoder.forward(
                batch_io_parser_output=truncated_src_batch,
                task_type=task_type,
                future_mask=truncated_future_mask,
            )

            category_probability, _ = model.category_map_classification_head.forward(e_one)
            predicted_category_map = category_vocab_builder.batch_decode(category_probability.tolist())
            print(f"Predicted category probability values:"
                  f" {predicted_category_map}")

            # Starting right side for output token
            e_two = model.output_token_decoder.forward(
                batch_io_parser_output=truncated_src_batch,
                task_type=task_type,
                future_mask=truncated_future_mask,
            )

            predicted_io_parser_output_without_token = PreTrainerUtils.convert_category_map_into_io_parser_output_without_token(
                batch_category_map=predicted_category_map,
            )
            batch_route_ids = category_vocab_builder.batch_encoder_output_token_classification_head_vocab_items(
                batch_io_parser_output=predicted_io_parser_output_without_token,
            )

            output_probability = model.category_router.forward(
                e_two=e_two,
                batch_route_ids=batch_route_ids,
                is_training=False,
            )

            predicted_output_token = output_vocab_builder.batch_decode_for_inference(output_probability)
            print(f"Predicted token values:"
                  f" {predicted_output_token}")

            # TODO remove Teacher forcing \/
            # truncated_src_batch = BatchBuilder.get_batch_io_parser_output(sentences, True, index + 1)
            # future_mask = BatchBuilder.construct_future_mask(index + 1)

        # TODO Calculate the loss here
        num_iters += 1
    pass


class TestClPreTrainerInference(unittest.TestCase):
    PATH = "./saved_models/cl_pre_trainer.pth"
    accepted_loss_threshold = 0.09
    accepted_accuracy_threshold = 0.99

    def test_cl_pre_trainer_model_load_and_inference(self):
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        batch_size = 3
        num_heads = 8
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 2
        dropout_p = 0.1
        max_decoding_length = 16
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value

        sentences = [
            "The quick brown fox jumps over the lazy dog in the meadow",
            "Adding 3 plus 2 equals ##addition(3,2)",
            "##subtraction(5,1) is the minus value of 1 from 5",
        ]
        corpus_io_parser_output = BatchBuilder.get_batch_io_parser_output(sentences, True, 15)
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
        batches, masks = BatchBuilder.construct_batches_for_cl_pre_trainer(
            sentences,
            batch_size=batch_size,
            max_decoder_sequence_length=max_decoding_length,
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
            index_to_output_vocabularies=output_vocabularies,
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
        cl_pre_trainer.load_saved_model_from_state_dict(
            ClPreTrainerCheckPointManager.get_checkpoint_item(
                checkpoint_map,
                ClPreTrainerCheckPointManager.CL_PRE_TRAINER_STATE,
            ),
        )
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

        print("Model loaded correctly...")

        criterion = nn.CrossEntropyLoss()

        # Call training loop for inference using the saved model ...
        # latest_batch_loss, latest_batch_accuracy, output_logits_map = \
        cl_pre_trainer_inference(
            model=cl_pre_trainer,
            category_vocab_builder=category_vocab_builder,
            output_vocab_builder=output_vocab_builder,
            criterion=criterion,
            batches=batches,
            masks=masks,
            start_epoch=start_epoch,
            task_type=task_type,
            max_decoding_length=max_decoding_length,
        )

        # print(f"batch loss {latest_batch_loss.item()}")
        # print(f"batch accuracy {latest_batch_accuracy}")
        # self.assertEqual(latest_batch_loss.item() <= TestClPreTrainerInference.accepted_loss_threshold, True)
        # self.assertEqual(latest_batch_accuracy >= TestClPreTrainerInference.accepted_accuracy_threshold, True)
        # for index, output_logits_item in output_logits_map.items():
        #     output_loss = output_logits_item[CURRENT_BATCH_OUTPUT_LOSS]
        #     output_accuracy = output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY]
        #     self.assertEqual(output_loss.item() <= TestClPreTrainerInference.accepted_loss_threshold, True)
        #     self.assertEqual(output_accuracy >= TestClPreTrainerInference.accepted_accuracy_threshold, True)


if __name__ == "__main__":
    unittest.main()
