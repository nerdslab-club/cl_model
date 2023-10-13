import unittest
from typing import List, Dict, Any

import torch
from torch import nn

from category_router.category_router import CategoryRouter
from cl_data.src.constants import TaskTypes
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.cl_pre_trainer import ClPreTrainer
from cl_pretrainer.lr_scheduler import NoamOpt
from cl_pretrainer.pre_trainer_checkpoint_manager import ClPreTrainerCheckPointManager
from cl_pretrainer.pre_trainer_utils import PreTrainerUtils
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder

CURRENT_BATCH_OUTPUT_LOSS = "current_batch_output_loss"
CURRENT_BATCH_OUTPUT_ACCURACY = "current_batch_output_accuracy"


def cl_pre_trainer_train(
        model: nn.Module,
        category_vocab_builder: CategoryVocabBuilder,
        output_vocab_builder: OutputVocabBuilder,
        scheduler: Any,
        criterion: Any,
        batches: Dict[str, List[List[List[dict]]]],
        masks: Dict[str, List[torch.Tensor]],
        n_epochs: int,
        task_type: str,
        start_epoch=0,
        is_training=True,
        verbose_log=False,
):
    model.train(is_training)
    if not is_training:
        n_epochs = 1

    num_iters = 0
    for e in range(start_epoch, start_epoch + n_epochs):
        for i, (src_batch, padding_mask, tgt_batch, future_mask) in enumerate(
                zip(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY],
                    masks[BatchBuilder.PADDING_MASK_KEY],
                    batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY],
                    masks[BatchBuilder.FUTURE_MASK_KEY])
        ):
            # ~~~~~~~~~~~~~~~~~~~~~~~~~ Compute category probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tgt_category_probability = torch.tensor(category_vocab_builder.batch_encoder(tgt_batch))
            # Removing the <BOS> category map
            tgt_category_probability = tgt_category_probability[:, 1:]

            e_one = model.category_map_decoder.forward(
                batch_io_parser_output=src_batch,
                task_type=task_type,
                future_mask=future_mask,
            )

            category_probability, category_logits = model.category_map_classification_head.forward(e_one)
            category_logits = category_logits[:, :-1, :]

            # Compute the average cross-entropy loss over all next-token predictions at each index i given [1, ..., i]
            # for the entire batch. Note that the original paper uses label smoothing (I was too lazy).
            batch_category_loss = criterion(
                category_logits.contiguous().permute(0, 2, 1),
                tgt_category_probability.contiguous().long(),
            )

            # Rough estimate of per-token accuracy in the current training batch
            batch_category_accuracy = (torch.sum(
                category_logits.argmax(dim=-1) == tgt_category_probability)) / torch.numel(tgt_category_probability)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~ Compute output token probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tgt_output_probability = output_vocab_builder.batch_encoder(tgt_batch, is_only_probability=False)

            e_two = model.output_token_decoder.forward(
                batch_io_parser_output=src_batch,
                task_type=task_type,
                future_mask=future_mask,
            )

            # As predicting proper category is the responsibility of the left side.
            # That's why we are only concerning ourselves with the output token prediction.
            # Provided we are always using the correct output classification head.
            # That's why using the src_batch instead of the predicted batch category map.
            batch_route_ids = category_vocab_builder.batch_encoder_output_token_classification_head_vocab_items(
                batch_io_parser_output=src_batch,
            )
            output_logits_map = model.category_router.forward(
                e_two=e_two,
                batch_route_ids=batch_route_ids,
                is_training=True,
            )

            # Calculate output loss & accuracy for each classification head
            for index, output_logits_item in output_logits_map.items():
                current_tgt_output_probability = PreTrainerUtils.create_tgt_tensor_for_output_classification_head(
                    output_classification_head_index=index,
                    tgt_batch_probability=tgt_output_probability,
                )
                current_output_logits = output_logits_item[CategoryRouter.OUTPUT_LOGITS]

                current_batch_output_loss = criterion(
                    current_output_logits.contiguous().permute(0, 2, 1),
                    current_tgt_output_probability.contiguous().long(),
                )
                current_batch_output_accuracy = (torch.sum(
                    current_output_logits.argmax(dim=-1) == current_tgt_output_probability)) / torch.numel(
                    current_tgt_output_probability)

                output_logits_item[CURRENT_BATCH_OUTPUT_LOSS] = current_batch_output_loss
                output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY] = current_batch_output_accuracy
                output_logits_map[index] = output_logits_item

            if num_iters % len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY]) == 0 or not is_training:
                print(
                    f"epoch: {e}, num_iters: {num_iters}, "
                    f"batch_category_loss: {batch_category_loss}, batch_category_accuracy: {batch_category_accuracy}"
                )
                for index, output_logits_item in output_logits_map.items():
                    output_loss = output_logits_item[CURRENT_BATCH_OUTPUT_LOSS]
                    output_accuracy = output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY]
                    print(f"output loss for index: {index} is {output_loss}\n"
                          f"output accuracy for index: {index} is {output_accuracy}")

                if verbose_log:
                    predicted_category_map = category_vocab_builder.batch_decode(category_probability.tolist())
                    print(f"Predicted category probability values:"
                          f" {predicted_category_map}")

                    for index, output_logits_item in output_logits_map.items():
                        current_head_output_probability = output_logits_item[CategoryRouter.OUTPUT_PROBABILITY]
                        current_head_predicted_output_token = output_vocab_builder.batch_decode_for_training(
                            index,
                            current_head_output_probability.tolist(),
                        )
                        print(f"Predicted token values for index: {index} is \n"
                              f"{current_head_predicted_output_token}")

            # Update parameters
            if is_training:
                batch_category_loss.backward(retain_graph=True)
                for index, output_logits_item in output_logits_map.items():
                    output_logits_item[CURRENT_BATCH_OUTPUT_LOSS].backward(retain_graph=True)
                scheduler.step()
                scheduler.optimizer.zero_grad()
            num_iters += 1
    return batch_category_loss, batch_category_accuracy, output_logits_map


class TestClPreTrainerTraining(unittest.TestCase):
    PATH = "./saved_models/cl_pre_trainer.pth"

    def test_cl_pre_trainer_train_and_save(self):
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        n_epochs = 35
        batch_size = 3
        num_heads = 8
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 2
        dropout_p = 0.1
        max_decoding_length = 16
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value

        # Creating the vocabulary corpus
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
        scheduler = NoamOpt(
            cl_pre_trainer.hidden_dim,
            factor=1,
            warmup=400,
            optimizer=optimizer,
        )
        criterion = nn.CrossEntropyLoss()

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy, output_logits_map = cl_pre_trainer_train(
            model=cl_pre_trainer,
            category_vocab_builder=category_vocab_builder,
            output_vocab_builder=output_vocab_builder,
            scheduler=scheduler,
            criterion=criterion,
            batches=batches,
            masks=masks,
            n_epochs=n_epochs,
            task_type=task_type,
            is_training=True,
            verbose_log=False,
        )

        print(f"batch loss {latest_batch_loss.item()}")
        print(f"batch accuracy {latest_batch_accuracy}")
        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)
        for index, output_logits_item in output_logits_map.items():
            output_loss = output_logits_item[CURRENT_BATCH_OUTPUT_LOSS]
            output_accuracy = output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY]
            self.assertEqual(output_loss.item() <= 0.01, True)
            self.assertEqual(output_accuracy >= 0.99, True)

        ClPreTrainerCheckPointManager.save_checkpoint_map(
            path=TestClPreTrainerTraining.PATH,
            epoch=n_epochs,
            model=cl_pre_trainer,
            optimizer=optimizer,
        )

    def test_cl_pre_trainer_model_load(self):
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        n_epochs = 35
        batch_size = 3
        num_heads = 8
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 2
        dropout_p = 0.1
        max_decoding_length = 16
        task_type = TaskTypes.NL_TO_NL_TRANSLATION.value

        # Creating the vocabulary corpus
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
            TestClPreTrainerTraining.PATH
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

        scheduler = NoamOpt(
            cl_pre_trainer.hidden_dim,
            factor=1,
            warmup=400,
            optimizer=optimizer,
        )
        criterion = nn.CrossEntropyLoss()

        # Call training loop for inference using the saved model ...
        latest_batch_loss, latest_batch_accuracy, output_logits_map = cl_pre_trainer_train(
            model=cl_pre_trainer,
            category_vocab_builder=category_vocab_builder,
            output_vocab_builder=output_vocab_builder,
            scheduler=scheduler,
            criterion=criterion,
            batches=batches,
            masks=masks,
            n_epochs=n_epochs,
            task_type=task_type,
            is_training=False,
            verbose_log=True,
        )

        print(f"batch loss {latest_batch_loss.item()}")
        print(f"batch accuracy {latest_batch_accuracy}")
        self.assertEqual(latest_batch_loss.item() <= 0.01, True)
        self.assertEqual(latest_batch_accuracy >= 0.99, True)
        for index, output_logits_item in output_logits_map.items():
            output_loss = output_logits_item[CURRENT_BATCH_OUTPUT_LOSS]
            output_accuracy = output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY]
            self.assertEqual(output_loss.item() <= 0.01, True)
            self.assertEqual(output_accuracy >= 0.99, True)


if __name__ == "__main__":
    unittest.main()
