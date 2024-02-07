import unittest
from typing import List, Dict, Any

import torch
from torch import nn

from category_router.category_router import CategoryRouter
from cl_data.src.constants import Constants
from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.cl_pre_trainer import ClPreTrainer
from cl_pretrainer.lr_scheduler import NoamOpt
from cl_pretrainer.pre_trainer_checkpoint_manager import ClPreTrainerCheckPointManager
from cl_pretrainer.pre_trainer_utils import PreTrainerUtils
from data_loader.data_loader import DataLoader
from evaluation_metric.bleu import calculate_corpus_bleu_score, get_n_gram_weights
from response_parser.response_parser import ResponseParser
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder

CURRENT_BATCH_OUTPUT_LOSS = "current_batch_output_loss"
CURRENT_BATCH_OUTPUT_ACCURACY = "current_batch_output_accuracy"


def cl_pre_trainer_train(
        model: nn.Module,
        category_vocab_builder: CategoryVocabBuilder,
        output_vocab_builder: OutputVocabBuilder,
        scheduler: Any,
        batches: Dict[str, List[List[List[dict]]]],
        masks: Dict[str, List[torch.Tensor]],
        n_epochs: int,
        category_criterion: any,
        output_criterion_map: dict[int, any],
        patience=5,
        start_epoch=0,
        is_training=True,
        verbose_log=False,
        only_language_training=0,
        device: torch.device = torch.device('cpu')
):
    model.train(is_training)
    if not is_training:
        n_epochs = 1

    num_iters = 0
    best_accuracy = 0
    best_loss = float('inf')
    epochs_without_improvement = 0
    execute_epoch = 40
    for epoch in range(start_epoch, start_epoch + n_epochs):
        total_accuracy = 0
        total_loss = 0
        for i, (src_batch, padding_mask, tgt_batch, future_mask, task_types) in enumerate(
                zip(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY],
                    masks[BatchBuilder.PADDING_MASK_KEY],
                    batches[BatchBuilder.DECODER_IO_PARSER_OUTPUT_KEY],
                    masks[BatchBuilder.FUTURE_MASK_KEY],
                    masks[BatchBuilder.TASK_TYPE_KEY],
                    )
        ):
            # Move to CPU or Cuda
            padding_mask = padding_mask.to(device)
            future_mask = future_mask.to(device)

            # ~~~~~~~~~~~~~~~~~~~~~~~~~ Compute category probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tgt_category_probability = torch.tensor(category_vocab_builder.batch_encoder(tgt_batch)).to(device)
            # Removing the <BOS> category map
            tgt_category_probability = tgt_category_probability[:, 1:]

            e_one = model.category_map_decoder.forward(
                batch_io_parser_output=src_batch,
                task_types=task_types,
                future_mask=future_mask,
                src_padding_mask=padding_mask,
            )

            category_probability, category_logits = model.category_map_classification_head.forward(e_one)
            category_probability = category_probability[:, :-1]
            category_logits = category_logits[:, :-1, :]

            # Compute the average cross-entropy loss over all next-token predictions at each index i given [1, ..., i]
            # for the entire batch. Note that the original paper uses label smoothing (I was too lazy).
            batch_category_loss = category_criterion(
                category_logits.contiguous().permute(0, 2, 1),
                tgt_category_probability.contiguous().long(),
            )
            total_loss += batch_category_loss

            # Rough estimate of per-token accuracy in the current training batch
            batch_category_accuracy = (torch.sum(
                category_logits.argmax(dim=-1) == tgt_category_probability)) / torch.numel(tgt_category_probability)

            total_accuracy += batch_category_accuracy

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~ Compute output token probability ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            tgt_output_probability = output_vocab_builder.batch_encoder(tgt_batch, is_only_probability=False)

            e_two = model.output_token_decoder.forward(
                batch_io_parser_output=src_batch,
                task_types=task_types,
                future_mask=future_mask,
                src_padding_mask=padding_mask,
            )

            # As predicting proper category is the responsibility of the left side.
            # That's why we are only concerning ourselves with the output token prediction.
            # Provided we are always using the correct output classification head.
            # That's why using the src_batch instead of the predicted batch category map.
            batch_route_ids = category_vocab_builder.batch_encoder_output_token_classification_head_vocab_items(
                batch_io_parser_output=src_batch,
            )
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~  is_hub=True ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # SO WE ARE ALSO USING output_vocab_builder.batch_decode_for_training
            output_logits_map = model.category_router.forward(
                e_two=e_two,
                batch_route_ids=batch_route_ids,
                is_hub=True,
            )

            combined_output_losses = []
            # Calculate output loss & accuracy for each classification head
            for index, output_logits_item in output_logits_map.items():
                if epoch >= only_language_training or index == 1:
                    current_tgt_output_probability = PreTrainerUtils.create_tgt_tensor_for_output_classification_head(
                        output_classification_head_index=index,
                        tgt_batch_probability=tgt_output_probability,
                    ).to(device)
                    # Removing the <BOS> tgt output probability
                    current_tgt_output_probability = current_tgt_output_probability[:, 1:]

                    current_output_logits = output_logits_item[CategoryRouter.OUTPUT_LOGITS]
                    current_output_logits.to(device)

                    # Removing the last garbage token from output logits
                    current_output_logits = current_output_logits[:, :-1, :]

                    current_batch_output_loss = output_criterion_map[index](
                        current_output_logits.contiguous().permute(0, 2, 1),
                        current_tgt_output_probability.contiguous().long(),
                    )
                    combined_output_losses.append(current_batch_output_loss)

                    current_batch_output_accuracy = (torch.sum(
                        current_output_logits.argmax(dim=-1) == current_tgt_output_probability)) / torch.numel(
                        current_tgt_output_probability)

                    output_logits_item[CURRENT_BATCH_OUTPUT_LOSS] = current_batch_output_loss
                    output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY] = current_batch_output_accuracy
                    total_accuracy += current_batch_output_accuracy
                    total_loss += current_batch_output_loss
                    output_logits_map[index] = output_logits_item
                else:
                    output_logits_item[CURRENT_BATCH_OUTPUT_LOSS] = 100
                    output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY] = 0
                    output_logits_map[index] = output_logits_item

            print_model_training_status(
                [sequence_list[1:] for sequence_list in tgt_batch],
                batch_category_accuracy,
                batch_category_loss,
                batches,
                category_probability,
                category_vocab_builder,
                epoch,
                is_training,
                num_iters,
                output_logits_map,
                output_vocab_builder,
                verbose_log,
            )

            # Update parameters
            if is_training:
                batch_category_loss.backward()
                total_loss = sum(combined_output_losses)
                total_loss.backward()
                scheduler.step()
                if num_iters % len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY]) == 0:
                    print(f"Current learning rate is {scheduler.get_current_rate()} and running rate is {scheduler.get_rate()}")
                scheduler.optimizer.zero_grad()
            num_iters += 1
            # batch

        # Saving the best model ...
        best_accuracy = save_best_model(best_accuracy, epoch, model, scheduler, total_accuracy)

        # Applying early stopping using the total loss ...
        if total_loss < best_loss:
            best_loss = total_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch + 1} epochs with no improvement.")
            execute_epoch = epoch + 1
            break
        # epoch
    # function
    print(f"Best accuracy found: {best_accuracy}")
    return batch_category_loss, batch_category_accuracy, output_logits_map, execute_epoch


def save_best_model(best_accuracy, epoch, model, scheduler, total_accuracy):
    if total_accuracy > best_accuracy and epoch > 20:
        best_accuracy = total_accuracy
        # Saving the best model
        ClPreTrainerCheckPointManager.save_checkpoint_map(
            path=TestClPreTrainerTraining.BEST_PATH,
            epoch=epoch + 1,
            model=model,
            optimizer=scheduler.optimizer,
        )
        print(f"Saved best model at epoch: {epoch + 1} with best accuracy: {best_accuracy}")
    return best_accuracy


def print_model_training_status(
        target_batch,
        batch_category_accuracy,
        batch_category_loss,
        batches,
        category_probability,
        category_vocab_builder,
        e,
        is_training,
        num_iters,
        output_logits_map,
        output_vocab_builder,
        verbose_log):
    if num_iters % len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY]) == 0 or not is_training:
        print(
            f"epoch: {e}, num_iters: {num_iters}, "
            f"batch_category_loss: {batch_category_loss}, batch_category_accuracy: {batch_category_accuracy}"
        )
        for index, output_logits_item in output_logits_map.items():
            output_loss = output_logits_item[CURRENT_BATCH_OUTPUT_LOSS]
            print(f"output loss for index: {index} is {output_loss}")
        for index, output_logits_item in output_logits_map.items():
            output_accuracy = output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY]
            print(f"output accuracy for index: {index} is {output_accuracy}")

        if verbose_log:
            predicted_category_map = category_vocab_builder.batch_decode(category_probability.tolist())
            print(f"Predicted category probability values:"
                  f" {predicted_category_map}")

            predicted_tokens_map = {}
            for index, output_logits_item in output_logits_map.items():
                current_head_output_probability = output_logits_item[CategoryRouter.OUTPUT_PROBABILITY]
                current_head_output_probability = current_head_output_probability[:, :-1]
                current_head_predicted_output_token = output_vocab_builder.batch_decode_for_training(
                    index,
                    current_head_output_probability.tolist(),
                )
                print(f"Predicted token values for index: {index} is \n"
                      f"{current_head_predicted_output_token}")

                if not is_training:
                    # Get the output token classification vocab item from index
                    current_output_token_classification_head_vocab_item = \
                        output_vocab_builder.index_to_output_vocabularies[index][
                            OutputVocabBuilder.OUTPUT_TOKEN_CLASSIFICATION_HEAD_VOCAB_ITEM]

                    # Add item to predicted tokens map using the output token classification head vocab item as key
                    predicted_tokens_map[current_output_token_classification_head_vocab_item] = {
                        OutputVocabBuilder.PREDICTED_TOKEN_KEY: current_head_predicted_output_token,
                        OutputVocabBuilder.INDEX: index,
                    }
            if not is_training:
                predicted_io_parser_output = PreTrainerUtils.recreate_io_parser_output_hub(predicted_category_map,
                                                                                           predicted_tokens_map,
                                                                                           start_from=1)
                parsed_response_list = ResponseParser.parse_corpus_io_parser_output(predicted_io_parser_output)
                print(f"Response parser output is: {parsed_response_list} ")

                target_batch_extracted_token = PreTrainerUtils.extract_tokens(target_batch)
                predicted_batch_extracted_token = PreTrainerUtils.extract_tokens(predicted_io_parser_output)
                bleu_score = calculate_corpus_bleu_score(
                    target_batch_extracted_token,
                    predicted_batch_extracted_token,
                    bleu_weights=get_n_gram_weights(2),
                )
                print(f"BLEU Score is: {bleu_score}")
        print("\n")


class TestClPreTrainerTraining(unittest.TestCase):
    PATH = "./saved_models/cl_pre_trainer_generative_last.pth"
    BEST_PATH = "./saved_models/cl_pre_trainer_generative_best.pth"
    accepted_loss_threshold = 0.40
    accepted_accuracy_threshold = 0.90

    def test_cl_pre_trainer_train_and_save(self):
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f'Selected Device: {device}')
        # Hyperparameters
        n_epochs = 40
        batch_size = 10 if device == torch.device("cpu") else 4
        num_heads = 8
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 2
        dropout_p = 0.1
        max_decoding_length = 16
        task_generator_indexes = [0,1,2,3]
        generator_range = 2 if device == torch.device("cpu") else 10
        number_of_batch = generator_range * len(task_generator_indexes)
        seed = 42
        add_bos_and_eos = True

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
        print(data_loader_result)
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
        print(f"Output vocabularies count: {len(output_vocabularies.keys())}\n")

        # Creating the batch and masks
        batches, masks = BatchBuilder.construct_batches_for_cl_pre_trainer_with_data_loader(
            data_loader_result,
            batch_size=batch_size,
            max_decoder_sequence_length=max_decoding_length,
            is_generative_training=True,
            add_bos_and_eos=add_bos_and_eos
        )
        print(f"Number of batch available is: {len(batches[BatchBuilder.ENCODER_IO_PARSER_OUTPUT_KEY])}\n")
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
        # Moved to CPU or GPU
        cl_pre_trainer.to(device)
        cl_pre_trainer.eval()
        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        optimizer = torch.optim.AdamW(
            cl_pre_trainer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01,
        )
        scheduler = NoamOpt(
            cl_pre_trainer.hidden_dim,
            factor=0.01,
            warmup=160,
            optimizer=optimizer,
            max_rate=0.00002
        )

        category_criterion = PreTrainerUtils.get_category_criterion(
            category_index_to_count=category_vocab_builder.index_to_count,
        )
        output_criterion_map = PreTrainerUtils.get_output_criterion_map(
            index_to_output_vocabularies=output_vocab_builder.index_to_output_vocabularies,
        )

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy, output_logits_map, execute_epoch = cl_pre_trainer_train(
            model=cl_pre_trainer,
            category_vocab_builder=category_vocab_builder,
            output_vocab_builder=output_vocab_builder,
            scheduler=scheduler,
            batches=batches,
            masks=masks,
            n_epochs=n_epochs,
            is_training=True,
            verbose_log=False,
            category_criterion=category_criterion,
            output_criterion_map=output_criterion_map,
            patience=80,
            device=device
        )

        # Saving the model...
        ClPreTrainerCheckPointManager.save_checkpoint_map(
            path=TestClPreTrainerTraining.PATH,
            epoch=n_epochs,
            model=cl_pre_trainer,
            optimizer=optimizer,
        )

        print(f"batch loss {latest_batch_loss.item()}")
        print(f"batch accuracy {latest_batch_accuracy}")
        self.assertEqual(latest_batch_loss.item() <= TestClPreTrainerTraining.accepted_loss_threshold, True)
        self.assertEqual(latest_batch_accuracy >= TestClPreTrainerTraining.accepted_accuracy_threshold, True)
        for index, output_logits_item in output_logits_map.items():
            output_loss = output_logits_item[CURRENT_BATCH_OUTPUT_LOSS]
            output_accuracy = output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY]
            self.assertEqual(output_loss.item() <= TestClPreTrainerTraining.accepted_loss_threshold, True)
            self.assertEqual(output_accuracy >= TestClPreTrainerTraining.accepted_accuracy_threshold, True)

    def test_cl_pre_trainer_model_load(self):
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        print(f'Selected Device: {device}')

        # Hyperparameters
        batch_size = 2 if device == torch.device("cpu") else 4
        num_heads = 8
        hidden_dim = 768
        ff_dim = 2048
        num_layers = 2
        dropout_p = 0.1
        max_decoding_length = 16
        task_generator_indexes = [3]
        generator_range = 2 if device == torch.device("cpu") else 10
        number_of_batch = generator_range * len(task_generator_indexes)
        seed = 42
        add_bos_and_eos = True

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
        print(data_loader_result)
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
            index_to_output_vocabularies=output_vocabularies,
        )
        # Moved to CPU or GPU
        cl_pre_trainer.to(device)
        cl_pre_trainer.eval()

        # Initialize learning rate scheduler, optimizer and loss (note: the original paper uses label smoothing)
        optimizer = torch.optim.AdamW(
            cl_pre_trainer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01,
        )

        # Load the model...
        checkpoint_map = ClPreTrainerCheckPointManager.load_checkpoint_map(
            TestClPreTrainerTraining.PATH
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

        scheduler = NoamOpt(
            cl_pre_trainer.hidden_dim,
            factor=0.01,
            warmup=160,
            optimizer=optimizer,
            max_rate=0.00002
        )

        category_criterion = PreTrainerUtils.get_category_criterion(
            category_index_to_count=category_vocab_builder.index_to_count,
        )
        output_criterion_map = PreTrainerUtils.get_output_criterion_map(
            index_to_output_vocabularies=output_vocab_builder.index_to_output_vocabularies,
        )

        # Start training and verify ~zero loss and >90% accuracy on the last batch
        latest_batch_loss, latest_batch_accuracy, output_logits_map, execute_epoch = cl_pre_trainer_train(
            model=cl_pre_trainer,
            category_vocab_builder=category_vocab_builder,
            output_vocab_builder=output_vocab_builder,
            scheduler=scheduler,
            batches=batches,
            masks=masks,
            n_epochs=start_epoch,
            is_training=False,
            verbose_log=True,
            category_criterion=category_criterion,
            output_criterion_map=output_criterion_map,
            device=device
        )

        print(f"batch loss {latest_batch_loss.item()}")
        print(f"batch accuracy {latest_batch_accuracy}")
        self.assertEqual(latest_batch_loss.item() <= TestClPreTrainerTraining.accepted_loss_threshold, True)
        self.assertEqual(latest_batch_accuracy >= TestClPreTrainerTraining.accepted_accuracy_threshold, True)
        for index, output_logits_item in output_logits_map.items():
            output_loss = output_logits_item[CURRENT_BATCH_OUTPUT_LOSS]
            output_accuracy = output_logits_item[CURRENT_BATCH_OUTPUT_ACCURACY]
            self.assertEqual(output_loss.item() <= TestClPreTrainerTraining.accepted_loss_threshold, True)
            self.assertEqual(output_accuracy >= TestClPreTrainerTraining.accepted_accuracy_threshold, True)


if __name__ == "__main__":
    unittest.main()
