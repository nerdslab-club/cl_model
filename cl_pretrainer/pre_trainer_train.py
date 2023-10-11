import unittest
from typing import List, Dict, Any

import torch
from torch import nn

from cl_pretrainer.batch_builder import BatchBuilder


def cl_pre_trainer_train(
        model: nn.Module,
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
            pass

    pass


class TestClPreTrainerTraining(unittest.TestCase):

    def test_cl_pre_trainer_train_and_save(self):
        pass

    def test_cl_pre_trainer_model_load(self):
        pass


if __name__ == "__main__":
    unittest.main()
