import torch
from torch.optim import Adam

from cl_pretrainer.transformer import Transformer


class CheckPointManager:
    """
    Manage the saved checkpoints for each nn.module
    """

    EPOCH = "epoch"
    TRANSFORMER_STATE = "transformer_state"
    OPTIM_STATE = "optim_state"
    ENCODER_STATE = "encoder_state"
    DECODER_STATE = "decoder_state"
    ENCODER_BLOCKS_STATE = "encoder_blocks_state"
    DECODER_BLOCKS_STATE = "decoder_blocks_state"

    @staticmethod
    def save_checkpoint_map(
        path: str, epoch: int, transformer: Transformer, optimizer: Adam
    ):
        torch.save(
            CheckPointManager.__create_checkpoint_map(
                epoch,
                transformer.state_dict(),
                optimizer.state_dict(),
                transformer.encoder.state_dict(),
                transformer.decoder.state_dict(),
                transformer.encoder.encoder_blocks.state_dict(),
                transformer.decoder.decoder_blocks.state_dict(),
            ),
            path,
        )

    @staticmethod
    def load_checkpoint_map(path: str) -> dict:
        return torch.load(path)

    @staticmethod
    def get_checkpoint_item(checkpoint_map: dict, item_key: str):
        return checkpoint_map.get(item_key)

    @staticmethod
    def __create_checkpoint_map(
        epoch: int,
        transformer_state: dict,
        optim_state: dict,
        transformer_encoder_state: dict,
        transformer_decoder_state: dict,
        transformer_encoder_blocks_state: dict,
        transformer_decoder_blocks_state: dict,
    ) -> dict:
        return {
            CheckPointManager.EPOCH: epoch,
            CheckPointManager.TRANSFORMER_STATE: transformer_state,
            CheckPointManager.OPTIM_STATE: optim_state,
            CheckPointManager.ENCODER_STATE: transformer_encoder_state,
            CheckPointManager.DECODER_STATE: transformer_decoder_state,
            CheckPointManager.ENCODER_BLOCKS_STATE: transformer_encoder_blocks_state,
            CheckPointManager.DECODER_BLOCKS_STATE: transformer_decoder_blocks_state,
        }
