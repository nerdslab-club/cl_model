import torch
from torch.optim import Adam

from cl_pretrainer.cl_pre_trainer import ClPreTrainer


class ClPreTrainerCheckPointManager:
    """
    Manage the saved checkpoints for each nn.module
    """

    EPOCH = "epoch"
    CL_PRE_TRAINER_STATE = "cl_pre_trainer_state"
    OPTIM_STATE = "optim_state"
    CATEGORY_MAP_DECODER_STATE = "category_map_decoder_state"
    CATEGORY_MAP_CLASSIFICATION_HEAD_STATE = "category_map_classification_map_head_state"
    OUTPUT_TOKEN_DECODER_STATE = "output_token_decoder_state"
    CATEGORY_ROUTER_STATE = "category_router_state"
    CATEGORY_MAP_DECODER_BLOCKS_STATE = "category_map_decoder_blocks_state"
    OUTPUT_TOKEN_DECODER_BLOCKS_STATE = "output_token_decoder_blocks_state"

    @staticmethod
    def save_checkpoint_map(
        path: str, epoch: int, model: ClPreTrainer, optimizer: Adam
    ):
        torch.save(
            ClPreTrainerCheckPointManager.__create_checkpoint_map(
                epoch,
                model.state_dict(),
                optimizer.state_dict(),
                model.category_map_decoder.state_dict(),
                model.category_map_classification_head.state_dict(),
                model.output_token_decoder.state_dict(),
                model.category_router.state_dict(),
                model.category_map_decoder.category_map_decoder_blocks.state_dict(),
                model.output_token_decoder.output_token_decoder_blocks.state_dict(),
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
        cl_pre_trainer_state: dict,
        optim_state: dict,
        cl_pre_trainer_category_map_decoder_state: dict,
        cl_pre_trainer_category_map_classification_head_state: dict,
        cl_pre_trainer_output_token_decoder_state: dict,
        cl_pre_trainer_category_router_state: dict,
        cl_pre_trainer_category_map_decoder_blocks_state: dict,
        cl_pre_trainer_output_token_decoder_blocks_state: dict,
    ) -> dict:
        return {
            ClPreTrainerCheckPointManager.EPOCH: epoch,
            ClPreTrainerCheckPointManager.CL_PRE_TRAINER_STATE: cl_pre_trainer_state,
            ClPreTrainerCheckPointManager.OPTIM_STATE: optim_state,
            ClPreTrainerCheckPointManager.CATEGORY_MAP_DECODER_STATE: cl_pre_trainer_category_map_decoder_state,
            ClPreTrainerCheckPointManager.CATEGORY_MAP_CLASSIFICATION_HEAD_STATE: cl_pre_trainer_category_map_classification_head_state,
            ClPreTrainerCheckPointManager.OUTPUT_TOKEN_DECODER_STATE: cl_pre_trainer_output_token_decoder_state,
            ClPreTrainerCheckPointManager.CATEGORY_ROUTER_STATE: cl_pre_trainer_category_router_state,
            ClPreTrainerCheckPointManager.CATEGORY_MAP_DECODER_BLOCKS_STATE: cl_pre_trainer_category_map_decoder_blocks_state,
            ClPreTrainerCheckPointManager.OUTPUT_TOKEN_DECODER_BLOCKS_STATE: cl_pre_trainer_output_token_decoder_blocks_state,
        }
