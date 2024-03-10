import torch
from torch.optim import Adam

from category_router.category_router import CategoryRouter
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
    OUTPUT_TOKEN_CLASSIFICATION_HEADS_STATE = "output_token_classification_map_heads_state"
    EMBEDDINGS_LAYER_STATE = "embeddings_layer_state"

    @staticmethod
    def __get_category_router_checkpoint_map(
        category_router: CategoryRouter,
    ):
        output_token_classification_head_state_map = {}
        for index, route in category_router.index_to_route.items():
            output_classification_head = route[CategoryRouter.ROUTE_CLASSIFICATION_HEAD]
            output_token_classification_head_state_map[index] = output_classification_head.state_dict()
        return output_token_classification_head_state_map

    @staticmethod
    def save_checkpoint_map(
        path: str, epoch: int, model: ClPreTrainer, optimizer: any
    ):
        cl_pre_trainer_output_token_classification_heads_state = \
            ClPreTrainerCheckPointManager.__get_category_router_checkpoint_map(model.category_router)

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
                cl_pre_trainer_output_token_classification_heads_state,
                model.embedding_layer.state_dict(),
            ),
            path,
        )

    @staticmethod
    def load_checkpoint_map(path: str) -> dict:
        device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        return torch.load(path, map_location=device)

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
        cl_pre_trainer_output_token_classification_heads_state: dict,
        cl_pre_trainer_embeddings_layer_state: dict,
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
            ClPreTrainerCheckPointManager.OUTPUT_TOKEN_CLASSIFICATION_HEADS_STATE: cl_pre_trainer_output_token_classification_heads_state,
            ClPreTrainerCheckPointManager.EMBEDDINGS_LAYER_STATE: cl_pre_trainer_embeddings_layer_state
        }
