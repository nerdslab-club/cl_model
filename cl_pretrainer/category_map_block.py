from typing import Optional

import torch
from torch import nn, Tensor

from cl_pretrainer.multi_head_attention import MultiHeadAttention


class CategoryMapBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.category_map_block_cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.category_map_block_self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.category_map_block_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        # Dropout is also known as regularization
        self.category_map_block_dropout1 = nn.Dropout(p=dropout_p)
        self.category_map_block_dropout2 = nn.Dropout(p=dropout_p)
        self.category_map_block_dropout3 = nn.Dropout(p=dropout_p)
        # Normalizing layer for propagating the token values
        self.category_map_block_layer_norm1 = nn.LayerNorm(hidden_dim)
        self.category_map_block_layer_norm2 = nn.LayerNorm(hidden_dim)
        self.category_map_block_layer_norm3 = nn.LayerNorm(hidden_dim)

    def forward(self,
                x: torch.FloatTensor,
                function_param_token_mask: Optional[tuple[bool, Tensor]],
                ):
        """Perform the category map block forward pass given the common block output with function param token mask.
        As common block is using both padding mask and future mask we are not using any mask in this layer.

        :param x:Tensor containing the output of the previous decoder block. Shape: (N, S, E)
        :param function_param_token_mask: If it's a function param then it will have values,
         like (True, IFT) otherwise, (False, None).
        :return: Updated intermediate decoder category map block token embeddings. Shape: (N, S, E)
        """

        # Self attention (with future masking during training)
        output = self.category_map_block_dropout1(self.category_map_block_self_mha.forward(x))
        x = self.category_map_block_layer_norm1(x + output)

        # TODO Replace cross attention tokens using cross_mha in the x.
        output = self.category_map_block_dropout2(self.update_function_params_token_using_cross_attention())
        x = self.category_map_block_layer_norm2(x + output)

        # Feed forward layers
        output = self.category_map_block_dropout3(self.category_map_block_feed_forward(x))
        x = self.category_map_block_layer_norm3(x + output)
        return x

    def update_function_params_token_using_cross_attention(self) -> Tensor:
        # TODO complete this function
        pass

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
