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
        self.category_map_block_dropout4 = nn.Dropout(p=dropout_p)
        # Normalizing layer for propagating the token values
        self.category_map_block_layer_norm1 = nn.LayerNorm(hidden_dim)
        self.category_map_block_layer_norm2 = nn.LayerNorm(hidden_dim)
        self.category_map_block_layer_norm3 = nn.LayerNorm(hidden_dim)
        self.category_map_block_layer_norm4 = nn.LayerNorm(hidden_dim)

    def forward(self,
                x: torch.FloatTensor,
                function_param_token_mask: Optional[tuple[bool, Tensor]],
                future_mask: Optional[torch.BoolTensor] = None,
                ):
        """Perform the category map block forward pass given the common block output with function param token mask
        with optional attention masks

        :param x:Tensor containing the output of the previous decoder block. Shape: (N, S, E)
        :param function_param_token_mask: If it's a function param then it will have values,
         like (True, IFT) otherwise, (False, None).
        :param future_mask: An attention mask to ignore future-tokens in the target input. Shape (S, S)
        :return: Updated intermediate decoder category map block token embeddings. Shape: (N, S, E)
        """
        # initial_function_embeddings need to be specific based on token
        # x -> input embeddings
        # future mask
        # is_function_param_token_mask

        # Self attention (with future masking during training)
        output = self.category_map_block_dropout1(self.category_map_block_self_mha.forward(x, future_mask=future_mask))
        x = self.category_map_block_layer_norm1(x + output)

        # TODO Replace cross attention tokens using cross_mha in the x.
        x = self.category_map_block_layer_norm2(x + output)

        # Feed forward layers
        output = self.category_map_block_dropout4(self.category_map_block_feed_forward(x))
        x = self.category_map_block_layer_norm4(x + output)
        return x
        pass

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
