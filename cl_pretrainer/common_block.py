import torch
from torch import nn
from typing import Optional

from cl_pretrainer.multi_head_attention import MultiHeadAttention


class CommonBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        # sa => self attention
        # Multi head attention layer
        self.common_block_self_mha = MultiHeadAttention(hidden_dim, num_heads)
        # Dropout is also known as regularization
        self.common_block_dropout = nn.Dropout(p=dropout_p)
        # Normalizing layer for propagating the token values
        self.common_block_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.FloatTensor,
        src_padding_mask: Optional[torch.BoolTensor] = None,
        future_mask: Optional[torch.BoolTensor] = None,
    ):
        """Performs one decoder *block* forward pass given the previous block's output and optional attention masks.
        N = batch size
        S = source sequence length
        E = embedding dimensionality

        :param x: Tensor containing the output of the previous encoder block. Shape: (N, S, E)
        :param src_padding_mask: An attention mask to ignore pad-tokens in the source input. Shape (N, S)
        :param future_mask: An attention mask to ignore future-tokens in the target input. Shape (S, S)
        :return: Updated intermediate decoder common block (contextualized) token embeddings. Shape: (N, S, E)
        """
        output = self.common_block_dropout(
            self.common_block_self_mha.forward(
                x, src_padding_mask=src_padding_mask, future_mask=future_mask
            )
        )
        x = self.common_block_layer_norm(x + output)
        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
