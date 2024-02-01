import torch
from torch import nn, Tensor

from cl_pretrainer.multi_head_attention import MultiHeadAttention
from cl_pretrainer.rmsnorm_torch import RMSNorm


class CategoryMapBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.category_map_block_cross_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.category_map_block_self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.category_map_block_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, hidden_dim),
        ).to(self.device)

        # Dropout is also known as regularization
        self.category_map_block_dropout1 = nn.Dropout(p=dropout_p)
        self.category_map_block_dropout2 = nn.Dropout(p=dropout_p)
        # Normalizing layer for propagating the token values
        self.category_map_block_layer_norm1 = RMSNorm(hidden_dim)
        self.category_map_block_layer_norm2 = RMSNorm(hidden_dim)

        # Move to device
        self.category_map_block_dropout1.to(self.device)
        self.category_map_block_dropout2.to(self.device)
        self.category_map_block_layer_norm1.to(self.device)
        self.category_map_block_layer_norm2.to(self.device)

    def forward(self,
                x: torch.FloatTensor,
                function_param_token_infos: list[dict],
                ):
        """
        Perform the category map block forward pass given the common block output with function param token mask.
        As common block is using both padding mask and future mask we are not using any mask in this layer.

        :param x:Tensor containing the output of the previous decoder block. Shape: (N, S, E)
        :param function_param_token_infos: It is a list of function param token info, each info is a map. ie.
        {
            "start": (r,c),
            "end": (r,c),
            "encoder_hidden_state": Tensor, # Embeddings of the function in question found using initial function encoder
            "token": Tensor, # Embedding of category map token of function params.
        };
        :return: Updated intermediate decoder category map block token embeddings. Shape: (N, S, E)
        """
        # Multi Head Self attention
        output = self.category_map_block_self_mha.forward(x).to(self.device)
        output = self.update_function_params_token_using_cross_attention(output, function_param_token_infos).to(self.device)

        output = self.category_map_block_dropout1(output)
        x = self.category_map_block_layer_norm1(x + output)

        # Feed forward layers
        output = self.category_map_block_dropout2(self.category_map_block_feed_forward(x))
        x = self.category_map_block_layer_norm2(x + output)
        return x

    def update_function_params_token_using_cross_attention(
            self,
            x: torch.FloatTensor,
            function_param_token_infos: list[dict],
    ) -> Tensor:
        """
        Calculate cross attention for funtion params and replace original tensors slice with that.

        :param x: original tensor where we will replace some value with cross attention tensors
        :param function_param_token_infos: It is a list of function param token info, each info is a map. ie.
        {
            "start": (r,c),
            "end": (r,c),
            "encoder_hidden_state": Tensor, # Embeddings of the function in question found using initial function encoder
            "token": Tensor, # Embedding of category map token of function params.
        };
        :return: Modified tensor after replacing with cross attention values.
        """
        for param_info in function_param_token_infos:
            start_row, start_col = param_info["start"]
            end_row, end_col = param_info["end"]
            encoder_hidden_state = param_info["encoder_hidden_state"]
            token_tensor = param_info["token_tensors"]
            cross_attention_output = self.category_map_block_cross_mha.forward(
                token_tensor.unsqueeze(0),
                encoder_hidden_state,
            )
            x[start_row, start_col:end_col, :] = cross_attention_output
        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
