import torch
from torch import nn

from cl_pretrainer.multi_head_attention import MultiHeadAttention
from cl_pretrainer.rmsnorm_torch import RMSNorm


class OutputTokenBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.output_token_block_self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

        self.output_token_block_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, hidden_dim),
        ).to(self.device)
        # Dropout is also known as regularization
        self.output_token_block_dropout1 = nn.Dropout(p=dropout_p)
        self.output_token_block_dropout2 = nn.Dropout(p=dropout_p)
        # Normalizing layer for propagating the token values
        self.output_token_block_layer_norm1 = RMSNorm(hidden_dim)
        self.output_token_block_layer_norm2 = RMSNorm(hidden_dim)

        # Move to device
        self.output_token_block_dropout1.to(self.device)
        self.output_token_block_dropout2.to(self.device)
        self.output_token_block_layer_norm1.to(self.device)
        self.output_token_block_layer_norm2.to(self.device)

    def forward(self, x: torch.FloatTensor):
        """ output token block takes the x tensor which is the output of the common block.
        As common block is using both padding mask and future mask we are not using any mask in this layer.

        :param x: The output of the common block.
        :return: The embeddings for the output token -> E2.
        """
        # Multi head attention layer
        output = self.output_token_block_dropout1(self.output_token_block_self_mha.forward(x)).to(self.device)
        x = self.output_token_block_layer_norm1(x + output).to(self.device)

        # Feed forward layers
        output = self.output_token_block_dropout2(self.output_token_block_feed_forward(x))
        x = self.output_token_block_layer_norm2(x + output)
        return x

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
