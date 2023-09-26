import torch
from torch import nn

from cl_pretrainer.multi_head_attention import MultiHeadAttention


class OutputTokenBlock(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, num_heads: int, dropout_p: float):
        super().__init__()
        self.self_mha = MultiHeadAttention(hidden_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )
        # Dropout is also known as regularization
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=dropout_p)
        # Normalizing layer for propagating the token values
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        pass

    def forward(self, x: torch.FloatTensor):
        # x -> input embeddings
        # src padding mask
        # future mask
        pass

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
