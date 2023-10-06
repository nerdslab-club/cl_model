import torch
from torch import nn


class CategoryMapClassificationHead(nn.Module):
    def __init__(self, hidden_dim: int, ff_dim: int, vocab_size: int, dropout_p: float):
        super().__init__()

        self.category_map_classification_head_feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.LeakyReLU(),
            nn.Linear(ff_dim, hidden_dim),
        )

        # Dropout is also known as regularization
        self.category_map_classification_head_dropout = nn.Dropout(p=dropout_p)
        # Normalizing layer for propagating the token values
        self.category_map_classification_head_layer_norm = nn.LayerNorm(hidden_dim)

        # Projecting the hidden dimension into vocabulary size,
        # so that we can use softmax and find specific word probability.
        self.category_map_classification_head_output_layer = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, e_one: torch.Tensor):
        # Feed forward layers
        output = self.category_map_classification_head_dropout(
            self.category_map_classification_head_feed_forward(e_one),
        )
        e_one = self.category_map_classification_head_output_layer(e_one + output)

        # Linear layer, output shape(batch_size, sequence_length, category_vocab_size)
        logits = self.output_layer(e_one)

        # Softmax layer, output shape(batch_size, sequence_length)
        category_probability = logits.argmax(dim=-1)
        return category_probability

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
