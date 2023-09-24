import torch
from torch import nn


class CommonBlock(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x: torch.FloatTensor):
        pass

    def save_model(self, path: str):
        torch.save(self.state_dict(), path)

    def load_saved_model_from_path(self, path: str):
        self.load_state_dict(torch.load(path))
        self.eval()

    def load_saved_model_from_state_dict(self, state_dict: dict):
        self.load_state_dict(state_dict)
        self.eval()
