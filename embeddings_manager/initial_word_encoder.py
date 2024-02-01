import torch
from sentence_transformers import SentenceTransformer


class InitialWordEncoder:
    def __init__(self):
        self.sentence_model = SentenceTransformer("all-mpnet-base-v2")
        # Moved to CPU or Cuda
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def get_sentence_encoder_model(self):
        return self.sentence_model

    def get_sentence_embedding(self, sentence: str, convert_to_tensor: bool, show_progress_bar: bool):
        sentence_embedding = self.sentence_model.encode(
            sentence, show_progress_bar=show_progress_bar, convert_to_tensor=convert_to_tensor, device=self.device
        )
        return sentence_embedding

    def get_list_of_sentence_embedding(
        self, sentence_list: list[str], convert_to_tensor: bool
    ):
        sentence_embeddings = self.sentence_model.encode(
            sentence_list, show_progress_bar=True, convert_to_tensor=convert_to_tensor, device=self.device
        )
        return sentence_embeddings
