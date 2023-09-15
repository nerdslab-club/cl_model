from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import RobertaConfig
from cl_data.function_representation.src.functions_manager import FunctionManager as fm
import types
import torch.nn.functional as functional
import torch


class InitialFunctionEncoder:
    """
    This class is responsible for creating
    FTE -> Function Token Embeddings (300 * 768)
    IFE -> Initial Function Embeddings (2 * 768) = (FDE + FSE)
    FDE -> Function Documentation Embeddings 768 TODO
    FSE -> Function Similarity Embeddings 768 (Siamese Network) TODO
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.config = RobertaConfig.from_pretrained(
            "microsoft/graphcodebert-base", output_hidden_states=True
        )
        self.model = AutoModelForMaskedLM.from_pretrained(
            "microsoft/graphcodebert-base", config=self.config
        )
        self.function_manager = fm()

    def _get_raw_function_output(self, function_name: str):
        """Calculate the output from the function input

        :param function_name: Name of the function.
        :return: The output which include hidden states and logits
        """
        function_ref = self.function_manager.get_name_to_reference().get(function_name)
        func_raw_str = self.function_manager.get_function_as_string_without_doc_string(
            function_ref
        )
        inputs = self.tokenizer(func_raw_str, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs

    def _get_raw_function_embedding_from_name(self, function_name: str):
        """Calculate the hidden states embedding for the function tokens

        :param function_name: Name of the function.
        :return: hidden states embedding of size [1 * n * 768]
        """
        function_ref = self.function_manager.get_name_to_reference().get(function_name)
        func_raw_str = self.function_manager.get_function_as_string_without_doc_string(
            function_ref
        )

        inputs = self.tokenizer(func_raw_str, return_tensors="pt")

        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states

    def get_perfect_function_embedding_from_name(
        self, function_name: str, max_length=300
    ):
        """Reshape the last hidden state of the embedding into [max_length, 768] tensor

        :param function_name: Name of the function.
        :param max_length: Max length for token that is supported.
        :return: Last hidden state embedding of size [n * 768]
        """
        hidden_states_embedding = self._get_raw_function_embedding_from_name(
            function_name
        )

        # Pad the tensor to the desired shape [300, 768]
        padded_tensor = functional.pad(
            hidden_states_embedding[0],
            (0, 0, 0, max_length - hidden_states_embedding[0].shape[1]),
        )
        return padded_tensor

    def _get_raw_function_embedding(self, function_ref: types):
        """Calculate the hidden states embedding for the function tokens

        :param function_ref: Reference to the function.
        :return: hidden states embedding of size [1 * n * 768]
        """
        func_raw_str = self.function_manager.get_function_as_string_without_doc_string(
            function_ref
        )

        inputs = self.tokenizer(func_raw_str, return_tensors="pt")

        outputs = self.model(**inputs)
        hidden_states = outputs.hidden_states
        return hidden_states

    def get_function_token_embedding(self, function_ref: types, max_length=300):
        """Reshape the last hidden state of the embedding into [max_length, 768] tensor

        :param function_ref: Reference to the function.
        :param max_length: Max length for token that is supported.
        :return: Last hidden state embedding of size [n * 768]
        """
        hidden_states_embedding = self._get_raw_function_embedding(function_ref)

        # Pad the tensor to the desired shape [300, 768]
        padded_tensor = functional.pad(
            hidden_states_embedding[0],
            (0, 0, 0, max_length - hidden_states_embedding[0].shape[1]),
        )
        # reshaped_tensor = padded_tensor.squeeze()
        return padded_tensor

    def get_logits(self, function_name: str, max_length=300):
        """Calculate the logits for the given function.

        :param function_name: Name of the function.
        :param max_length: Max length for token that is supported.
        :return: Logits embedding.
        """
        output = self._get_raw_function_output(function_name)
        logits = output.logits
        # Pad the tensor to the desired shape [300, 768]
        padded_tensor = functional.pad(logits, (0, 0, 0, max_length - logits.shape[1]))
        return padded_tensor

    def get_siamese_network_embedding(self):
        pass

    @staticmethod
    def get_shape(embedding):
        """Print and return the shape and length of an embedding

        :param embedding: The embedding tensor whom shape and length is to be calculated
        :return: Tuple as (shape, length)
        """
        shape = embedding.shape
        length = len(embedding)
        print(shape)
        print(length)
        return shape, length

    @staticmethod
    def get_final_initial_function_embedding(
        documentation_embedding, siamese_embedding, dim=1
    ):
        """Concat documentation_embedding-> [1*768] and siamese_embedding-> [1*768] to get final function embedding-> [2*768]

        :param dim: dim=1 will return tensor of [1, 1536] while dim=0 will return tensor of [2,768]
        :param documentation_embedding: This tensor is retrieved from sentence encoder using function documentation
        :param siamese_embedding: This tensor is retrieved from siamese network using function token embeddings
        :return: final function embedding tensor of size [2*768]
        """
        concatenated_embedding = torch.cat(
            (documentation_embedding, siamese_embedding), dim=dim
        )
        return concatenated_embedding
