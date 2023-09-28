from torch import Tensor

from embeddings_manager.Initial_function_encoder import InitialFunctionEncoder
from embeddings_manager.alibibi_positional_encoder import ALiBiBiEncoder
from embeddings_manager.category_and_task_encoder import CategoryAndTaskEncoder
from embeddings_manager.initial_word_encoder import InitialWordEncoder
from cl_data.src.constants import CategoryType, Constants
from cl_data.function_representation.src.functions_manager import FunctionManager


class EmbeddingsManager:
    # if function documentation token == true else == false
    FUNTION_TOKEN_EMBEDDING = "FTE"
    # Initial_word_encoder
    TOKEN_EMBEDDING = "TE"
    # Version of positional embedding
    ALIBIBI_EMBEDDING = "ALIBIBI"
    # Category and task encoder
    CATEGORY_AND_TASK_EMBEDDING = "CTE"
    # TOKEN_EMBEDDING + CATEGORY_AND_TASK_EMBEDDING
    COMBINED_EMBEDDING = "TECTE"
    # Frequency embedding
    FREQUENCY_EMBEDDING = "FE"
    TOKEN = "T"
    CATEGORY_MAP = "CM"
    POSITION = "P"
    TASK_TYPE = "TT"

    def __init__(
            self,
            batch_size: int,
            n_heads: int,
            max_sequence_length: int,
            with_mask: bool):
        self.initial_word_encoder = InitialWordEncoder()
        self.initial_function_encoder = InitialFunctionEncoder()
        self.aLiBiBi_encoder = ALiBiBiEncoder()
        self.category_and_task_encoder = CategoryAndTaskEncoder()
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.max_sequence_length = max_sequence_length
        self.with_mask = with_mask

    def get_batch_embeddings_map(self, input_map_list: list[dict], task_type: str) -> list[dict]:
        embedding_map_list = []
        for input_map in input_map_list:
            current_embeddings_map = self.get_embeddings_map(
                input_map[Constants.TOKEN],
                input_map[Constants.CATEGORY],
                input_map[Constants.POSITION],
                task_type,
            )
            embedding_map_list.append(current_embeddings_map)
        return embedding_map_list

    def get_embeddings_map(
            self,
            token: any,
            category_map: dict,
            position: int,
            task_type: str,
    ) -> dict:
        # {
        # 'token': <function MathFunctions.addition at 0x11645a8c0>,
        # 'category': {
        #   'type': 'function',
        #   'subType': 'integer',
        #   'subSubType': 'execute'
        #  },
        # 'position': 0
        # }
        category_type: str = category_map.get(
            Constants.CATEGORY_TYPE,
            CategoryType.WORD.value,
        )
        token_embedding = self.get_token_embedding(token, category_type)
        category_and_task_embedding = self.get_category_and_task_embedding(
            category_map,
            task_type,
        )
        combined_embedding = self.get_combined_embedding(
            token_embedding,
            category_and_task_embedding,
        )
        alibibi_embedding = self.get_alibibi_embedding(
            self.batch_size,
            self.n_heads,
            self.max_sequence_length,
            self.with_mask,
        )
        frequency_embedding = self.get_frequency_embedding(category_and_task_embedding)
        function_token_embedding = self.get_function_token_embedding(
            token,
            category_type,
        )
        return EmbeddingsManager.create_embeddings_map(
            token_embedding,
            category_and_task_embedding,
            combined_embedding,
            alibibi_embedding,
            frequency_embedding,
            token,
            category_map,
            position,
            task_type,
            function_token_embedding,
        )

    def get_token_embedding(self, token: any, category_type: str) -> Tensor:
        if category_type == CategoryType.FUNCTION.value:
            token = FunctionManager.get_doc_string_of_function(token)
        return self.initial_word_encoder.get_sentence_embedding(str(token), True)

    def get_category_and_task_embedding(
            self,
            category_map: dict,
            task_type: str,
    ) -> Tensor:
        return self.category_and_task_encoder.categorical_encoding(
            category_map, task_type
        )

    def get_combined_embedding(
            self,
            token_embedding: Tensor,
            categorical_embedding: Tensor,
    ) -> Tensor:
        return self.category_and_task_encoder.get_combined_embedding(
            token_embedding, categorical_embedding
        )

    def get_alibibi_embedding(
            self,
            batch_size: int,
            n_heads: int,
            max_sequence_length: int,
            with_mask: bool,
    ) -> Tensor:
        return self.aLiBiBi_encoder.get_alibi_biases(
            batch_size=batch_size,
            n_heads=n_heads,
            sequence_length=max_sequence_length,
            with_mask=with_mask,
        )

    def get_frequency_embedding(self, category_and_task_embedding: Tensor) -> Tensor:
        return self.category_and_task_encoder.frequency_encoding(
            category_and_task_embedding
        )

    def get_function_token_embedding(self, token: any, category_type) -> Tensor | None:
        if category_type == CategoryType.FUNCTION.value:
            return self.initial_function_encoder.get_perfect_function_token_embedding(
                token
            )
        else:
            return None

    @staticmethod
    def create_embeddings_map(
            token_embedding: Tensor,
            alibibi_embedding: Tensor,
            combined_embedding: Tensor,
            category_embedding: Tensor,
            frequency_embedding: Tensor,
            token: any,
            category_map: dict,
            position: int,
            task_type: str,
            function_token_embeddings=None | Tensor,
    ) -> dict:
        return {
            EmbeddingsManager.TOKEN_EMBEDDING: token_embedding,
            EmbeddingsManager.ALIBIBI_EMBEDDING: alibibi_embedding,
            EmbeddingsManager.COMBINED_EMBEDDING: combined_embedding,
            EmbeddingsManager.CATEGORY_AND_TASK_EMBEDDING: category_embedding,
            EmbeddingsManager.FREQUENCY_EMBEDDING: frequency_embedding,
            EmbeddingsManager.FUNTION_TOKEN_EMBEDDING: function_token_embeddings,
            EmbeddingsManager.TOKEN: token,
            EmbeddingsManager.CATEGORY_MAP: category_map,
            EmbeddingsManager.POSITION: position,
            EmbeddingsManager.TASK_TYPE: task_type,
        }
