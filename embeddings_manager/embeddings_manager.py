from embeddings_manager.Initial_function_encoder import InitialFunctionEncoder
from embeddings_manager.alibibi_positional_encoder import ALiBiBiEncoder
from embeddings_manager.category_and_task_encoder import CategoryAndTaskEncoder
from embeddings_manager.initial_word_encoder import InitialWordEncoder
from cl_data.src.constants import (
    PretrainTasks,
    TaskTypes,
    CategoryType,
    CategorySubType,
    CategorySubSubType,
    Constants,
)
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

    def __init__(self):
        self.initial_word_encoder = InitialWordEncoder()
        self.initial_function_encoder = InitialFunctionEncoder()
        self.aLiBiBi_encoder = ALiBiBiEncoder()
        self.category_and_task_encoder = CategoryAndTaskEncoder()

    def get_embeddings_map(self, token, category_type, category_map, task_type, position):
        # {'token': <function MathFunctions.addition at 0x11645a8c0>,
        # 'category': {'type': 'function', 'subType': 'integer', 'subSubType': 'execute'},
        # 'position': 0},
        token_embedding = self.get_token_embedding(token, category_type)
        category_and_task_embedding = self.get_category_and_task_embedding(category_map, task_type)
        combined_embedding = self.get_combined_embedding(token_embedding, category_and_task_embedding)
        alibibi_embedding = self.get_alibibi_embedding(position)
        frequency_embedding = self.get_frequency_embedding(category_and_task_embedding)
        function_token_embedding = self.get_function_token_embedding(token, category_type)

        return EmbeddingsManager.create_embeddings_map(
            token_embedding,
            category_and_task_embedding,
            combined_embedding,
            alibibi_embedding,
            frequency_embedding,
            function_token_embedding
        )

    def get_token_embedding(self, token, category_type):
        if category_type == CategoryType.FUNCTION.value:
            token = FunctionManager.get_doc_string_of_function(token)
        return self.initial_word_encoder.get_sentence_embedding(str(token), True)

    def get_category_and_task_embedding(self, category_map: dict, task_type: str):
        return self.category_and_task_encoder.categorical_encoding(category_map, task_type)

    def get_combined_embedding(self, token_embedding, category_and_task_embedding):
        return token_embedding + category_and_task_embedding

    def get_alibibi_embedding(self, position: int):
        return self.aLiBiBi_encoder.generate_distance_matrix(position)

    def get_frequency_embedding(self, category_and_task_embedding):
        return self.category_and_task_encoder.frequency_encoding(category_and_task_embedding)

    def get_function_token_embedding(self, token, category_type):
        if category_type == CategoryType.FUNCTION.value:
            function_name = FunctionManager.get_function_as_string(token)
            return self.initial_function_encoder.get_perfect_function_embedding_from_name(function_name)
        else:
            return None

    @staticmethod
    def create_embeddings_map(
        token_embedding,
        alibibi_embedding,
        combined_embedding,
        category_embedding,
        frequency_embedding,
        function_token_embeddings=None,
    ):
        return {
            EmbeddingsManager.TOKEN_EMBEDDING: token_embedding,
            EmbeddingsManager.ALIBIBI_EMBEDDING: alibibi_embedding,
            EmbeddingsManager.COMBINED_EMBEDDING: combined_embedding,
            EmbeddingsManager.CATEGORY_AND_TASK_EMBEDDING: category_embedding,
            EmbeddingsManager.FREQUENCY_EMBEDDING: frequency_embedding,
            EmbeddingsManager.FUNTION_TOKEN_EMBEDDING: function_token_embeddings,
        }
