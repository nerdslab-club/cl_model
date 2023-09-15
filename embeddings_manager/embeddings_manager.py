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

    def get_embeddings_map(self, inputMap: dict, taskType: str):
        # {'token': <function MathFunctions.addition at 0x11645a8c0>,
        # 'category': {'type': 'function', 'subType': 'integer', 'subSubType': 'execute'},
        # 'position': 0},
        return EmbeddingsManager.create_embeddings_map()

    def get_token_embedding(self, token, category_type):
        if category_type == CategoryType.FUNCTION.value:
            token = FunctionManager.get_doc_string_of_function(token)
        return self.initial_word_encoder.get_sentence_embedding(str(token), True)

    def get_category_and_task_embedding(self, category_map: dict, task_type: str):
        pass

    def get_combined_embedding(self, token_embedding, category_and_task_embedding):
        pass

    def get_alibibi_embedding(self, position: int):
        pass

    def get_frequency_embedding(self, category_and_task_embedding):
        pass

    def get_function_token_embedding(self, token, category_type):
        if category_type == CategoryType.FUNCTION.value:
            pass
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
