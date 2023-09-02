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
    FUNTION_TOKEN_EMBEDDING = "FTE"
    TOKEN_EMBEDDING = "TE"
    POSITIONAL_EMBEDDING = "PE"
    CATEGORICAL_EMBEDDING = "CE"

    def __init__(self):
        self.initial_word_encoder = InitialWordEncoder()
        pass

    def get_embeddings_map(self, inputMap: dict, taskType: str):
        # {'token': <function MathFunctions.addition at 0x11645a8c0>,
        # 'category': {'type': 'function', 'subType': 'integer', 'subSubType': 'execute'},
        # 'position': 0},
        return {
            "FTE": "",
            "TE": "",
            "PE": "",
            "CE": ""
        }

    def get_token_embedding(self, token, category_type):
        if category_type == CategoryType.FUNCTION.value:
            token = FunctionManager.get_doc_string_of_function(token)
        return self.initial_word_encoder.get_sentence_embedding(str(token), True)

    def get_categorical_embedding(self, category_map:dict, task_type: str):
        pass

    def get_positional_embedding(self, position):
        pass

    def get_function_token_embedding(self, token, category_type):
        if category_type == CategoryType.FUNCTION.value:
            pass
        else:
            return None

    @staticmethod
    def create_embeddings_map(token_embedding, categorical_embedding, positional_embedding,
                              function_token_embeddings=None):
        return {
            EmbeddingsManager.TOKEN_EMBEDDING: token_embedding,
            EmbeddingsManager.CATEGORICAL_EMBEDDING: categorical_embedding,
            EmbeddingsManager.POSITIONAL_EMBEDDING: positional_embedding,
            EmbeddingsManager.FUNTION_TOKEN_EMBEDDING: function_token_embeddings,
        }
