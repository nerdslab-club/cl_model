import types

import torch
from torch import Tensor, nn
from torch.nn import Embedding

from cl_pretrainer.batch_builder import BatchBuilder
from cl_pretrainer.tokenizer import Tokenizer
from embeddings_manager.initial_function_encoder import InitialFunctionEncoder
from embeddings_manager.alibibi_positional_encoder import ALiBiBiEncoder
from embeddings_manager.category_and_task_encoder import CategoryAndTaskEncoder
from embeddings_manager.initial_word_encoder import InitialWordEncoder
from cl_data.src.constants import CategoryType, Constants, CategorySubSubType
from cl_data.function_representation.src.functions_manager import FunctionManager
from vocabulary_builder.category_vocabulary_builder import CategoryVocabBuilder
from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


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
            with_mask: bool,
            embedding_layer: Embedding,
            output_vocab_builder: OutputVocabBuilder,
            use_our_tokenizer: bool = False,
    ):
        self.initial_word_encoder = InitialWordEncoder()
        self.initial_function_encoder = InitialFunctionEncoder()
        self.aLiBiBi_encoder = ALiBiBiEncoder()
        self.category_and_task_encoder = CategoryAndTaskEncoder()
        self.batch_size = batch_size
        self.n_heads = n_heads
        self.max_sequence_length = max_sequence_length
        self.with_mask = with_mask
        self.embedding_layer = embedding_layer
        self.output_vocab_builder = output_vocab_builder
        self.use_our_tokenizer = use_our_tokenizer
        self.device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    def get_batch_combined_embeddings(
            self,
            batch_io_parser_output: list[list[dict]],
            task_type: list[str],
    ) -> Tensor:
        """Batch of the io parser output, it converts every io parser item into it's combined embedding

        use_our_tokenizer: Flag for whether to use our own tokenizer
        output_vocab_builder: This is the output vocab builder instance
        :param batch_io_parser_output: batch of io_parser_output
        :param task_type: Type of task. ie: func_to_nl_translation.
        :return: Return the combined embeddings of list of sentence.
        Shape [ len(batch_io_parser_output), len(io_parser_output), 768]
        """
        batch_item_tensors = torch.empty(
            (0, len(batch_io_parser_output[0]), 768), dtype=torch.float32, device=self.device
        )
        for index, io_parser_output in enumerate(batch_io_parser_output):
            item_tensors = self.get_sentence_combined_embeddings(
                io_parser_output,
                task_type[index],
            )
            batch_item_tensors = torch.cat(
                (batch_item_tensors, item_tensors.unsqueeze(0)), dim=0
            )
        return batch_item_tensors

    def get_sentence_combined_embeddings(
            self,
            io_parser_output: list[dict],
            task_type: str,
    ) -> Tensor:
        """Given the io parser output it convert every io parser item into it's combined embedding

        use_our_tokenizer: Flag for whether to use our own tokenizer
        output_vocab_builder: This is the output vocab builder instance
        :param io_parser_output: input string -> io parser
        [
             {
                "token":126,
                "category":{
                   "type":"integer",
                   "subType":"default",
                   "subSubType":"none"
                },
                "position":0
             },
             {
                "token":"plus",
                "category":{
                   "type":"word",
                   "subType":"default",
                   "subSubType":"none"
                },
                "position":1
             },
         ]
        :param task_type: Type of task. ie: func_to_nl_translation.
        :return: Return the combined embeddings of sentence. Shape [ len(io_parser_output), 768]
        """
        item_tensors = torch.empty((0, 768), dtype=torch.float32, device=self.device)
        for io_parser_output_item in io_parser_output:
            token: any = io_parser_output_item[Constants.TOKEN]
            category_map: dict = io_parser_output_item[Constants.CATEGORY]
            category_type: str = category_map[Constants.CATEGORY_TYPE]

            token_embedding = self.get_token_embedding(
                token,
                category_type,
                io_parser_output_item,
            )
            category_and_task_embedding = self.get_category_and_task_embedding(
                category_map,
                task_type,
            )
            combined_embedding = self.get_combined_embedding(
                token_embedding,
                category_and_task_embedding,
            )
            item_tensors = torch.cat(
                (item_tensors, combined_embedding.unsqueeze(0)), dim=0
            )
        return item_tensors

    def get_batch_combined_embeddings_with_mask(
            self, batch_io_parser_output: list[list[dict]], task_type: list[str]
    ) -> tuple[Tensor, Tensor, list[list[Tensor]]]:
        """Batch of the io parser output, it converts every io parser item into
        it's combined embedding and cross attention mask and batch of encoder hidden state

        :param batch_io_parser_output: batch of io_parser_output
        :param task_type: Type of task. ie: func_to_nl_translation.
        :return: Return the combined embeddings of list of sentence.
        Shape [ len(batch_io_parser_output), len(io_parser_output), 768]
        and cross attention mask. Shape[len(batch_io_parser_output), len(io_parser_output)]
        and batch of encoder hidden state.
        """
        batch_item_tensors = torch.empty(
            (0, len(batch_io_parser_output[0]), 768), dtype=torch.float32, device=self.device
        )
        batch_function_param_mask = torch.empty(
            (0, len(batch_io_parser_output[0])), dtype=torch.bool, device=self.device
        )
        batch_encoder_hidden_states = []
        for index, io_parser_output in enumerate(batch_io_parser_output):
            item_tensors, mask_tensors, list_of_encoder_hidden_states = self.get_sentence_combined_embeddings_with_mask(
                io_parser_output, task_type[index]
            )
            batch_item_tensors = torch.cat(
                (batch_item_tensors, item_tensors.unsqueeze(0)), dim=0
            )
            batch_function_param_mask = torch.cat(
                (batch_function_param_mask, mask_tensors.unsqueeze(0)), dim=0
            )
            batch_encoder_hidden_states.append(list_of_encoder_hidden_states)
        return batch_item_tensors, batch_function_param_mask, batch_encoder_hidden_states

    def get_sentence_combined_embeddings_with_mask(
        self, io_parser_output: list[dict], task_type: str
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Given the io parser output it convert every io parser item into
        it's combined embedding and cross attention mask and list of encoder hidden state

        :param io_parser_output: input string -> io parser
        [
             {
                "token":126,
                "category":{
                   "type":"integer",
                   "subType":"default",
                   "subSubType":"none"
                },
                "position":0
             },
             {
                "token":"plus",
                "category":{
                   "type":"word",
                   "subType":"default",
                   "subSubType":"none"
                },
                "position":1
             },
         ]
        :param task_type: Type of task. ie: func_to_nl_translation.
        :return: Return the combined embeddings of sentence. Shape [ len(io_parser_output), 768]
        and cross attention mask. Shape[len(io_parser_output)]
        and list of encoder hidden state.
        """
        item_tensors = torch.empty((0, 768), dtype=torch.float32, device=self.device)
        mask_tensors = torch.empty(0, dtype=torch.bool, device=self.device)
        list_of_encoder_hidden_states = []

        for io_parser_output_item in io_parser_output:
            token: any = io_parser_output_item[Constants.TOKEN]
            category_map: dict = io_parser_output_item[Constants.CATEGORY]
            category_type: str = category_map[Constants.CATEGORY_TYPE]
            category_sub_subtype: str = category_map[Constants.CATEGORY_SUB_SUB_TYPE]

            token_embedding = self.get_token_embedding(
                token,
                category_type,
                io_parser_output_item
            ).to(self.device)

            category_and_task_embedding = self.get_category_and_task_embedding(
                category_map,
                task_type,
            ).to(self.device)
            combined_embedding = self.get_combined_embedding(
                token_embedding,
                category_and_task_embedding,
            )
            # Creating the combined embedding tensor
            item_tensors = torch.cat(
                (item_tensors, combined_embedding.unsqueeze(0)), dim=0
            )
            # Creating cross attention mask
            is_function_param = torch.tensor(
                [EmbeddingsManager.is_function_param_token(category_sub_subtype)],
                dtype=torch.bool,
                device=self.device
            )
            mask_tensors = torch.cat((mask_tensors, is_function_param), dim=0)
            if EmbeddingsManager.is_function_token(category_type):
                list_of_encoder_hidden_states.append(self.get_function_token_embedding(token, category_type).to(self.device))

        return item_tensors, mask_tensors, list_of_encoder_hidden_states

    def get_batch_embeddings_maps(
            self, batch_io_parser_output: list[list[dict]], task_type: list[str]
    ) -> list[list[dict]]:
        batch_embedding_maps = []
        for index, io_parser_output in enumerate(batch_io_parser_output):
            batch_embedding_maps.append(
                self.get_sentence_embeddings_maps(io_parser_output, task_type[index])
            )
        return batch_embedding_maps

    def get_sentence_embeddings_maps(
            self, io_parser_output: list[dict], task_type: str
    ) -> list[dict]:
        # TODO instead of creating a list can't we modify the given one ?
        embedding_maps = []
        for io_parser_output_item in io_parser_output:
            current_embeddings_map = self.get_embeddings_map(
                io_parser_output_item[Constants.TOKEN],
                io_parser_output_item[Constants.CATEGORY],
                io_parser_output_item[Constants.POSITION],
                task_type,
                io_parser_output_item,
            )
            embedding_maps.append(current_embeddings_map)
        return embedding_maps

    def get_embeddings_map(
            self,
            token: any,
            category_map: dict,
            position: int,
            task_type: str,
            io_parser_output_item: dict
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
        token_embedding = self.get_token_embedding(token, category_type, io_parser_output_item)
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

    def get_token_embedding(self,
                            token: any,
                            category_type: str,
                            io_parser_output_item: dict,
                            show_progress_bar: bool = False) -> Tensor:
        if category_type == CategoryType.FUNCTION.value:
            token = FunctionManager.get_doc_string_of_function(token)

        if self.use_our_tokenizer and category_type != CategoryType.FUNCTION.value:
            # print(f'Current io_parser_output_item is: {io_parser_output_item}')
            current_token_index = Tokenizer.get_index_of_token(
                    output_vocab_builder=self.output_vocab_builder,
                    io_parser_output_item=io_parser_output_item,
                )
            # print(f'Current token index: {current_token_index}')
            current_tensor = torch.tensor(current_token_index, dtype=torch.long).to(self.device)

            # print(f'Current embedded tensor is: {current_tensor}')
            return self.embedding_layer.forward(
                current_tensor
            ).to(self.device)
        else:
            return self.initial_word_encoder.get_sentence_embedding(
                str(token),
                True,
                show_progress_bar=show_progress_bar
            )

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
        if EmbeddingsManager.is_function_token(category_type):
            try:
                if not isinstance(token, types.FunctionType):
                    print(f"EmbeddingsManager the converted type of token is {type(token)} with value {token}")
                    token = FunctionManager().get_name_to_reference().get("nOtMyToKeN")
                return self.initial_function_encoder.get_perfect_function_signature_token_embedding(
                    token
                )
            except Exception as e:
                print(f"An unexpected error occurred in embedding manager module get_function_token_embedding function: {e}")
                print(f"Wrong token is: {token}")
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

    @staticmethod
    def is_function_param_token(category_sub_subtype: str) -> bool:
        """
        Check if the category map category sub subtype is a param of a function or not.

        :param category_sub_subtype:
        :return: Ture if function param otherwise false
        """
        if (
                category_sub_subtype == CategorySubSubType.PARAM_ONE.value
                or category_sub_subtype == CategorySubSubType.PARAM_TWO.value
                or category_sub_subtype == CategorySubSubType.PARAM_THREE.value
                or category_sub_subtype == CategorySubSubType.PARAM_FOUR.value
                or category_sub_subtype == CategorySubSubType.PARAM_FIVE.value
                or category_sub_subtype == CategorySubSubType.PARAM_LAST.value
        ):
            return True
        return False

    @staticmethod
    def is_function_token(category_type: str) -> bool:
        return True if category_type == CategoryType.FUNCTION.value else False


if __name__ == "__main__":
    sentences = [
        "The quick brown fox jumps over the lazy dog in the meadow",
        "Adding 3 plus 2 equals ##addition(3,2)",
    ]
    max_sequence_length = 16
    hidden_dim = 768
    corpus_io_parser_output = BatchBuilder.get_batch_io_parser_output(sentences, True, max_sequence_length)
    # Initialize category vocabulary builder instance
    category_vocab_builder = CategoryVocabBuilder(corpus_io_parser_output)
    category_vocab_size = len(category_vocab_builder.category_vocab_item_to_index.keys())

    print(f"Output token classification head count:"
          f" {len(category_vocab_builder.index_to_output_token_classification_head_vocab_item.keys())}\n"
          f"Output token classification head category type:"
          f" {category_vocab_builder.index_to_output_token_classification_head_vocab_item}")

    # Initialize output vocabulary builder instance
    output_vocab_builder = OutputVocabBuilder(
        corpus_of_io_parser_output=corpus_io_parser_output,
        index_to_output_token_classification_head_vocab_item=
        category_vocab_builder.index_to_output_token_classification_head_vocab_item
    )

    index_to_output_vocabularies = output_vocab_builder.index_to_output_vocabularies
    token_vocab_size = Tokenizer.get_token_vocab_size(index_to_output_vocabularies)
    embedding_layer = nn.Embedding(token_vocab_size, hidden_dim)
    embeddings_manager = EmbeddingsManager(
        batch_size=2,
        n_heads=8,
        max_sequence_length=max_sequence_length,
        with_mask=True,
        embedding_layer=embedding_layer,
        output_vocab_builder= output_vocab_builder,
        use_our_tokenizer= True,
    )

    item_tensors, mask_tensors, _ = embeddings_manager.get_sentence_combined_embeddings_with_mask(
        corpus_io_parser_output[0],
        "func_to_nl_translation",
    )
    print(f"items tensors shape: {item_tensors.shape}")

    batch_item_tensors, batch_mask_tensors, _ = embeddings_manager.get_batch_combined_embeddings_with_mask(
        corpus_io_parser_output,
        ["func_to_nl_translation", "func_to_nl_translation"]
    )
    print(f"batch item tensors shape: {batch_item_tensors.shape}")
    print(f"batch mask tensors shape: {batch_mask_tensors.shape}")
    print(f"mask tensors: {batch_mask_tensors}")
