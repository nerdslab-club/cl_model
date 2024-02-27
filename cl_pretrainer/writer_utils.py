from vocabulary_builder.category_vocabulary_builder import OutputTokenClassificationHeadVocabItem


class WriterUtils:
    AVG_LOSS_TAG = 'Loss: Average'
    AVG_ACCURACY_TAG = 'Accuracy: Average'
    CATEGORY_MAP_LOSS_TAG = 'Loss: Category Map'
    CATEGORY_MAP_ACCURACY_TAG = 'Accuracy: Category Map'
    LEARNING_RATE = 'Learning Rate'

    @staticmethod
    def get_output_head_loss_tag(head_no: int, info: OutputTokenClassificationHeadVocabItem) -> str:
        return f'Loss: Output Head {head_no} {info.category_type} {info.category_subtype}'

    @staticmethod
    def get_output_head_accuracy_tag(head_no: int, info: OutputTokenClassificationHeadVocabItem) -> str:
        return f'Accuracy: Output Head {head_no} {info.category_type} {info.category_subtype}'
