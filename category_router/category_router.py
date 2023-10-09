# It will keep all instance of output token classification head
# and route each token
# encode decode for output vocab builder pass as format
from torch import nn

from vocabulary_builder.output_vocabulary_builder import OutputVocabBuilder


class CategoryRouter(nn.Module):

    def __init__(
            self,
            output_vocab_builder: OutputVocabBuilder,
    ):
        super().__init__()
        pass

