# create multiple vocabulary for token based on unique category type and subtype.
from dataclasses import dataclass


@dataclass
class OutputTokenVocabItem:
    token: any

    def __hash__(self):
        """
        Calculate a hash value based on the fields that determine equality

        :return: The calculated hash
        """
        return hash(
            (
                self.token,
            )
        )

    def __str__(self):
        return f"OutputTokenVocabItem( token={self.token} )"


class OutputTokenVocabBuilder:
    """
    integer token -> Category vocab item -> Category map
    """
    def __init__(self):
        pass

