import json

from cl_data.src.constants import Constants


class SimpleResponseParser:
    @staticmethod
    def print_response_to_console(data: list[dict] | list[list[dict]]):
        """
        Pretty prints a list of dictionaries or a nested list of dictionaries.

        :param data: serializable data which is to be print.
        """
        json_str = json.dumps(SimpleResponseParser.serialize_function_tokens_for_printing(data), indent=4)
        print(json_str)

    @staticmethod
    def serialize_function_tokens_for_printing(data: list[dict] | list[list[dict]]) -> list[dict] | list[list[dict]]:
        """
        Cast all non-serializable callable funtion tokens to str

        :param data: batch io parser outputs or io parser outputs
        :return: serializable data after casting function token to string.
        """
        for i, item in enumerate(data):
            if isinstance(item, list):
                for j, second_level_item in enumerate(item):
                    if callable(data[i][j][Constants.TOKEN]):
                        data[i][j][Constants.TOKEN] = str(data[i][j][Constants.TOKEN])
            elif isinstance(item, dict):
                if callable(data[i][Constants.TOKEN]):
                    data[i][Constants.TOKEN] = str(data[i][Constants.TOKEN])
        return data


if __name__ == "__main__":
    item = [
        {
            "token": Constants.FUNCTION_DEFAULT_VALUE,
            "category": {
                "type": "integer",
                "subType": "default",
                "subSubType": "none"
            },
            "position": 0
        },
        {
            "token": "plus",
            "category": {
                "type": "word",
                "subType": "default",
                "subSubType": "none"
            },
            "position": 1
        },
        {
            "token": 840,
            "category": {
                "type": "integer",
                "subType": "default",
                "subSubType": "none"
            },
            "position": 2
        },
        {
            "token": "equals?",
            "category": {
                "type": "word",
                "subType": "default",
                "subSubType": "none"
            },
            "position": 3
        }
    ]
    SimpleResponseParser.print_response_to_console([item, item])

