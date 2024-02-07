import types
import unittest
from typing import Tuple, List

from cl_data.function_representation.src.functions_manager import FunctionManager
from cl_data.function_representation.src.math_functions import MathFunctions
from cl_data.io_parser.category_parser_utility import create_category_map
from cl_data.src.constants import Constants, CategoryType, CategorySubType, CategorySubSubType


class ResponseParser:

    @staticmethod
    def parse_sentence_io_parser_output(sentence_io_parser_output: list[dict]) -> str:
        sentence_response_parser_output = []
        i = 0
        while i < len(sentence_io_parser_output):
            io_parser_item = sentence_io_parser_output[i]

            token: any = io_parser_item[Constants.TOKEN]
            category_map: dict = io_parser_item[Constants.CATEGORY]
            category_type: str = category_map[Constants.CATEGORY_TYPE]
            category_sub_type: str = category_map[Constants.CATEGORY_SUB_TYPE]

            if category_type == CategoryType.FUNCTION.value:
                f_io_parser_output, last_index = ResponseParser.find_valid_function(
                    sentence_io_parser_output, i
                )
                result_token = ResponseParser.get_function_token(f_io_parser_output)
                sentence_response_parser_output.append(str(result_token))
                i = last_index
            elif category_type == CategoryType.SPECIAL.value:
                i += 1
                continue
            elif category_type == CategoryType.WORD.value:
                if ResponseParser.is_valid_data_token(category_sub_type):
                    sentence_response_parser_output.append(str(token))
            elif category_type == CategoryType.INTEGER.value:
                if ResponseParser.is_valid_data_token(category_sub_type):
                    sentence_response_parser_output.append(str(token))
            elif category_type == CategoryType.FLOAT.value:
                if ResponseParser.is_valid_data_token(category_sub_type):
                    sentence_response_parser_output.append(str(token))
            elif category_type == CategoryType.BOOL.value:
                if ResponseParser.is_valid_data_token(category_sub_type):
                    sentence_response_parser_output.append(str(token))
            elif category_type == CategoryType.LIST.value:
                if ResponseParser.is_valid_data_token(category_sub_type):
                    sentence_response_parser_output.append(str(token))
            i += 1

        return ' '.join(sentence_response_parser_output)

    @staticmethod
    def parse_corpus_io_parser_output(corpus_io_parser_output: list[list[dict]]) -> list[str]:
        corpus_response_parser_output = []
        for sentence_io_parser_output in corpus_io_parser_output:
            corpus_response_parser_output.append(
                ResponseParser.parse_sentence_io_parser_output(sentence_io_parser_output)
            )
        return corpus_response_parser_output

    @staticmethod
    def is_valid_data_token(category_sub_type: str) -> bool:
        return (category_sub_type == CategorySubType.DEFAULT.value or
                category_sub_type == CategorySubType.RETURN_VALUE.value or
                category_sub_type == CategorySubType.PLACEHOLDER.value)

    @staticmethod
    def find_valid_function(lst: list[dict], start_from=0) -> Tuple[List[dict], int]:
        stack = []

        for i in range(start_from, len(lst)):
            io_parser_item = lst[i]
            category_map: dict = io_parser_item[Constants.CATEGORY]
            category_type: str = category_map[Constants.CATEGORY_TYPE]
            category_sub_sub_type: str = category_map[Constants.CATEGORY_SUB_SUB_TYPE]

            if category_type == CategoryType.FUNCTION.value:
                stack.append("one function found")
            if category_type == CategoryType.FUNCTION.value and len(
                    stack) > 0 and category_sub_sub_type == CategorySubSubType.PARAM_LAST.value:
                stack.pop()
            elif len(stack) > 0 and category_sub_sub_type == CategorySubSubType.PARAM_LAST.value:
                stack.pop()
                if len(stack) == 0:
                    valid_sublist = lst[start_from:i + 1]
                    return valid_sublist, i

        # If no valid sublist is found
        return [], -1

    @staticmethod
    def get_function_token(f_io_parser_output: list[dict]) -> str:
        stack = []
        copied_list = []
        function_action = CategorySubSubType.EXECUTE

        for i in range(0, len(f_io_parser_output)):
            io_parser_item = f_io_parser_output[i]
            category_map: dict = io_parser_item[Constants.CATEGORY]
            category_type: str = category_map[Constants.CATEGORY_TYPE]
            category_sub_sub_type: str = category_map[Constants.CATEGORY_SUB_SUB_TYPE]
            if category_type == CategoryType.FUNCTION.value:
                if len(stack) == 0:
                    # This is the parent function
                    function_action = category_sub_sub_type
                stack.append((i, io_parser_item))

            elif len(stack) > 0 and category_sub_sub_type == CategorySubSubType.PARAM_LAST.value:
                index, current_io_parser_item = stack.pop()
                current_function_to_execute: any = current_io_parser_item[Constants.TOKEN]
                current_category_map: dict = current_io_parser_item[Constants.CATEGORY]
                current_function_category_sub_sub_type: str = current_category_map[Constants.CATEGORY_SUB_SUB_TYPE]

                # Extract the relevant portion of the list using slicing
                params_io_parser_output = f_io_parser_output[index + 1:i + 1]
                params_to_pass = [item[Constants.TOKEN] for item in params_io_parser_output]
                params_to_pass = ResponseParser.process_params(params_to_pass)

                if function_action == CategorySubSubType.EXECUTE.value:
                    if not isinstance(current_function_to_execute, types.FunctionType):
                        print(f"ResponseParser the converted type of token is {type(current_function_to_execute)} with value {current_function_to_execute}")
                        current_function_to_execute = FunctionManager().get_name_to_reference().get("nOtMyToKeN")
                        result = current_function_to_execute()
                    else:
                        result = current_function_to_execute(*params_to_pass)
                # function_action == CategorySubSubType.REPRESENT.value
                # function_action == CategorySubSubType.PLACEHOLDER.value
                else:
                    if not isinstance(current_function_to_execute, types.FunctionType):
                        print(f"ResponseParser the converted type of token is {type(current_function_to_execute)} with value {current_function_to_execute}")
                        current_function_to_execute = FunctionManager().get_name_to_reference().get("nOtMyToKeN")
                    name_of_function = FunctionManager.get_name_of_function(current_function_to_execute)
                    comma_separated_params = ', '.join(str(item) for item in params_to_pass)
                    result = f"{name_of_function}({comma_separated_params})"

                if len(stack) == 0:
                    return result
                else:
                    # Replace from index to i with new value
                    copied_list.append({
                        Constants.TOKEN: result,
                        Constants.CATEGORY: create_category_map(
                            ResponseParser.get_runtime_type(result),
                            CategorySubType.RETURN_VALUE,
                            ResponseParser.get_enum_item(current_function_category_sub_sub_type, CategorySubSubType),
                        ),
                        Constants.POSITION: index,
                    })

        if len(stack) > 0:
            for j in range(0, len(stack)):
                current_item_index, current_item = stack.pop()
                copied_list.insert(0, current_item)

            return ResponseParser.get_function_token(copied_list)

    @staticmethod
    def process_params(params_to_pass):
        # Iterating over the values in params_to_pass
        for i in range(len(params_to_pass)):
            value = params_to_pass[i]

            # Checking if the value is a string starting with "[" and ending with "]"
            if isinstance(value, str) and value.startswith("[") and value.endswith("]"):
                # Casting the string to a list and replacing the value in the original list
                new_list = eval(value)  # Using eval to safely convert the string to a list
                params_to_pass[i] = new_list

        return params_to_pass

    @staticmethod
    def get_enum_item(str_value, my_enum):
        for item in my_enum:
            if item.value == str_value:
                return item
        return None

    @staticmethod
    def get_runtime_type(variable) -> CategoryType:
        if isinstance(variable, str):
            return CategoryType.WORD
        elif isinstance(variable, int):
            return CategoryType.INTEGER
        elif isinstance(variable, float):
            return CategoryType.FLOAT
        elif isinstance(variable, list):
            return CategoryType.LIST
        else:
            return CategoryType.BOOL


class ResponseParserTest(unittest.TestCase):
    response_parser = ResponseParser()
    batch_io_parser_output = [
        [
            {
                "token": '<BOS>',
                "category": {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
                "position": 0
            },
            {
                "token": 'The',
                "category": {'type': 'word', 'subType': 'default', 'subSubType': 'none'},
                "position": 1
            },
            {
                "token": 'average',
                "category": {'type': 'word', 'subType': 'default', 'subSubType': 'none'},
                "position": 2
            },
            {
                "token": 'of',
                "category": {'type': 'word', 'subType': 'default', 'subSubType': 'none'},
                "position": 3
            },
            {
                "token": 2,
                "category": {'type': 'integer', 'subType': 'default', 'subSubType': 'none'},
                "position": 4
            },
            {
                "token": ',',
                "category": {'type': 'word', 'subType': 'default', 'subSubType': 'none'},
                "position": 5
            },
            {
                "token": 3,
                "category": {'type': 'integer', 'subType': 'default', 'subSubType': 'none'},
                "position": 6
            },
            {
                "token": ',',
                "category": {'type': 'word', 'subType': 'default', 'subSubType': 'none'},
                "position": 7
            },
            {
                "token": 4,
                "category": {'type': 'integer', 'subType': 'default', 'subSubType': 'none'},
                "position": 8
            },
            {
                "token": 'is',
                "category": {'type': 'word', 'subType': 'default', 'subSubType': 'none'},
                "position": 9
            },
            {
                "token": '=',
                "category": {'type': 'word', 'subType': 'default', 'subSubType': 'none'},
                "position": 10
            },
            {
                "token": MathFunctions.average,
                "category": {'type': 'function', 'subType': 'float', 'subSubType': 'execute'},
                "position": 11
            },
            {
                "token": [2, 3, 4],
                "category": {'type': 'list', 'subType': 'default', 'subSubType': 'param_last'},
                "position": 12
            },
            {
                "token": '<EOS>',
                "category": {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
                "position": 13
            }
        ],
        [
            {
                "token": '<BOS>',
                "category": {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
                "position": 0
            },
            {
                "token": MathFunctions.average,
                "category": {'type': 'function', 'subType': 'float', 'subSubType': 'represent'},
                "position": 1
            },
            {
                "token": [1, 2, 3],
                "category": {'type': 'list', 'subType': 'default', 'subSubType': 'param_last'},
                "position": 2
            },
            {
                "token": '<EOS>',
                "category": {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
                "position": 3
            }
        ],
        [
            {
                "token": '<BOS>',
                "category": {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
                "position": 0
            },
            {
                "token": MathFunctions.average,
                "category": {'type': 'function', 'subType': 'float', 'subSubType': 'placeholder'},
                "position": 1
            },
            {
                "token": '@list',
                "category": {'type': 'word', 'subType': 'placeholder', 'subSubType': 'param_last'},
                "position": 2
            },
            {
                "token": '<EOS>',
                "category": {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
                "position": 3
            }
        ],
        [
            {
                'token': '<BOS>',
                'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
                'position': 0,
            },
            {
                'token': MathFunctions.division,
                'category': {'type': 'function', 'subType': 'float', 'subSubType': 'execute'},
                'position': 1,
            },
            {
                'token': MathFunctions.sum,
                'category': {'type': 'function', 'subType': 'float', 'subSubType': 'param_one'},
                'position': 2,
            },
            {
                'token': [1, 2, 3],
                'category': {'type': 'list', 'subType': 'default', 'subSubType': 'param_last'},
                'position': 3,
            },
            # {
            #     'token': 6,
            #     'category': {'type': "integer", 'subType': 'return_type', 'subSubType': 'param_one'},
            #     'position': 2,
            # },
            {
                'token': MathFunctions.length,
                'category': {'type': 'function', 'subType': 'integer', 'subSubType': 'param_last'},
                'position': 4,
            },
            {
                'token': [1, 2, 3],
                'category': {'type': 'list', 'subType': 'default', 'subSubType': 'param_last'},
                'position': 5,
            },
            # {
            #     'token': 3,
            #     'category': {'type': "integer", 'subType': 'return_type', 'subSubType': 'param_last'},
            #     'position': 2,
            # },
            {
                'token': '<EOS>',
                'category': {'type': 'special', 'subType': 'word', 'subSubType': 'none'},
                'position': 6,
            },
        ]
    ]

    batch_response_parser_output = [
        'The average of 2 , 3 , 4 is = 3.0',
        'average([1, 2, 3])',
        'average(@list)',
        '2.0',
    ]

    def test_get_enum_item(self):
        value = ResponseParser.get_enum_item("param_last", CategorySubSubType)
        print(f'The enum item is {value}')
        self.assertEqual(value, CategorySubSubType.PARAM_LAST)

    def test_find_functions(self):
        f_io_parser_output, last_index = ResponseParser.find_valid_function(
            ResponseParserTest.batch_io_parser_output[0], 11
        )
        self.assertEqual(last_index, 12)
        self.assertEqual(len(f_io_parser_output), 2)
        print(f"Only Function token list: {f_io_parser_output}")
        print(f"Last index: {last_index}")

        result_token = ResponseParser.get_function_token(f_io_parser_output)
        print(f"The result of the function run is: {result_token}")
        self.assertEqual(result_token, 3.0)

    def test_parse_sentence_io_parser_output(self):
        result = ResponseParser.parse_sentence_io_parser_output(ResponseParserTest.batch_io_parser_output[0])
        print(f"Sentence response parser output: \n{result}")
        self.assertEqual(result, 'The average of 2 , 3 , 4 is = 3.0')

    def test_parse_corpus_io_parser_output(self):
        result = ResponseParser.parse_corpus_io_parser_output(ResponseParserTest.batch_io_parser_output)
        print(result)
        self.assertEqual(result, ResponseParserTest.batch_response_parser_output)


if __name__ == "__main__":
    unittest.main()
