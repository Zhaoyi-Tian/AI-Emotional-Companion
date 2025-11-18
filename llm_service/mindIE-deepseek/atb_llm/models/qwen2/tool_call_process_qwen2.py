# Copyright Huawei Technologies Co., Ltd. 2023-2024.

import string
import random
import json
import re


FN_NAME = '✿FUNCTION✿'
FN_ARGS = '✿ARGS✿'
FN_RESULT = '✿RESULT✿'
FN_EXIT = '✿RETURN✿'
NAME = "name"
ARGUMENTS = "arguments"


class ToolsCallProcessorQwen2:

    def __init__(self, is_qwen1_5_or_2):
        self.is_qwen1_5_or_2 = is_qwen1_5_or_2

    @staticmethod
    def decode_qwen2_5(content):
        '''
        <tool_call>
        {"name": "get_rectangle_property", "arguments": {"perimeter": 14, "area": 15, "property": "length"}}
        </tool_call>
        '''
        lines = content.strip()
        pattern = re.compile(r'<tool_call>\s*({.*?})\s*</tool_call>', re.DOTALL)
        matches = pattern.findall(lines)
        is_tool_call = True
        if matches:
            try:
                tool_calls = [json.loads(match) for match in matches]
                for item in tool_calls:
                    _ = item[NAME]
                    _ = item[ARGUMENTS]
            except json.JSONDecodeError:
                is_tool_call = False

            if is_tool_call:
                call_res = []
                for item in tool_calls:
                    tool_call = {
                        NAME: item[NAME],
                        ARGUMENTS: json.dumps(item[ARGUMENTS], ensure_ascii=False)
                    }
                    characters = string.ascii_letters + string.digits
                    call_id = "call_" + ''.join(random.choice(characters) for _ in range(8))
                    res = {
                        "type": "function",
                        "id": call_id,
                        "function": tool_call
                    }
                    call_res.append(res)
                return {"tool_calls": call_res}
        return content.strip()

    @staticmethod
    def decode_qwen1_5_or_2(content):
        lines = content.strip()
        arguments_json = None
        is_tool_call = False
        if FN_NAME in lines and FN_ARGS in lines:
            arguments = lines.split(FN_ARGS)[1].split('✿')[0].strip(':').strip('\n').strip()
            function_name = lines.split(FN_NAME)[1].split('✿')[0].strip(':').strip('\n').strip()

            if function_name:
                is_tool_call = True
                try:
                    arguments_json = json.loads(arguments)
                except json.JSONDecodeError:
                    is_tool_call = False

            if is_tool_call:
                content = {
                    NAME: function_name,
                    ARGUMENTS: json.dumps(arguments_json if isinstance(arguments_json, dict) else arguments,
                                            ensure_ascii=False)
                }
                characters = string.ascii_letters + string.digits
                call_id = "call_" + ''.join(random.choice(characters) for _ in range(8))
                call_res = {
                    "type": "function",
                    "id": call_id,
                    "function": content
                }
                return {
                    "tool_calls": [call_res]
                }
        return content.strip()

    def decode(self, content):
        if self.is_qwen1_5_or_2:
            return self.decode_qwen1_5_or_2(content)
        else:
            return self.decode_qwen2_5(content)
