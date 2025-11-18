# Copyright (c) 2023-2024 DeepSeek.

import copy
import dataclasses
from enum import IntEnum, auto
from typing import Dict, List, Tuple


_CONTENT = 'content'
_ROLE = 'role'


class SeparatorStyle(IntEnum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    NO_COLON_TWO = auto()
    ADD_NEW_LINE_SINGLE = auto()
    LLAMA2 = auto()
    CHATGLM = auto()
    CHATML = auto()
    CHATINTERN = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    DeepSeek = auto()
    PLAIN = auto()
    ALIGNMENT = auto()


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str, str] = (("USER", "ASSISTANT"),)
    # All messages. Each item is (role, message).
    messages: List[List[str]] = None
    # The number of few shot examples
    offset: int = 0
    # The separator style and configurations
    sep_style: SeparatorStyle = SeparatorStyle.ADD_COLON_SINGLE
    sep: str = "\n"
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: List[str] = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None

    def generate_prompt_with_conversation(self):
        prompts = []
        messages = self.messages

        for i in range(0, len(messages), 2):
            prompt = {
                _ROLE: messages[i][0],
                _CONTENT: (
                    messages[i][1][0]
                    if isinstance(messages[i][1], tuple)
                    else messages[i][1]
                ),
                "images": [messages[i][1][1]] if isinstance(messages[i][1], tuple) else [],
            }
            response = {_ROLE: messages[i + 1][0], _CONTENT: messages[i + 1][1]}
            prompts.extend([prompt, response])

        return prompts

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        system_prompt = self.system_template.format(system_message=self.system_message)

        if self.sep_style == SeparatorStyle.DeepSeek:
            seps = [self.sep, self.sep2]
            if system_prompt == "" or system_prompt is None:
                ret = ""
            else:
                ret = system_prompt + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.LLAMA2:
            seps = [self.sep, self.sep2]
            if self.system_message:
                ret = system_prompt
            else:
                ret = "[INST] "
            for i, (role, message) in enumerate(self.messages):
                tag = self.roles[i % 2]
                if message:
                    if isinstance(message, tuple):  # multimodal message
                        message, _ = message
                    if i == 0:
                        ret += message + " "
                    else:
                        ret += tag + " " + message + seps[i % 2]
                else:
                    ret += tag
            return ret
        elif self.sep_style == SeparatorStyle.PLAIN:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    if i % 2 == 0:
                        ret += message + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        elif self.sep_style == SeparatorStyle.ALIGNMENT:
            seps = [self.sep, self.sep2]
            ret = ""
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if isinstance(message, tuple):
                        message, _, _ = message
                    if i % 2 == 0:
                        ret += "<image>\n" + seps[i % 2]
                    else:
                        ret += message + seps[i % 2]
                else:
                    ret += ""
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def get_prompt_for_current_round(self, content=None):
        """Get current round formatted question prompt during sft training"""
        if self.sep_style == SeparatorStyle.PLAIN:
            formatted_question = "<image>\n"
        elif self.sep_style == SeparatorStyle.DeepSeek:
            formatted_question = (
                f"{self.roles[0]}: " + content.strip() + self.sep + f"{self.roles[1]}:"
            )
        else:
            raise ValueError(f"Unsupported sep_style: {self.sep_style}")
        return formatted_question

    def set_system_message(self, system_message: str):
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def reset_message(self):
        """Reset a new message."""
        self.messages = []

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        system_prompt = self.system_template.format(system_message=self.system_message)
        ret = [{_ROLE: "system", _CONTENT: system_prompt}]

        for i, (_, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append({_ROLE: "user", _CONTENT: msg})
            else:
                if msg is not None:
                    ret.append({_ROLE: "assistant", _CONTENT: msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=copy.deepcopy(self.messages),
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override and template.name in conv_templates:
        raise ValueError(f"{template.name} has been registered.")

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    template = conv_templates.get(name)
    return template.copy() if template else None


# deepseek template
register_conv_template(
    Conversation(
        name="deepseek",
        system_template="{system_message}",
        system_message="",
        roles=("User", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.DeepSeek,
        sep="\n\n",
        sep2="<｜end▁of▁sentence｜>",
        stop_token_ids=[100001],
        stop_str=["User:", "<｜end▁of▁sentence｜>"],
    )
)
