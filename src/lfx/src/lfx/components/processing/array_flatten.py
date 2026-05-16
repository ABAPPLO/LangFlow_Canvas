"""Array Flatten component - flatten JSON array objects into key-value strings."""

from __future__ import annotations

import json

from lfx.custom import Component
from lfx.inputs import MessageTextInput
from lfx.io import Output
from lfx.schema.message import Message


class ArrayFlattenComponent(Component):
    display_name = "JSON 键值拼接"
    description = (
        "将 JSON 数组中每个元素展开为 key：value。key：value。 格式的字符串，以 JSON 字符串数组输出。"
        '\n\n示例：输入 [{"name":"张三","age":25},{"name":"李四","age":30}]'
        '\n输出 ["name：张三。age：25","name：李四。age：30"]'
    )
    icon = "list"
    name = "ArrayFlatten"

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info="JSON 数组格式的字符串。",
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Flattened", name="flattened", method="flatten_array", type_=Message),
    ]

    def _flatten_value(self, value) -> str:
        """Convert values to strings; recursively flatten dicts and join lists with semicolons."""
        if isinstance(value, dict):
            result = self._format_element(value)
        elif isinstance(value, list):
            items = [self._flatten_value(item) for item in value]
            result = "；".join(items)
        else:
            result = str(value)
        return result.rstrip("。")

    def _format_element(self, element) -> str:
        """Flatten one dict into key-value text."""
        if isinstance(element, dict):
            parts = []
            for key, value in element.items():
                parts.append(f"{key}：{self._flatten_value(value)}")
            return "。".join(parts)
        return str(element)

    def flatten_array(self) -> Message:
        raw = self.input_text
        if isinstance(raw, Message):
            raw = raw.get_text()
        text = str(raw).strip() if raw else ""

        if not text:
            self.status = "输入为空"
            return Message(text="[]")

        try:
            array = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            self.status = "JSON 解析失败"
            return Message(text="[]")

        if not isinstance(array, list):
            array = [array]

        result = [self._format_element(element) for element in array]
        self.status = f"已展开 {len(result)} 个元素"
        return Message(text=json.dumps(result, ensure_ascii=False))
