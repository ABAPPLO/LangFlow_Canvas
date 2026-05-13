"""JSON to Data component - parse JSON string into LangFlow Data object."""

from __future__ import annotations

import json
import re

from lfx.custom import Component
from lfx.inputs import MessageTextInput
from lfx.io import Output
from lfx.schema.data import Data
from lfx.schema.message import Message


class JsonToDataComponent(Component):
    display_name = "JSON 转 Data"
    description = "将 JSON 字符串解析为 Data 对象。"
    icon = "braces"
    name = "JsonToData"

    inputs = [
        MessageTextInput(
            name="json_string",
            display_name="JSON 字符串",
            info="要解析的 JSON 字符串，支持带或不带 ```json 代码块包裹。",
            required=True,
        ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="parse_json", types=["Data"]),
    ]

    def _extract_json(self, text: str) -> dict | list | None:
        """Extract JSON from text, trying multiple strategies."""
        text = text.strip()

        # 1. Direct parse
        result = self._try_parse(text)
        if result is not None:
            return result

        # 2. JSON in code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            result = self._try_parse(match.group(1))
            if result is not None:
                return result

        # 3. tool_calls format: {"name": "...", "arguments": "{...}"}
        match = re.search(r'"arguments"\s*:\s*"(.*?)(?:"\s*[,}])', text, re.DOTALL)
        if match:
            # Unescape nested JSON string
            args_text = match.group(1).replace('\\"', '"').replace("\\n", "\n")
            result = self._try_parse(args_text)
            if result is not None:
                return result

        # 4. Brace/bracket matching for embedded JSON
        result = self._extract_balanced(text)
        if result is not None:
            return result

        return None

    @staticmethod
    def _try_parse(text: str) -> dict | list | None:
        try:
            result = json.loads(text.strip())
            if isinstance(result, dict | list):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
        return None

    @staticmethod
    def _extract_balanced(text: str) -> dict | list | None:
        """Find the first valid JSON object or array using brace/bracket counting."""
        for open_ch, close_ch in [("{", "}"), ("[", "]")]:
            start = text.find(open_ch)
            while start != -1:
                depth = 0
                in_str = False
                escape = False
                for i in range(start, len(text)):
                    ch = text[i]
                    if escape:
                        escape = False
                        continue
                    if ch == "\\":
                        escape = True
                        continue
                    if ch == '"':
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == open_ch:
                        depth += 1
                    elif ch == close_ch:
                        depth -= 1
                        if depth == 0:
                            try:
                                result = json.loads(text[start : i + 1])
                                if isinstance(result, dict | list):
                                    return result
                            except (json.JSONDecodeError, TypeError):
                                pass
                            break
                start = text.find(open_ch, start + 1)
        return None

    def parse_json(self) -> Data:
        raw = self.json_string
        if isinstance(raw, Message):
            raw = raw.get_text()
        text = str(raw).strip()

        if not text:
            self.status = "输入为空"
            return Data(data={})

        result = self._extract_json(text)
        if result is None:
            self.status = "JSON 解析失败"
            return Data(data={"error": "无法解析 JSON", "raw": text[:200]})

        # Wrap list result so downstream Data components can consume it
        if isinstance(result, list):
            self.status = f"解析成功，JSON 数组包含 {len(result)} 个元素"
            return Data(data={"items": result})

        self.status = f"解析成功，共 {len(result)} 个字段"
        return Data(data=result)