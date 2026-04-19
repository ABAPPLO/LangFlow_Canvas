"""Smart Extract component - use LLM to precisely extract specified fields from text."""

from __future__ import annotations

import json
import re
from typing import Any

from lfx.custom import Component
from lfx.inputs.inputs import DropdownInput, MessageTextInput, ModelInput, StrInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.schema.message import Message
from lfx.template.field.base import Output

MODE_TABLE = "Table"
MODE_DIRECT = "Direct"

EXTRACTION_PROMPT = """你是一个精确的文本提取器。请从输入文本中提取以下字段的值。

规则：
1. 尽量使用原文中的确切文字，不要改写、总结或转述
2. 如果某个字段在文本中找不到，返回空字符串
3. 只返回JSON，不要返回其他内容

需要提取的字段：
{fields_text}

{instructions_section}

输入文本：
{input_text}

请严格按以下JSON格式返回结果：
{json_template}"""


class SmartExtractComponent(Component):
    display_name = "Smart Extract"
    description = "用 LLM 从文本中精确提取指定字段，支持表格和直接输出两种模式。"
    icon = "scan-text"
    name = "SmartExtract"

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info="要提取字段的文本。",
        ),
        ModelInput(
            name="language_model",
            display_name="Language Model",
        ),
        DropdownInput(
            name="mode",
            display_name="Output Mode",
            options=[MODE_TABLE, MODE_DIRECT],
            value=MODE_TABLE,
            info="Table：输出一行表格；Direct：每个字段一个独立输出端口。",
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="fields",
            display_name="Fields",
            info="要提取的字段名，点击 Add Field 逐个添加。",
            is_list=True,
            list_add_label="Add Field",
            placeholder="Enter field name...",
            input_types=[],
            real_time_refresh=True,
        ),
        StrInput(
            name="instructions",
            display_name="Instructions",
            info="额外的提取指令，如字段的含义或格式要求。",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Table", name="table", method="extract_table", types=["DataFrame"]),
    ]

    # ------------------------------------------------------------------
    # Dynamic outputs
    # ------------------------------------------------------------------

    def _get_fields_list(self, raw) -> list[str]:
        if not raw:
            return []
        if isinstance(raw, list):
            return [str(f).strip() for f in raw if str(f).strip()]
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        return []

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        template = frontend_node.get("template", {})

        # Resolve current mode
        if field_name == "mode":
            mode = field_value
        else:
            mode = template.get("mode", {}).get("value", MODE_TABLE)

        # Resolve current fields
        if field_name == "fields":
            raw_fields = field_value
        else:
            raw_fields = template.get("fields", {}).get("value", [])
        fields = self._get_fields_list(raw_fields)

        frontend_node["outputs"] = []

        if mode == MODE_TABLE:
            frontend_node["outputs"].append(
                Output(display_name="Table", name="table", method="extract_table", types=["DataFrame"]),
            )
        else:
            for i, field in enumerate(fields):
                frontend_node["outputs"].append(
                    Output(
                        display_name=field,
                        name=f"field_{i + 1}",
                        method="extract_field",
                        types=["Message"],
                        group_outputs=True,
                    ),
                )
            frontend_node["outputs"].append(
                Output(
                    display_name="All Fields",
                    name="all_fields",
                    method="extract_all",
                    types=["Data"],
                    group_outputs=True,
                ),
            )

        return frontend_node

    # ------------------------------------------------------------------
    # LLM extraction (cached per run)
    # ------------------------------------------------------------------

    def _build_prompt(self, fields: list[str]) -> str:
        fields_text = "\n".join(f"- {f}" for f in fields)
        json_template = json.dumps({f: "..." for f in fields}, ensure_ascii=False, indent=2)
        instructions_section = f"额外指令：\n{self.instructions}" if self.instructions else ""
        input_text = self.input_text
        if isinstance(input_text, Message):
            input_text = input_text.get_text()
        return EXTRACTION_PROMPT.format(
            fields_text=fields_text,
            json_template=json_template,
            instructions_section=instructions_section,
            input_text=str(input_text),
        )

    def _parse_json(self, text: str) -> dict:
        # Direct parse
        try:
            return json.loads(text)
        except (json.JSONDecodeError, TypeError):
            pass
        # JSON in code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except (json.JSONDecodeError, TypeError):
                pass
        # First JSON object in text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (json.JSONDecodeError, TypeError):
                pass
        return {}

    def _do_extract(self) -> dict:
        if hasattr(self, "_cached_smart_extract"):
            return self._cached_smart_extract

        fields = self._get_fields()
        if not fields:
            self._cached_smart_extract = {}
            return {}

        prompt = self._build_prompt(fields)
        llm = self.language_model
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        result = self._parse_json(response_text)

        # Ensure all fields exist in result
        for f in fields:
            if f not in result:
                result[f] = ""

        self.status = f"Extracted {len(fields)} fields"
        self._cached_smart_extract = result
        return result

    def _get_fields(self) -> list[str]:
        return self._get_fields_list(getattr(self, "fields", None))

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def extract_table(self) -> DataFrame:
        result = self._do_extract()
        return DataFrame([result])

    def extract_field(self) -> Message:
        result = self._do_extract()
        output_name = getattr(self, "_current_output", "")
        fields = self._get_fields()
        idx = 0
        if output_name.startswith("field_"):
            try:
                idx = int(output_name.split("_")[1]) - 1
            except (ValueError, IndexError):
                idx = 0

        if idx < len(fields):
            value = result.get(fields[idx], "")
        else:
            value = ""
        return Message(text=str(value))

    def extract_all(self) -> Data:
        result = self._do_extract()
        return Data(data=result)
