"""Smart Extract component - use LLM to precisely extract specified fields from text."""

from __future__ import annotations

import json
import re
from typing import Any

from lfx.custom import Component
from lfx.inputs.inputs import DropdownInput, MessageTextInput, ModelInput, MultilineInput, StrInput
from lfx.schema.data import Data
from lfx.schema.message import Message
from lfx.template.field.base import Output

MODE_TEXT = "Text"
MODE_JSON = "JSON"
MODE_AUTO_SPLIT = "Auto Split"

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

EXTRACTION_PROMPT_WITH_EXAMPLE = """你是一个精确的文本提取器。请从输入文本中提取信息，严格按照给定的JSON格式输出。

规则：
1. 仔细阅读输入文本，从中提取与示例格式中各字段对应的信息
2. 严格按照下面的JSON示例格式返回结果，保持相同的结构和字段名
3. 如果示例是数组格式，对文本中每个独立的条目都生成一个数组元素
4. 尽量使用原文中的确切文字，不要改写、总结或转述
5. 如果某个字段在文本中找不到，返回空字符串
6. 示例中的具体值仅表示格式，你需要用从文本中提取的真实数据替换它们
7. 只返回JSON，不要返回其他内容

{instructions_section}

输入文本：
{input_text}

请严格按照以下JSON格式返回结果（保持示例中的结构和字段名，用从文本中提取的真实数据替换示例中的占位值）：
{json_example}"""

AUTO_SPLIT_PROMPT = """请将以下输入拆分为独立片段。

规则：
1. 分析输入结构，自动识别拆分边界：
   - JSON数组或JSON对象中的数组 → 每个元素为一个片段
   - 文本中有重复出现的标题/标记/序号/emoji → 按这些边界拆分
   - 有明确段落/章节/模块结构的长文本 → 按逻辑单元拆分
2. 每个片段保持原始内容的完整结构和格式，不要遗漏任何字段、表格、列表等细节
3. 返回JSON数组格式：["片段1", "片段2", ...]
4. 每个片段以JSON字符串形式返回（片段内含JSON时需正确转义）
5. 只返回JSON数组，不要其他内容

输入：
{input_text}"""


class SmartExtractComponent(Component):
    display_name = "Smart Extract"
    description = "用 LLM 从文本中精确提取指定字段，支持文本直接输出、JSON 格式输出和自动拆分三种模式。"
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
            options=[MODE_TEXT, MODE_JSON, MODE_AUTO_SPLIT],
            value=MODE_TEXT,
            info="Text：每个字段直接输出；JSON：按JSON格式输出（可提供格式示例）；Auto Split：自动识别模块并拆分输出。",
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
        MultilineInput(
            name="json_example",
            display_name="JSON Example",
            info="JSON 输出格式示例，LLM 会参考此格式输出。仅在 JSON 模式下生效。",
            value="",
            show=False,
            placeholder='例如：{"name": "张三", "age": 25, "skills": ["Python", "Go"]}',
            advanced=False,
        ),
        StrInput(
            name="instructions",
            display_name="Instructions",
            info="额外的提取指令，如字段的含义或格式要求。",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="extract_text"),
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

    def update_build_config(self, build_config: dict, field_value: Any, field_name: str | None = None) -> dict:
        if field_name == "mode":
            is_auto = (field_value == MODE_AUTO_SPLIT)
            is_json = (field_value == MODE_JSON)
            build_config["fields"]["hidden"] = is_auto
            build_config["fields"]["show"] = not is_auto
            build_config["fields"]["value"] = [] if is_auto else build_config["fields"].get("value", [])
            build_config["json_example"]["show"] = is_json
        return build_config

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        template = frontend_node.get("template", {})

        # Resolve current mode
        if field_name == "mode":
            mode = field_value
        else:
            mode = template.get("mode", {}).get("value", MODE_TEXT)

        # Resolve current fields
        if field_name == "fields":
            raw_fields = field_value
        else:
            raw_fields = template.get("fields", {}).get("value", [])
        fields = self._get_fields_list(raw_fields)

        frontend_node["outputs"] = []

        if mode == MODE_TEXT:
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
        elif mode == MODE_JSON:
            frontend_node["outputs"].append(
                Output(
                    display_name="JSON Result",
                    name="json_result",
                    method="extract_json",
                    types=["Message"],
                ),
            )
            frontend_node["outputs"].append(
                Output(
                    display_name="Data Result",
                    name="data_result",
                    method="extract_data",
                    types=["Data"],
                ),
            )
        elif mode == MODE_AUTO_SPLIT:
            frontend_node["outputs"].append(
                Output(
                    display_name="所有段落",
                    name="all_segments",
                    method="extract_all_segments",
                    types=["Message"],
                ),
            )

        return frontend_node

    # ------------------------------------------------------------------
    # LLM extraction (cached per run)
    # ------------------------------------------------------------------

    def _get_input_text(self) -> str:
        input_text = self.input_text
        if isinstance(input_text, Message):
            input_text = input_text.get_text()
        return str(input_text)

    def _build_prompt(self, fields: list[str]) -> str:
        instructions_section = f"额外指令：\n{self.instructions}" if self.instructions else ""

        json_example = getattr(self, "json_example", "").strip()
        if json_example:
            # Normalize fullwidth commas/colons to ASCII for valid JSON
            json_example = json_example.replace("，", ",").replace("：", ":")
            return EXTRACTION_PROMPT_WITH_EXAMPLE.format(
                instructions_section=instructions_section,
                input_text=self._get_input_text(),
                json_example=json_example,
            )

        fields_text = "\n".join(f"- {f}" for f in fields)
        json_template = json.dumps(dict.fromkeys(fields, "..."), ensure_ascii=False, indent=2)
        return EXTRACTION_PROMPT.format(
            fields_text=fields_text,
            json_template=json_template,
            instructions_section=instructions_section,
            input_text=self._get_input_text(),
        )

    def _build_split_prompt(self) -> str:
        return AUTO_SPLIT_PROMPT.format(input_text=self._get_input_text())

    def _parse_json(self, text: str) -> dict | list:
        # Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, (dict, list)):
                return result
        except (json.JSONDecodeError, TypeError):
            pass
        # JSON in code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, (dict, list)):
                    return result
            except (json.JSONDecodeError, TypeError):
                pass
        # First JSON array in text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return result
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

    def _parse_json_array(self, text: str) -> list[str]:
        # Direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(item) for item in result]
        except (json.JSONDecodeError, TypeError):
            pass
        # JSON array in code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(1))
                if isinstance(result, list):
                    return [str(item) for item in result]
            except (json.JSONDecodeError, TypeError):
                pass
        # First JSON array in text
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                result = json.loads(match.group(0))
                if isinstance(result, list):
                    return [str(item) for item in result]
            except (json.JSONDecodeError, TypeError):
                pass
        return []

    def _do_extract(self) -> dict | list:
        if hasattr(self, "_cached_smart_extract"):
            return self._cached_smart_extract

        fields = self._get_fields()
        json_example = getattr(self, "json_example", "").strip()

        print(f"[SmartExtract] fields={fields}, json_example={repr(json_example[:80]) if json_example else '(empty)'}", flush=True)

        if not fields and not json_example:
            self._cached_smart_extract = {}
            return {}

        prompt = self._build_prompt(fields)
        print(f"[SmartExtract] prompt:\n{prompt}", flush=True)

        llm = self.language_model
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        print(f"[SmartExtract] LLM response (first 300): {response_text[:300]}", flush=True)

        result = self._parse_json(response_text)
        print(f"[SmartExtract] parsed result type={type(result).__name__}, len={len(result)}", flush=True)

        # Ensure all fields exist in result (only when using fields-based prompt)
        if not json_example and isinstance(result, dict):
            for f in fields:
                if f not in result:
                    result[f] = ""

        self.status = f"Extracted {len(result)} items"
        self._cached_smart_extract = result
        return result

    def _do_split(self) -> list[str]:
        if hasattr(self, "_cached_smart_split"):
            return self._cached_smart_split

        prompt = self._build_split_prompt()
        llm = self.language_model
        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        segments = self._parse_json_array(response_text)
        segments = segments[:10]  # 最多10个

        self.status = f"Split into {len(segments)} segments"
        self._cached_smart_split = segments
        return segments

    def _get_fields(self) -> list[str]:
        return self._get_fields_list(getattr(self, "fields", None))

    # ------------------------------------------------------------------
    # Output methods
    # ------------------------------------------------------------------

    def extract_text(self, *args) -> Message:
        result = self._do_extract()
        return Message(text=json.dumps(result, ensure_ascii=False, indent=2))

    def extract_field(self, *args) -> Message:
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

    def extract_all(self, *args) -> Data:
        result = self._do_extract()
        return Data(data={"items": result} if isinstance(result, list) else result)

    def extract_json(self, *args) -> Message:
        result = self._do_extract()
        return Message(text=json.dumps(result, ensure_ascii=False, indent=2))

    def extract_data(self, *args) -> Data:
        result = self._do_extract()
        return Data(data={"items": result} if isinstance(result, list) else result)

    def extract_all_segments(self, *args) -> Message:
        segments = self._do_split()
        return Message(text=json.dumps(segments, ensure_ascii=False))