"""Text Aggregator component - aggregate multiple text inputs into one output."""

from lfx.custom import Component
from lfx.inputs import DropdownInput, MessageTextInput
from lfx.io import Output
from lfx.schema.data import Data
from lfx.schema.message import Message

FORMAT_MESSAGE = "Message"
FORMAT_JSON = "JSON"

SEPARATOR_NEWLINE = "\\n"
SEPARATOR_SPACE = " "
SEPARATOR_COMMA = ", "
SEPARATOR_DASH = "---"
SEPARATOR_EMPTY = "(empty)"


class TextAggregatorComponent(Component):
    display_name = "Text Aggregator"
    description = "聚合多个文本输入，支持手动输入或从其他组件连接。"
    icon = "merge"
    name = "TextAggregator"

    inputs = [
        MessageTextInput(
            name="texts",
            display_name="Texts",
            info="多个文本输入。点击 Add Text 添加更多，或从其他组件连接。",
            placeholder="Enter text...",
            input_types=["Message", "Text"],
            is_list=True,
            list_add_label="Add Text",
        ),
        DropdownInput(
            name="separator",
            display_name="Separator",
            info="合并文本时使用的分隔符。",
            options=[
                SEPARATOR_NEWLINE,
                SEPARATOR_SPACE,
                SEPARATOR_COMMA,
                SEPARATOR_DASH,
                SEPARATOR_EMPTY,
            ],
            value=SEPARATOR_NEWLINE,
            advanced=True,
        ),
        DropdownInput(
            name="output_format",
            display_name="Output Format",
            info="输出格式：Message 为合并文本，JSON 为结构化数据。",
            options=[FORMAT_MESSAGE, FORMAT_JSON],
            value=FORMAT_MESSAGE,
        ),
    ]

    outputs = [
        Output(
            display_name="Combined",
            name="combined",
            method="aggregate",
        ),
    ]

    def _resolve_texts(self) -> list[str]:
        """Extract text strings from the list input."""
        raw = self.texts
        if not raw:
            return []

        result: list[str] = []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, Message):
                    text = item.get_text()
                else:
                    text = str(item) if item else ""
                if text.strip():
                    result.append(text.strip())
        elif isinstance(raw, Message):
            text = raw.get_text()
            if text.strip():
                result.append(text.strip())
        elif isinstance(raw, str) and raw.strip():
            result.append(raw.strip())

        return result

    def _get_separator(self) -> str:
        """Resolve the separator string."""
        mapping = {
            SEPARATOR_NEWLINE: "\n",
            SEPARATOR_SPACE: " ",
            SEPARATOR_COMMA: ", ",
            SEPARATOR_DASH: "\n---\n",
            SEPARATOR_EMPTY: "",
        }
        return mapping.get(self.separator, self.separator)

    def aggregate(self) -> Message | Data:
        """Aggregate all text inputs."""
        texts = self._resolve_texts()
        sep = self._get_separator()
        combined = sep.join(texts)

        if self.output_format == FORMAT_JSON:
            self.status = f"Aggregated {len(texts)} texts as JSON"
            return Data(data={
                "texts": texts,
                "count": len(texts),
                "combined": combined,
            })

        self.status = f"Aggregated {len(texts)} texts"
        return Message(text=combined)
