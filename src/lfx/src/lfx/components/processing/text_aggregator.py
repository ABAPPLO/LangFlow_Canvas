"""Text Aggregator component - aggregate multiple text inputs into one output."""

from lfx.custom import Component
from lfx.inputs import DropdownInput, IntInput
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

INPUT_FIELD_PREFIX = "text_"


class TextAggregatorComponent(Component):
    display_name = "Text Aggregator"
    description = "聚合多个文本输入，每个输入端口独立，支持手动输入或从其他组件连接。"
    icon = "merge"
    name = "TextAggregator"

    inputs = [
        IntInput(
            name="input_count",
            display_name="Input Count",
            info="输入端口数量，修改后自动增减端口。",
            value=2,
            real_time_refresh=True,
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

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "input_count":
            count = max(1, int(field_value)) if field_value else 2

            # Remove old dynamic text fields
            to_remove = [k for k in build_config if k.startswith(INPUT_FIELD_PREFIX) and k[len(INPUT_FIELD_PREFIX):].isdigit()]
            for k in to_remove:
                del build_config[k]

            # Create individual input fields with their own handles
            for i in range(1, count + 1):
                field_name_i = f"{INPUT_FIELD_PREFIX}{i}"
                build_config[field_name_i] = {
                    "type": "str",
                    "input_types": ["Message"],
                    "name": field_name_i,
                    "display_name": f"Text {i}",
                    "value": "",
                    "show": True,
                    "advanced": False,
                    "multiline": True,
                    "placeholder": "Enter text or connect...",
                }

        return build_config

    def _resolve_texts(self) -> list[str]:
        """Collect text from all dynamic input fields."""
        texts: list[str] = []
        i = 1
        while True:
            val = getattr(self, f"{INPUT_FIELD_PREFIX}{i}", None)
            if val is None:
                break
            if isinstance(val, Message):
                text = val.get_text()
            elif isinstance(val, str):
                text = val
            elif val:
                text = str(val)
            else:
                text = ""
            if text.strip():
                texts.append(text.strip())
            i += 1
        return texts

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
