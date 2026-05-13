"""Add Fields component - add a key-value field to a Data object."""

from __future__ import annotations

from lfx.custom import Component
from lfx.io import DataInput, Output, StrInput
from lfx.schema.data import Data


class AddFieldsComponent(Component):
    display_name = "Add Fields"
    description = "向 Data 对象中添加一个字段。"
    icon = "plus-circle"
    name = "AddFields"

    inputs = [
        DataInput(
            name="input_data",
            display_name="Data",
            info="要添加字段的 Data 对象。",
            required=True,
        ),
        StrInput(
            name="field_key",
            display_name="Key",
            info="要添加的字段名。",
        ),
        StrInput(
            name="field_value",
            display_name="Value",
            info="要添加的字段值。",
        ),
    ]

    outputs = [
        Output(display_name="Data", name="data", method="add_fields", types=["Data"]),
    ]

    def add_fields(self) -> Data:
        if not self.input_data:
            self.status = "输入为空"
            return Data(data={})

        base = self.input_data.data if isinstance(self.input_data, Data) else {}
        key = (self.field_key or "").strip()
        value = self.field_value or ""

        if not key:
            self.status = "Key 为空，未添加字段"
            return Data(data=base)

        merged = {**base, key: value}
        self.status = f"已添加字段 '{key}'，共 {len(merged)} 个字段"
        return Data(data=merged)