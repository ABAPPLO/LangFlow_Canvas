"""Field Extractor component - extract specified fields from structured data into separate outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lfx.custom import Component
from lfx.io import HandleInput, MessageTextInput, Output
from lfx.log.logger import logger
from lfx.schema.dataframe import DataFrame


class FieldExtractorComponent(Component):
    display_name = "Field Extractor"
    description = "Extract specified fields from structured data into separate output ports."
    icon = "columns-3"
    name = "FieldExtractor"
    metadata = {
        "keywords": [
            "extract",
            "field",
            "column",
            "split",
            "table",
            "dataframe",
        ],
    }

    inputs = [
        HandleInput(
            name="data",
            display_name="Data",
            info="Input table data (Table, DataFrame, or JSON list of objects).",
            input_types=["Data", "JSON", "DataFrame", "Table"],
            required=True,
        ),
        MessageTextInput(
            name="fields",
            display_name="Fields",
            info="Comma-separated field names to extract. Each field becomes a separate output port.",
            value="",
            placeholder="name, email, phone",
            real_time_refresh=True,
        ),
    ]

    outputs: list[Output] = []

    def _parse_fields(self, field_value: Any = None) -> list[str]:
        """Parse comma-separated field names into a list."""
        raw = field_value if field_value is not None else getattr(self, "fields", "")
        if not raw:
            return []
        return [f.strip() for f in str(raw).split(",") if f.strip()]

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert input data to a pandas DataFrame."""
        if isinstance(data, DataFrame):
            return data
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list):
            if len(data) > 0 and hasattr(data[0], "data"):
                return pd.DataFrame([d.data for d in data])
            return pd.DataFrame(data)
        if isinstance(data, dict):
            return pd.DataFrame([data])
        msg = f"Unsupported data type: {type(data).__name__}"
        raise ValueError(msg)

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Dynamically create output ports based on field names."""
        if field_name != "fields":
            return frontend_node

        frontend_node["outputs"] = []
        fields = self._parse_fields(field_value)

        for i, field_name_str in enumerate(fields):
            frontend_node["outputs"].append(
                Output(
                    display_name=field_name_str,
                    name=f"field_{i + 1}",
                    method="extract_field",
                    types=["Table"],
                    group_outputs=True,
                ),
            )

        return frontend_node

    def extract_field(self) -> DataFrame:
        """Extract the field for the current output port."""
        data = self.data
        df = self._to_dataframe(data)

        # Determine field index from _current_output (field_1 -> 0, field_2 -> 1, ...)
        output_name = getattr(self, "_current_output", "")
        idx = 0
        if output_name.startswith("field_"):
            try:
                idx = int(output_name.split("_")[1]) - 1
            except (ValueError, IndexError):
                idx = 0

        fields = self._parse_fields()
        if not fields:
            logger.warning("Field Extractor: no fields defined")
            return DataFrame(pd.DataFrame())

        if idx >= len(fields):
            logger.warning(f"Field Extractor: output index {idx} exceeds field count {len(fields)}")
            return DataFrame(pd.DataFrame())

        field_name = fields[idx]

        if field_name not in df.columns:
            logger.warning(f"Field Extractor: field '{field_name}' not found in data columns: {list(df.columns)}")
            return DataFrame(pd.DataFrame())

        result_df = df[[field_name]].copy()
        logger.debug(f"Field Extractor: extracted '{field_name}' ({len(result_df)} rows)")
        self.status = f"Extracted '{field_name}': {len(result_df)} rows"

        return DataFrame(result_df)
