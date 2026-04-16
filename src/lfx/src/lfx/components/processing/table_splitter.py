"""Table Splitter component for splitting table data into multiple outputs."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lfx.custom import Component
from lfx.io import DropdownInput, HandleInput, IntInput, Output, StrInput
from lfx.log.logger import logger
from lfx.schema.dataframe import DataFrame


class TableSplitterComponent(Component):
    display_name = "Table Splitter"
    description = "Split table or JSON data into multiple outputs by rows, columns, or custom count."
    icon = "table"
    name = "TableSplitter"
    metadata = {
        "keywords": [
            "split",
            "table",
            "dataframe",
            "partition",
            "divide",
            "batch",
            "columns",
            "rows",
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
        DropdownInput(
            name="split_mode",
            display_name="Split Mode",
            info="How to split the data.",
            options=["By Rows", "By Columns", "Custom"],
            value="By Rows",
            real_time_refresh=True,
        ),
        IntInput(
            name="batch_size",
            display_name="Batch Size",
            info="Number of rows per output when splitting by rows.",
            value=1,
            real_time_refresh=True,
        ),
        IntInput(
            name="number_of_outputs",
            display_name="Number of Outputs",
            info="Number of output ports in Custom mode.",
            value=2,
            real_time_refresh=True,
        ),
        StrInput(
            name="output_names",
            display_name="Output Names",
            info='Comma-separated display names for outputs (optional). E.g. "Group A,Group B,Group C"',
            value="",
            advanced=True,
        ),
    ]

    outputs: list[Output] = []

    def _to_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert input data to a pandas DataFrame."""
        if isinstance(data, DataFrame):
            return data
        if isinstance(data, pd.DataFrame):
            return data
        if isinstance(data, list):
            if len(data) > 0 and hasattr(data[0], "data"):
                # List of Data objects
                return pd.DataFrame([d.data for d in data])
            # List of dicts
            return pd.DataFrame(data)
        if isinstance(data, dict):
            # Single dict -> one-row DataFrame
            return pd.DataFrame([data])
        msg = f"Unsupported data type: {type(data).__name__}"
        raise ValueError(msg)

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Dynamically create output ports based on split configuration."""
        if field_name not in {"split_mode", "batch_size", "number_of_outputs"}:
            return frontend_node

        frontend_node["outputs"] = []

        split_mode = field_value if field_name == "split_mode" else getattr(self, "split_mode", "By Rows")

        # Parse custom output names
        names_str = field_value if field_name == "output_names" else getattr(self, "output_names", "")
        custom_names = [n.strip() for n in names_str.split(",") if n.strip()] if names_str else []

        if split_mode == "By Rows":
            batch_size = field_value if field_name == "batch_size" else getattr(self, "batch_size", 1)
            count = max(batch_size, 1)
            # We don't know the row count at config time, show a reasonable preview
            for i in range(count):
                display = custom_names[i] if i < len(custom_names) else f"Batch {i + 1}"
                frontend_node["outputs"].append(
                    Output(
                        display_name=display,
                        name=f"batch_{i + 1}",
                        method="get_split_part",
                        types=["Table"],
                        group_outputs=True,
                    ),
                )

        elif split_mode == "By Columns":
            # Show placeholder outputs - actual count depends on data columns at runtime
            # Default to 3 preview columns
            count = 3
            for i in range(count):
                display = custom_names[i] if i < len(custom_names) else f"Column {i + 1}"
                frontend_node["outputs"].append(
                    Output(
                        display_name=display,
                        name=f"column_{i + 1}",
                        method="get_split_part",
                        types=["Table"],
                        group_outputs=True,
                    ),
                )

        elif split_mode == "Custom":
            count = field_value if field_name == "number_of_outputs" else getattr(self, "number_of_outputs", 2)
            count = max(int(count), 1)
            for i in range(count):
                display = custom_names[i] if i < len(custom_names) else f"Output {i + 1}"
                frontend_node["outputs"].append(
                    Output(
                        display_name=display,
                        name=f"output_{i + 1}",
                        method="get_split_part",
                        types=["Table"],
                        group_outputs=True,
                    ),
                )

        return frontend_node

    def _do_split(self) -> list[pd.DataFrame]:
        """Split the input data and cache the result."""
        if hasattr(self, "_split_cache") and self._split_cache is not None:
            return self._split_cache

        data = self.data
        df = self._to_dataframe(data)
        split_mode = getattr(self, "split_mode", "By Rows")
        parts: list[pd.DataFrame] = []

        if split_mode == "By Rows":
            batch_size = max(getattr(self, "batch_size", 1), 1)
            parts.extend(
                df.iloc[start : start + batch_size].reset_index(drop=True) for start in range(0, len(df), batch_size)
            )

        elif split_mode == "By Columns":
            parts.extend(df[[col]].copy() for col in df.columns)

        elif split_mode == "Custom":
            n_outputs = max(getattr(self, "number_of_outputs", 2), 1)
            chunk_size = max(len(df) // n_outputs, 1)
            for i in range(n_outputs):
                start = i * chunk_size
                if i == n_outputs - 1:
                    # Last chunk gets remaining rows
                    parts.append(df.iloc[start:].reset_index(drop=True))
                else:
                    end = start + chunk_size
                    parts.append(df.iloc[start:end].reset_index(drop=True))

        if not parts:
            parts = [df]

        self._split_cache = parts
        self.status = f"Split into {len(parts)} parts ({split_mode})"
        logger.debug(f"TableSplitter: {split_mode} -> {len(parts)} parts, sizes: {[len(p) for p in parts]}")
        return parts

    def get_split_part(self) -> DataFrame:
        """Return the data partition for the current output port."""
        parts = self._do_split()

        # Determine which output is being called via _current_output
        output_name = getattr(self, "_current_output", "")
        split_mode = getattr(self, "split_mode", "By Rows")

        # Match output name to index
        if split_mode == "By Rows":
            # batch_1 -> 0, batch_2 -> 1, ...
            idx = 0
            if output_name.startswith("batch_"):
                try:
                    idx = int(output_name.split("_")[1]) - 1
                except (ValueError, IndexError):
                    idx = 0
        elif split_mode == "By Columns":
            # column_1 -> 0, column_2 -> 1, ...
            idx = 0
            if output_name.startswith("column_"):
                try:
                    idx = int(output_name.split("_")[1]) - 1
                except (ValueError, IndexError):
                    idx = 0
        elif split_mode == "Custom":
            # output_1 -> 0, output_2 -> 1, ...
            idx = 0
            if output_name.startswith("output_"):
                try:
                    idx = int(output_name.split("_")[1]) - 1
                except (ValueError, IndexError):
                    idx = 0
        else:
            idx = 0

        result = parts[idx] if idx < len(parts) else (parts[-1] if parts else pd.DataFrame())

        return DataFrame(result)
