"""Field Loop component - iterate through data rows and process specific fields in separate subgraphs."""

from __future__ import annotations

from collections import deque
from typing import Any

from lfx.base.flow_controls.loop_utils import (
    execute_loop_body,
    get_loop_body_start_edge,
    get_loop_body_start_vertex,
    get_loop_body_vertices,
    validate_data_input,
)
from lfx.custom.custom_component.component import Component
from lfx.inputs.inputs import HandleInput, MessageTextInput
from lfx.schema.data import Data
from lfx.schema.dataframe import DataFrame
from lfx.template.field.base import Output


class FieldLoopComponent(Component):
    display_name = "Field Loop"
    description = (
        "Iterate through data rows, processing specified fields in separate subgraphs. "
        "Each field gets its own output port and downstream processing chain."
    )
    documentation: str = "https://docs.langflow.org/field-loop"
    icon = "iterate"
    name = "FieldLoop"

    inputs = [
        HandleInput(
            name="data",
            display_name="Inputs",
            info="The table data to iterate over.",
            input_types=["DataFrame", "Table"],
        ),
        MessageTextInput(
            name="fields",
            display_name="Fields",
            info="Add field names. Each field becomes a loop output port.",
            is_list=True,
            list_add_label="Add Field",
            placeholder="Enter field name...",
            input_types=[],
            real_time_refresh=True,
        ),
    ]

    outputs: list[Output] = []

    def _get_fields(self) -> list[str]:
        """Get field names from the list input."""
        raw = getattr(self, "fields", None)
        if not raw:
            return []
        if isinstance(raw, str):
            return [raw.strip()] if raw.strip() else []
        if isinstance(raw, list):
            return [str(f).strip() for f in raw if str(f).strip()]
        return [str(raw).strip()]

    def update_outputs(self, frontend_node: dict, field_name: str, field_value: Any) -> dict:
        """Dynamically create loop output ports based on field list."""
        # Read current fields from frontend_node if not the fields field itself
        if field_name == "fields":
            raw_fields = field_value
        else:
            raw_fields = frontend_node.get("fields", {}).get("value", [])

        # Parse field list
        fields: list[str] = []
        if isinstance(raw_fields, list):
            fields = [str(f).strip() for f in raw_fields if str(f).strip()]
        elif isinstance(raw_fields, str) and raw_fields.strip():
            fields = [raw_fields.strip()]

        frontend_node["outputs"] = []

        # "Item" output — passes the whole row, like Loop's item output
        frontend_node["outputs"].append(
            Output(
                display_name="Item",
                name="item",
                method="item_output",
                types=["Data"],
                allows_loop=True,
                loop_types=["Message"],
                group_outputs=True,
            ),
        )

        # Per-field outputs
        for i, field in enumerate(fields):
            frontend_node["outputs"].append(
                Output(
                    display_name=field,
                    name=f"field_{i + 1}",
                    method="field_output",
                    types=["Data"],
                    allows_loop=True,
                    loop_types=["Message"],
                    group_outputs=True,
                ),
            )

        # Always add "Done" output for aggregated results
        frontend_node["outputs"].append(
            Output(
                display_name="Done",
                name="done",
                method="done_output",
                types=["DataFrame"],
                group_outputs=True,
            ),
        )

        return frontend_node

    def field_output(self) -> Data:
        """No-op — actual execution happens in done_output()."""
        self.stop(self._current_output)
        return Data(text="")

    def item_output(self) -> Data:
        """No-op — actual execution happens in done_output()."""
        self.stop("item")
        return Data(text="")

    def _get_field_configs(self, fields: list[str]) -> list[dict]:
        """Identify loop body configuration for each field.

        Supports two wiring patterns:
        1. Feedback loop: downstream component feeds back to this loop (standard Loop behavior)
        2. One-way chain: downstream component does NOT feed back (Field Loop specific)
        """
        if not hasattr(self, "_vertex") or self._vertex is None:
            return []

        configs = []
        for i, field_name in enumerate(fields):
            output_name = f"field_{i + 1}"

            # Try standard loop body detection first (works with feedback)
            body_vertices = get_loop_body_vertices(
                vertex=self._vertex,
                graph=self.graph,
                get_incoming_edge_by_target_param_fn=self.get_incoming_edge_by_target_param,
                loop_output_name=output_name,
            )
            start_vertex = get_loop_body_start_vertex(
                vertex=self._vertex,
                loop_output_name=output_name,
            )
            start_edge = get_loop_body_start_edge(
                vertex=self._vertex,
                loop_output_name=output_name,
            )
            end_vertex = self.get_incoming_edge_by_target_param(output_name)

            # If no feedback loop found, discover downstream chain forward
            if not body_vertices and start_edge:
                body_vertices, end_vertex = self._traverse_downstream(start_vertex)

            configs.append({
                "field_name": field_name,
                "output_name": output_name,
                "body_vertices": body_vertices,
                "start_vertex_id": start_vertex,
                "start_edge": start_edge,
                "end_vertex_id": end_vertex,
            })

        return configs

    def _traverse_downstream(self, start_vertex_id: str | None) -> tuple[set[str], str | None]:
        """BFS forward from a field's first downstream vertex to discover the full chain.

        Returns (vertex_ids, end_vertex_id). The end vertex is the last node
        in the chain (no successors within the discovered set).
        """
        if not start_vertex_id:
            return set(), None

        visited: set[str] = {self._vertex.id}
        queue = deque([start_vertex_id])
        vertices: set[str] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            vertices.add(current)

            for successor_id in self.graph.successor_map.get(current, []):
                if successor_id not in visited:
                    queue.append(successor_id)

        # Also include predecessors of discovered vertices (e.g. LLM model deps)
        self._add_predecessors(vertices)

        # End vertex: the one with no successors inside the chain
        end_vertex = start_vertex_id
        for vid in vertices:
            successors = self.graph.successor_map.get(vid, [])
            if not any(s in vertices for s in successors):
                end_vertex = vid

        return vertices, end_vertex

    def _add_predecessors(self, vertices: set[str]) -> None:
        """Recursively add all predecessors of vertices to include dependencies (e.g. LLM models)."""
        visited: set[str] = set()
        for vid in list(vertices):
            self._add_preds_recursive(vid, vertices, visited)

    def _add_preds_recursive(self, vertex_id: str, vertices: set[str], visited: set[str]) -> None:
        """Recursively add predecessors, excluding the loop component itself."""
        for pred_id, successors in self.graph.successor_map.items():
            if (
                vertex_id in successors
                and pred_id != self._vertex.id
                and pred_id not in visited
                and pred_id not in vertices
            ):
                visited.add(pred_id)
                vertices.add(pred_id)
                self._add_preds_recursive(pred_id, vertices, visited)

    async def done_output(self) -> DataFrame:
        """Iterate through rows and execute item + field subgraphs."""
        data_list = validate_data_input(self.data)
        if not data_list:
            return DataFrame([])

        fields = self._get_fields()

        # --- Item loop body (whole row, like Loop) ---
        item_body_vertices = set()
        item_start_vertex = None
        item_start_edge = None
        item_end_vertex = None
        if hasattr(self, "_vertex") and self._vertex is not None:
            item_body_vertices = get_loop_body_vertices(
                vertex=self._vertex,
                graph=self.graph,
                get_incoming_edge_by_target_param_fn=self.get_incoming_edge_by_target_param,
                loop_output_name="item",
            )
            item_start_vertex = get_loop_body_start_vertex(self._vertex, loop_output_name="item")
            item_start_edge = get_loop_body_start_edge(self._vertex, loop_output_name="item")
            item_end_vertex = self.get_incoming_edge_by_target_param("item")

        # --- Per-field loop bodies ---
        configs = self._get_field_configs(fields) if fields else []

        # Collect results
        field_results: dict[str, list[Data]] = {cfg["field_name"]: [] for cfg in configs}
        item_results: list[Data] = []

        for data_item in data_list:
            row_data = data_item.data if hasattr(data_item, "data") and isinstance(data_item.data, dict) else {}

            # Execute item loop body (whole row)
            if item_body_vertices:
                try:
                    results = await execute_loop_body(
                        graph=self.graph,
                        data_list=[data_item],
                        loop_body_vertex_ids=item_body_vertices,
                        start_vertex_id=item_start_vertex,
                        start_edge=item_start_edge,
                        end_vertex_id=item_end_vertex,
                        event_manager=self._event_manager,
                    )
                    if results:
                        item_results.append(results[0])
                except Exception as e:
                    from lfx.log.logger import logger

                    await logger.aerror(f"Field Loop error for item: {e}")
                    raise

            # Execute per-field loop bodies
            for config in configs:
                value = row_data.get(config["field_name"], "")

                if not config["body_vertices"]:
                    field_results[config["field_name"]].append(Data(text=str(value)))
                    continue

                value_data = Data(text=str(value))

                try:
                    results = await execute_loop_body(
                        graph=self.graph,
                        data_list=[value_data],
                        loop_body_vertex_ids=config["body_vertices"],
                        start_vertex_id=config["start_vertex_id"],
                        start_edge=config["start_edge"],
                        end_vertex_id=config["end_vertex_id"],
                        event_manager=self._event_manager,
                    )
                    if results:
                        field_results[config["field_name"]].append(results[0])
                    else:
                        field_results[config["field_name"]].append(Data(text=str(value)))
                except Exception as e:
                    from lfx.log.logger import logger

                    await logger.aerror(f"Field Loop error for field '{config['field_name']}': {e}")
                    raise

        # Build result DataFrame
        all_data: list[dict] = []

        # Prefer item results (whole-row processing) if available
        if item_results:
            for result in item_results:
                if isinstance(result, Data) and isinstance(result.data, dict):
                    all_data.append(result.data)
                else:
                    all_data.append({"result": str(result)})
        elif field_results:
            max_rows = max(len(v) for v in field_results.values()) if field_results else 0
            for row_idx in range(max_rows):
                row: dict = {}
                for config in configs:
                    results_list = field_results[config["field_name"]]
                    if row_idx < len(results_list):
                        result = results_list[row_idx]
                        if isinstance(result, Data):
                            row[config["field_name"]] = result.text if result.text else str(result.data)
                        else:
                            row[config["field_name"]] = str(result)
                    else:
                        row[config["field_name"]] = ""
                all_data.append(row)

        self.status = f"Processed {len(data_list)} rows x {len(fields)} fields"

        return DataFrame(all_data)
