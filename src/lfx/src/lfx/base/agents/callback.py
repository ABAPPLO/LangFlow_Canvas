import json
import time
from typing import Any
from uuid import UUID

from langchain.callbacks.base import AsyncCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish

from lfx.schema.log import LogFunctionType


def _parse_json_text(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _parse_tool_input(input_str: str, inputs: dict[str, Any] | None) -> Any:
    if inputs is not None:
        return inputs
    return _parse_json_text(input_str)


def _parse_tool_output(output: Any) -> Any:
    if isinstance(output, str):
        return _parse_json_text(output)
    if isinstance(output, dict):
        return output
    content = getattr(output, "content", None)
    if content is not None:
        if len(content) == 1 and hasattr(content[0], "text"):
            return _parse_json_text(content[0].text)
        parsed_content = []
        for item in content:
            text = getattr(item, "text", None)
            parsed_content.append(_parse_json_text(text) if isinstance(text, str) else item)
        return parsed_content
    if hasattr(output, "model_dump"):
        return output.model_dump()
    return output


def _tool_output_error(raw_output: Any) -> dict[str, str] | None:
    if isinstance(raw_output, dict):
        error = raw_output.get("error")
        if raw_output.get("success") is False or error:
            if isinstance(error, dict):
                return {
                    "code": str(error.get("code") or "tool_execution_failed"),
                    "message": str(error.get("message") or error),
                }
            return {
                "code": "tool_execution_failed",
                "message": str(error or raw_output),
            }
        return None

    if isinstance(raw_output, str):
        text = raw_output.strip()
        lower_text = text.lower()
        if lower_text.startswith(("input validation error:", "invalid input:", "tool execution failed:")):
            return {
                "code": "tool_input_validation_error",
                "message": text,
            }
    return None


class ToolCallRecorder(AsyncCallbackHandler):
    """Record tool calls from the execution layer."""

    def __init__(self):
        self.tool_calls: list[dict[str, Any]] = []
        self._active_runs: dict[UUID, int] = {}

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        tags: list[str] | None = None,  # noqa: ARG002
        metadata: dict[str, Any] | None = None,  # noqa: ARG002
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        tool_name = (
            serialized.get("name")
            or kwargs.get("name")
            or serialized.get("id")
            or "unknown"
        )
        if isinstance(tool_name, list):
            tool_name = tool_name[-1] if tool_name else "unknown"

        record = {
            "tool_name": str(tool_name),
            "arguments": _parse_tool_input(input_str, inputs),
            "start_time": time.time(),
            "_perf_start": time.perf_counter(),
            "elapsed_ms": None,
            "raw_output": None,
            "success": None,
            "error": None,
        }
        self._active_runs[run_id] = len(self.tool_calls)
        self.tool_calls.append(record)

    async def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:  # noqa: ARG002
        index = self._active_runs.pop(run_id, None)
        if index is None:
            return
        record = self.tool_calls[index]
        raw_output = _parse_tool_output(output)
        error = _tool_output_error(raw_output)
        record["elapsed_ms"] = int((time.perf_counter() - record["_perf_start"]) * 1000)
        record["raw_output"] = raw_output
        record["success"] = error is None
        record["error"] = error
        record.pop("_perf_start", None)

    async def on_tool_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        index = self._active_runs.pop(run_id, None)
        if index is None:
            return
        record = self.tool_calls[index]
        record["elapsed_ms"] = int((time.perf_counter() - record["_perf_start"]) * 1000)
        record["success"] = False
        record["error"] = {
            "code": "tool_execution_failed",
            "message": str(error),
        }
        record.pop("_perf_start", None)


class AgentAsyncHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    def __init__(self, log_function: LogFunctionType | None = None):
        self.log_function = log_function

    async def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.log_function is None:
            return
        self.log_function(
            {
                "type": "chain_start",
                "serialized": serialized,
                "inputs": inputs,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "metadata": metadata,
                **kwargs,
            },
            name="Chain Start",
        )

    async def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.log_function is None:
            return
        self.log_function(
            {
                "type": "tool_start",
                "serialized": serialized,
                "input_str": input_str,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                "metadata": metadata,
                "inputs": inputs,
                **kwargs,
            },
            name="Tool Start",
        )

    async def on_tool_end(self, output: Any, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> None:
        if self.log_function is None:
            return
        self.log_function(
            {
                "type": "tool_end",
                "output": output,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                **kwargs,
            },
            name="Tool End",
        )

    async def on_agent_action(
        self,
        action: AgentAction,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.log_function is None:
            return
        self.log_function(
            {
                "type": "agent_action",
                "action": action,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                **kwargs,
            },
            name="Agent Action",
        )

    async def on_agent_finish(
        self,
        finish: AgentFinish,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if self.log_function is None:
            return
        self.log_function(
            {
                "type": "agent_finish",
                "finish": finish,
                "run_id": run_id,
                "parent_run_id": parent_run_id,
                "tags": tags,
                **kwargs,
            },
            name="Agent Finish",
        )
