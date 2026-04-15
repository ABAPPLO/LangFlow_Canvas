"""AI Assistant endpoint for building workflows on the canvas via natural language."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from lfx.log.logger import logger
from pydantic import BaseModel

from langflow.api.utils import CurrentActiveUser
from langflow.helpers.flow import get_flow_by_id_or_endpoint_name
from langflow.processing.expand_flow import expand_compact_flow

router = APIRouter(tags=["Assistant"])

SSE_HEARTBEAT_TIMEOUT = 15.0


# ── Request / Response models ──────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class AssistantChatRequest(BaseModel):
    flow_id: str
    message: str
    selected_model: dict  # {name, provider, metadata} from frontend ModelInput
    history: list[ChatMessage] = []


# ── Helpers ────────────────────────────────────────────────────────────


def _sse_event(event: str, data: dict) -> str:
    """Format a single SSE event."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _get_all_types_dict() -> dict[str, Any]:
    """Retrieve the full component types dictionary."""
    from langflow.interface.components import get_and_cache_all_types_dict
    from langflow.services.deps import get_settings_service

    return await get_and_cache_all_types_dict(settings_service=get_settings_service())


def _build_component_catalog(all_types: dict[str, Any]) -> list[dict]:
    """Build a lightweight component catalog for the AI."""
    catalog = []
    for _category, components in all_types.items():
        if not isinstance(components, dict):
            continue
        for comp_name, comp_data in components.items():
            if not isinstance(comp_data, dict):
                continue
            # Extract output names
            outputs = [
                out["name"]
                for out in comp_data.get("outputs", [])
                if isinstance(out, dict) and out.get("name")
            ]

            catalog.append({
                "type": comp_name,
                "display_name": comp_data.get("display_name", comp_name),
                "description": comp_data.get("description", ""),
                "category": _category,
                "outputs": outputs,
            })
    return catalog


# ── Tool definitions (OpenAI-compatible format for bind_tools) ─────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "list_components",
            "description": (
                "List all available components in the system. Returns component type, display name, "
                "description, category, and outputs. Use this to discover what components are available "
                "for building a workflow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g. 'inputs', 'models', 'outputs')",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_flow",
            "description": (
                "Build or update a workflow on the canvas. Provide nodes and edges in compact format. "
                "Nodes: [{id, type, values?}]. Edges: [{source, source_output, target, target_input}]. "
                "The 'type' field must be a component type from list_components (e.g. 'ChatInput', 'OpenAI'). "
                "'values' can override default field values. This replaces the entire canvas content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "values": {"type": "object"},
                            },
                            "required": ["id", "type"],
                        },
                        "description": "List of nodes to place on the canvas",
                    },
                    "edges": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {"type": "string", "description": "Source node ID"},
                                "source_output": {"type": "string", "description": "Source output name"},
                                "target": {"type": "string", "description": "Target node ID"},
                                "target_input": {"type": "string", "description": "Target input field name"},
                            },
                            "required": ["source", "source_output", "target", "target_input"],
                        },
                        "description": "List of edges connecting nodes",
                    },
                },
                "required": ["nodes"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_flow",
            "description": (
                "Run the current workflow on the canvas and return the results. "
                "Optionally provide an input_value to feed into the flow's input node."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "input_value": {
                        "type": "string",
                        "description": "Optional input text to feed into the flow",
                    },
                },
            },
        },
    },
]


# ── Tool execution ─────────────────────────────────────────────────────


class ToolExecutor:
    """Execute assistant tools and cache state across the conversation."""

    def __init__(self, flow_id: str, user_id: str | None = None):
        self.flow_id = flow_id
        self.user_id = user_id
        self._all_types: dict[str, Any] | None = None
        self._catalog: list[dict] | None = None

    async def _ensure_types(self) -> dict[str, Any]:
        if self._all_types is None:
            self._all_types = await _get_all_types_dict()
        return self._all_types

    async def execute(self, tool_name: str, tool_input: dict) -> tuple[str, dict | None]:
        """Execute a tool and return (result_text, optional_canvas_data)."""
        if tool_name == "list_components":
            return await self._list_components(tool_input)
        if tool_name == "build_flow":
            return await self._build_flow(tool_input)
        if tool_name == "run_flow":
            return await self._run_flow(tool_input)
        return f"Unknown tool: {tool_name}", None

    async def _list_components(self, params: dict) -> tuple[str, None]:
        all_types = await self._ensure_types()
        if self._catalog is None:
            self._catalog = _build_component_catalog(all_types)

        category_filter = params.get("category", "").lower()
        if category_filter:
            filtered = [c for c in self._catalog if category_filter in c["category"].lower()]
        else:
            filtered = self._catalog

        return json.dumps(filtered, ensure_ascii=False), None

    async def _build_flow(self, params: dict) -> tuple[str, dict | None]:
        all_types = await self._ensure_types()
        compact_data = {
            "nodes": params.get("nodes", []),
            "edges": params.get("edges", []),
        }

        try:
            expanded = expand_compact_flow(compact_data, all_types)
        except ValueError as e:
            return f"Error building flow: {e}", None

        node_count = len(expanded.get("nodes", []))
        edge_count = len(expanded.get("edges", []))
        result_text = f"Canvas updated: {node_count} nodes, {edge_count} edges."
        return result_text, expanded

    async def _run_flow(self, params: dict) -> tuple[str, None]:
        from langflow.api.v1.schemas import SimplifiedAPIRequest

        input_value = params.get("input_value")
        try:
            flow = await get_flow_by_id_or_endpoint_name(
                flow_id_or_name=self.flow_id,
                user_id=self.user_id,
            )
        except (ValueError, RuntimeError) as e:
            return f"Error loading flow: {e}", None

        if flow.data is None:
            return "Flow has no data. Please build the flow first.", None

        input_request = SimplifiedAPIRequest(
            input_value=input_value,
            input_type="chat",
        )

        try:
            from langflow.api.v1.endpoints import simple_run_flow

            result = await simple_run_flow(
                flow=flow,
                input_request=input_request,
                stream=False,
            )

            # Format output
            outputs = result.outputs if hasattr(result, "outputs") else []
            results: list[dict] = []
            for output in outputs:
                if hasattr(output, "results"):
                    results.extend(
                        {"component": key, "result": str(value)}
                        for key, value in output.results.items()
                    )
                elif hasattr(output, "messages"):
                    results.extend(
                        {"message": str(msg)}
                        for msg in output.messages
                    )

            return json.dumps(results, ensure_ascii=False), None
        except ValueError as e:
            return f"Error running flow: {e}", None


# ── LLM helpers ────────────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are an AI assistant that helps users build workflows in Langflow. "
    "You can:\n"
    "1. List available components using list_components\n"
    "2. Build workflows on the canvas using build_flow\n"
    "3. Run the workflow to test it using run_flow\n\n"
    "When building a workflow:\n"
    "- First use list_components to discover available components\n"
    "- Then use build_flow with nodes and edges in compact format\n"
    "- Each node needs a unique 'id' (like 'node1', 'node2') "
    "and a 'type' matching the component type\n"
    "- Edges connect outputs to inputs using "
    "source_output and target_input field names\n"
    "- After building, you can run_flow to test the workflow\n\n"
    "Always explain what you're doing before taking actions. "
    "Respond in the same language as the user's message."
)


def _create_llm(model_config: dict, user_id: str):
    """Create an LLM instance from model config using get_llm()."""
    from lfx.base.models.unified_models import get_llm

    return get_llm(
        model=[model_config],
        user_id=user_id,
        stream=True,
    )


# ── SSE streaming endpoint ─────────────────────────────────────────────


@router.post("/assistant/chat")
async def assistant_chat(
    request: Request,
    body: AssistantChatRequest,
    current_user: CurrentActiveUser,
):
    """SSE endpoint for AI assistant chat with tool_use for canvas manipulation."""
    if not body.selected_model or not body.selected_model.get("name"):
        return StreamingResponse(
            iter([_sse_event("error", {"error": "Please select a model first"})]),
            media_type="text/event-stream",
        )

    user_id = str(current_user.id) if current_user else None
    executor = ToolExecutor(flow_id=body.flow_id, user_id=user_id)

    async def event_generator() -> AsyncGenerator[str, None]:
        logger.info(f"Assistant chat: creating LLM for model={body.selected_model.get('name')}")
        try:
            llm = _create_llm(body.selected_model, user_id)
        except (ValueError, ImportError, KeyError, TypeError, RuntimeError, OSError):
            logger.exception("Assistant chat: failed to create LLM")
            yield _sse_event("error", {"error": "Failed to create model. Check model configuration and API key."})
            return

        logger.info("Assistant chat: binding tools to LLM")
        try:
            llm_with_tools = llm.bind_tools(TOOL_DEFINITIONS)
        except (ValueError, NotImplementedError, TypeError, AttributeError):
            logger.exception("Assistant chat: bind_tools failed")
            yield _sse_event("error", {"error": "Model does not support tool calling."})
            return

        # Build conversation using LangChain message objects for correct format handling
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        lc_messages: list = [
            SystemMessage(content=SYSTEM_PROMPT),
        ]
        lc_messages.extend(
            HumanMessage(content=msg.content) if msg.role == "user" else AIMessage(content=msg.content)
            for msg in body.history
        )
        lc_messages.append(HumanMessage(content=body.message))

        yield _sse_event("connected", {"status": "ok"})

        max_iterations = 10
        for iteration in range(max_iterations):
            try:
                if await request.is_disconnected():
                    break

                logger.info(f"Assistant chat: iteration {iteration + 1}, {len(lc_messages)} messages")

                # Use ainvoke for clean message format handling.
                # Streaming text tokens are sacrificed, but tool_call reliability is gained.
                ai_response: AIMessage = await llm_with_tools.ainvoke(lc_messages)

                # Stream text content to frontend
                text_content = ""
                if isinstance(ai_response.content, str):
                    text_content = ai_response.content
                elif isinstance(ai_response.content, list):
                    for block in ai_response.content:
                        if isinstance(block, str):
                            text_content += block
                        elif isinstance(block, dict) and block.get("type") == "text":
                            text_content += block.get("text", "")
                        elif hasattr(block, "text"):
                            text_content += block.text

                if text_content:
                    yield _sse_event("token", {"text": text_content})

                # Add AI response to message history
                lc_messages.append(ai_response)

                # Extract tool calls — support both OpenAI and Anthropic formats
                tool_calls: list[dict] = []
                if hasattr(ai_response, "tool_calls") and ai_response.tool_calls:
                    for tc in ai_response.tool_calls:
                        if isinstance(tc, dict):
                            tool_calls.append({
                                "id": tc.get("id", str(uuid.uuid4())),
                                "name": tc.get("name", ""),
                                "input": tc.get("args", {}),
                            })
                        else:
                            tool_calls.append({
                                "id": getattr(tc, "id", str(uuid.uuid4())),
                                "name": getattr(tc, "name", ""),
                                "input": getattr(tc, "args", {}),
                            })
                elif isinstance(ai_response.content, list):
                    tool_calls.extend(
                        {
                            "id": block.id,
                            "name": block.name,
                            "input": block.input if hasattr(block, "input") else {},
                        }
                        for block in ai_response.content
                        if hasattr(block, "type") and block.type == "tool_use"
                    )

                # No tool calls — we're done
                if not tool_calls:
                    break

                # Execute each tool call and collect results
                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_input = tc.get("input", {})
                    tool_id = tc["id"]

                    yield _sse_event("tool_start", {
                        "name": tool_name,
                        "input": tool_input,
                    })

                    result_text, canvas_data = await executor.execute(tool_name, tool_input)

                    tool_result_data: dict = {
                        "name": tool_name,
                        "result": result_text[:500],
                    }
                    if canvas_data:
                        tool_result_data["canvas_updated"] = True

                    yield _sse_event("tool_result", tool_result_data)

                    if canvas_data:
                        yield _sse_event("canvas_update", canvas_data)

                    # Append ToolMessage with matching tool_call_id
                    lc_messages.append(ToolMessage(
                        content=result_text,
                        tool_call_id=tool_id,
                    ))

            except asyncio.CancelledError:
                break
            except (ValueError, KeyError, TypeError, RuntimeError, OSError) as e:
                logger.exception(f"Assistant chat error: {e}")
                yield _sse_event("error", {"error": str(e)})
                break

        yield _sse_event("done", {})

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
