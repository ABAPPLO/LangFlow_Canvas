"""AI Assistant endpoint powered by Claude Agent SDK.

Uses claude-agent-sdk for LLM calls with custom MCP tools for canvas manipulation.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import AsyncGenerator
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SdkMcpTool,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
)
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


class AssistantChatRequest(BaseModel):
    flow_id: str
    message: str
    selected_model: dict | None = None  # {name, provider, metadata} from Model Providers
    session_id: str | None = None


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


# ── System prompt ──────────────────────────────────────────────────────


SYSTEM_PROMPT = (
    "You are an AI assistant that helps users build workflows in Langflow. "
    "You can:\n"
    "1. List available components using list_components\n"
    "2. Build workflows on the canvas using build_flow\n"
    "3. Run the workflow to test it using run_flow\n\n"
    "IMPORTANT RULES:\n"
    "- The 'type' field in build_flow MUST match the exact 'type' value from "
    "list_components output (e.g. 'ChatInput', 'ChatOutput', 'LanguageModelComponent'). "
    "Do NOT use provider names like 'OpenAI' or 'Anthropic' as component types.\n"
    "- The 'source_output' MUST match the exact output name from list_components "
    "(e.g. 'message', 'text_output').\n"
    "- The 'target_input' MUST match the field name in the target component's template "
    "(e.g. 'input_value').\n"
    "- Always use list_components first to get exact type names and output names.\n"
    "- Edges are REQUIRED for the workflow to work. Always include edges.\n\n"
    "Always explain what you're doing before taking actions. "
    "Respond in the same language as the user's message."
)


# ── Tool context and MCP tool factory ──────────────────────────────────


class _ToolContext:
    """Holds per-request state shared across MCP tool invocations."""

    def __init__(self, flow_id: str, user_id: str | None = None):
        self.flow_id = flow_id
        self.user_id = user_id
        self.all_types: dict[str, Any] | None = None
        self.catalog: list[dict] | None = None
        # Populated by build_flow tool for SSE layer to pick up
        self.last_canvas_data: dict | None = None

    async def ensure_types(self) -> dict[str, Any]:
        if self.all_types is None:
            self.all_types = await _get_all_types_dict()
        return self.all_types

    async def list_components(self, category: str = "") -> dict:
        await self.ensure_types()
        if self.catalog is None:
            self.catalog = _build_component_catalog(self.all_types)  # type: ignore[arg-type]

        filtered = (
            [c for c in self.catalog if category.lower() in c["category"].lower()]
            if category
            else self.catalog
        )
        return {"content": [{"type": "text", "text": json.dumps(filtered, ensure_ascii=False)}]}

    async def build_flow(self, nodes: list[dict], edges: list[dict]) -> dict:
        await self.ensure_types()
        compact_data = {"nodes": nodes, "edges": edges}

        try:
            expanded = expand_compact_flow(compact_data, self.all_types)  # type: ignore[arg-type]
        except ValueError as e:
            return {"content": [{"type": "text", "text": f"Error building flow: {e}"}]}

        node_count = len(expanded.get("nodes", []))
        edge_count = len(expanded.get("edges", []))
        result_text = f"Canvas updated: {node_count} nodes, {edge_count} edges."

        self.last_canvas_data = expanded
        return {"content": [{"type": "text", "text": result_text}]}

    async def run_flow(self, input_value: str = "") -> dict:
        from langflow.api.v1.schemas import SimplifiedAPIRequest

        try:
            flow = await get_flow_by_id_or_endpoint_name(
                flow_id_or_name=self.flow_id,
                user_id=self.user_id,
            )
        except (ValueError, RuntimeError) as e:
            return {"content": [{"type": "text", "text": f"Error loading flow: {e}"}]}

        if flow.data is None:
            return {"content": [{"type": "text", "text": "Flow has no data. Please build the flow first."}]}

        input_request = SimplifiedAPIRequest(
            input_value=input_value or None,
            input_type="chat",
        )

        try:
            from langflow.api.v1.endpoints import simple_run_flow

            result = await simple_run_flow(
                flow=flow,
                input_request=input_request,
                stream=False,
            )

            outputs = result.outputs if hasattr(result, "outputs") else []
            results: list[dict] = []
            for output in outputs:
                if hasattr(output, "results"):
                    results.extend(
                        {"component": key, "result": str(value)}
                        for key, value in output.results.items()
                    )
                elif hasattr(output, "messages"):
                    results.extend({"message": str(msg)} for msg in output.messages)

            return {"content": [{"type": "text", "text": json.dumps(results, ensure_ascii=False)}]}
        except ValueError as e:
            return {"content": [{"type": "text", "text": f"Error running flow: {e}"}]}


def _create_mcp_server(ctx: _ToolContext):
    """Create a per-request MCP server with tools that close over the context."""

    async def list_components_handler(args: dict) -> dict:
        return await ctx.list_components(args.get("category", ""))

    async def build_flow_handler(args: dict) -> dict:
        return await ctx.build_flow(args.get("nodes", []), args.get("edges", []))

    async def run_flow_handler(args: dict) -> dict:
        return await ctx.run_flow(args.get("input_value", ""))

    tools = [
        SdkMcpTool(
            name="list_components",
            description=(
                "List all available components in the system. Returns component type, display name, "
                "description, category, and outputs. Use this to discover what components are available "
                "for building a workflow."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": "Optional category filter (e.g. 'inputs', 'models', 'outputs')",
                    },
                },
            },
            handler=list_components_handler,
        ),
        SdkMcpTool(
            name="build_flow",
            description=(
                "Build or update a workflow on the canvas. Provide nodes and edges. "
                "Nodes: [{id, type, values?}]. Edges: [{source, source_output, target, target_input}]. "
                "The 'type' field must be a component type from list_components. "
                "This replaces the entire canvas content."
            ),
            input_schema={
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
                                "source": {"type": "string"},
                                "source_output": {"type": "string"},
                                "target": {"type": "string"},
                                "target_input": {"type": "string"},
                            },
                            "required": ["source", "source_output", "target", "target_input"],
                        },
                        "description": "List of edges connecting nodes",
                    },
                },
                "required": ["nodes"],
            },
            handler=build_flow_handler,
        ),
        SdkMcpTool(
            name="run_flow",
            description=(
                "Run the current workflow on the canvas and return the results. "
                "Optionally provide an input_value to feed into the flow."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "input_value": {
                        "type": "string",
                        "description": "Optional input text to feed into the flow",
                    },
                },
            },
            handler=run_flow_handler,
        ),
    ]

    return create_sdk_mcp_server("langflow-tools", "1.0.0", tools)


# ── SSE streaming endpoint ─────────────────────────────────────────────


@router.post("/assistant/chat")
async def assistant_chat(
    request: Request,
    body: AssistantChatRequest,
    current_user: CurrentActiveUser,
):
    """SSE endpoint for AI assistant chat powered by Claude Agent SDK."""
    user_id = str(current_user.id) if current_user else None

    # Resolve credentials: prefer Model Providers, fall back to env var
    model_name = "claude-sonnet-4-20250514"
    env_overrides: dict[str, str] = {}

    selected = body.selected_model
    if selected and isinstance(selected, dict) and selected.get("name"):
        model_name = selected["name"]
        provider = selected.get("provider", "")

        try:
            from lfx.base.models.unified_models import (
                get_all_variables_for_provider,
                get_api_key_for_provider,
            )

            api_key = get_api_key_for_provider(user_id, provider)
            if api_key:
                env_overrides["ANTHROPIC_API_KEY"] = api_key

            provider_vars = get_all_variables_for_provider(user_id, provider)
            for var_key, value in provider_vars.items():
                if "BASE_URL" in var_key and value:
                    env_overrides["ANTHROPIC_BASE_URL"] = value
                    break
        except Exception:  # noqa: BLE001
            logger.warning("Failed to resolve credentials from Model Providers, using env vars")

    # Fall back to environment variable if no credentials from Model Providers
    if "ANTHROPIC_API_KEY" not in env_overrides:
        env_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not env_api_key:
            err_msg = (
                "No API key configured. "
                "Please select an Anthropic model from Model Providers "
                "or set the ANTHROPIC_API_KEY environment variable."
            )
            return StreamingResponse(
                iter([_sse_event("error", {"error": err_msg})]),
                media_type="text/event-stream",
            )
        env_overrides["ANTHROPIC_API_KEY"] = env_api_key

    async def event_generator() -> AsyncGenerator[str, None]:
        # Set up per-request tool context and MCP server
        ctx = _ToolContext(flow_id=body.flow_id, user_id=user_id)
        mcp_server = _create_mcp_server(ctx)

        options = ClaudeAgentOptions(
            model=model_name,
            system_prompt=SYSTEM_PROMPT,
            mcp_servers={"langflow": mcp_server},
            allowed_tools=[
                "mcp__langflow-tools__list_components",
                "mcp__langflow-tools__build_flow",
                "mcp__langflow-tools__run_flow",
            ],
            max_turns=10,
            resume=body.session_id,
            env=env_overrides,
        )

        yield _sse_event("connected", {"status": "ok"})

        try:
            async for message in query(prompt=body.message, options=options):
                if await request.is_disconnected():
                    break

                # SystemMessage — extract session_id
                if isinstance(message, SystemMessage):
                    if message.subtype == "init":
                        session_data = message.data or {}
                        sid = session_data.get("session_id", "")
                        if sid:
                            yield _sse_event("session_id", {"session_id": sid})
                    continue

                # AssistantMessage — extract text and tool interactions
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            if block.text:
                                yield _sse_event("token", {"text": block.text})

                        elif isinstance(block, ToolUseBlock):
                            yield _sse_event("tool_start", {
                                "name": block.name,
                                "input": block.input,
                            })

                        elif isinstance(block, ToolResultBlock):
                            # Extract text from tool result
                            result_text = ""
                            if isinstance(block.content, str):
                                result_text = block.content
                            elif isinstance(block.content, list):
                                for item in block.content:
                                    if isinstance(item, dict) and item.get("type") == "text":
                                        result_text += item.get("text", "")

                            # Find the matching tool name from ToolUseBlock in this message
                            tool_name = ""
                            for b in message.content:
                                if isinstance(b, ToolUseBlock) and b.id == block.tool_use_id:
                                    tool_name = b.name
                                    break

                            tool_result_data: dict = {
                                "name": tool_name,
                                "result": result_text[:500],
                            }

                            # Check if build_flow produced canvas data
                            if ctx.last_canvas_data is not None:
                                tool_result_data["canvas_updated"] = True

                            yield _sse_event("tool_result", tool_result_data)

                            # Emit canvas update separately
                            if ctx.last_canvas_data is not None:
                                yield _sse_event("canvas_update", ctx.last_canvas_data)
                                ctx.last_canvas_data = None

                    continue

                # ResultMessage — final result
                if isinstance(message, ResultMessage):
                    if message.result:
                        yield _sse_event("token", {"text": message.result})
                    continue

                # StreamEvent — ignored (partial tokens not needed for SSE)
                if isinstance(message, StreamEvent):
                    continue

        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Assistant chat error: {e}")
            yield _sse_event("error", {"error": str(e)})

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
