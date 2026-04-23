"""AI Assistant endpoint using direct HTTP calls to LLM providers.

Hybrid approach: tries OpenAI native tool calling first, falls back to prompt-based.
Includes session persistence for multi-turn conversations.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import uuid
from collections.abc import AsyncGenerator
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from lfx.log.logger import logger
from pydantic import BaseModel

from langflow.api.utils import CurrentActiveUser
from langflow.helpers.flow import get_flow_by_id_or_endpoint_name
from langflow.processing.expand_flow import expand_compact_flow

router = APIRouter(tags=["Assistant"])

# ── Request / Response models ──────────────────────────────────────────


class AssistantChatRequest(BaseModel):
    flow_id: str
    message: str
    selected_model: dict | None = None  # {name, provider, metadata} from Model Providers
    session_id: str | None = None


# ── Priority components ────────────────────────────────────────────────

PRIORITY_COMPONENTS = {
    "ChatInput", "ChatOutput", "TextInput", "TextOutput",
    "LanguageModelComponent", "OpenAIModel", "AnthropicModel",
    "Prompt", "ConversationComponent", "MemoryComponent",
    "RecursiveCharacterTextSplitter", "FAISS",
}

# ── Tool definitions (OpenAI function-calling format) ──────────────────

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_components",
            "description": (
                "Search available components. Returns type, display_name, description, category, outputs, priority. "
                "When multiple similar components exist, prefer priority=recommended."
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
            "name": "get_component_details",
            "description": (
                "Get detailed info for a specific component: all configurable fields (name, type, default, "
                "input_types, options) and outputs (name, types). Use to check if components can connect."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "description": "The component type name (e.g. 'ChatInput')",
                    },
                },
                "required": ["type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_flow",
            "description": (
                "Build or update a workflow on the canvas. "
                "Nodes: [{id, type, values?}]. Edges: [{source, source_output, target, target_input}]. "
                "The 'type' must be from list_components. 'values' sets component parameters. "
                "Edges are REQUIRED. This replaces the entire canvas."
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
            "description": "Run the current workflow and return results. Optionally provide input_value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_value": {
                        "type": "string",
                        "description": "Optional input text to test the flow",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_flow",
            "description": "Save the current workflow permanently to the database. Call after testing passes.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


# ── Session store (in-memory) ──────────────────────────────────────────

_sessions: dict[str, list[dict[str, Any]]] = {}
_MAX_SESSION_MESSAGES = 50


def _get_session_messages(session_id: str) -> list[dict[str, Any]]:
    return _sessions.get(session_id, [])


def _save_session_messages(session_id: str, messages: list[dict[str, Any]]) -> None:
    # Keep only last N messages (excluding system prompt which is always first)
    _sessions[session_id] = messages[-_MAX_SESSION_MESSAGES:]


# ── Helpers ────────────────────────────────────────────────────────────


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def _get_all_types_dict() -> dict[str, Any]:
    from langflow.interface.components import get_and_cache_all_types_dict
    from langflow.services.deps import get_settings_service

    return await get_and_cache_all_types_dict(settings_service=get_settings_service())


def _build_component_catalog(all_types: dict[str, Any]) -> list[dict]:
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
                "priority": "recommended" if comp_name in PRIORITY_COMPONENTS else "normal",
            })
    catalog.sort(key=lambda c: (0 if c["priority"] == "recommended" else 1, c["type"]))
    return catalog


def _extract_component_details(comp_data: dict) -> dict:
    template = comp_data.get("template", {})
    fields = []
    for field_name, field_data in template.items():
        if not isinstance(field_data, dict):
            continue
        if field_name.startswith("_") or field_name in ("code", "trace_as_metadata"):
            continue
        field_info: dict[str, Any] = {
            "name": field_name,
            "display_name": field_data.get("display_name", field_name),
            "type": field_data.get("type", "str"),
        }
        if field_data.get("required"):
            field_info["required"] = True
        if field_data.get("value") is not None:
            field_info["default"] = field_data["value"]
        if field_data.get("input_types"):
            field_info["input_types"] = field_data["input_types"]
        if field_data.get("options"):
            field_info["options"] = field_data["options"]
        if field_data.get("info"):
            field_info["description"] = field_data["info"]
        fields.append(field_info)

    outputs = []
    for out in comp_data.get("outputs", []):
        if isinstance(out, dict) and out.get("name"):
            out_info: dict[str, Any] = {"name": out["name"]}
            if out.get("types"):
                out_info["types"] = out["types"]
            if out.get("display_name"):
                out_info["display_name"] = out["display_name"]
            outputs.append(out_info)

    return {
        "display_name": comp_data.get("display_name", ""),
        "description": comp_data.get("description", ""),
        "fields": fields,
        "outputs": outputs,
    }


# ── System prompt ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an AI assistant that helps users build workflows in Langflow.

TOOL CALL FORMAT — When you need to call a tool, you MUST output EXACTLY this format:
<tool_call_name>tool_name_here</tool_call_name>
<tool_call_args>{"arg1": "value1"}</tool_call_args>

Do NOT just describe what you want to do. You MUST actually output the XML tags above to invoke tools.
You may include explanatory text, but the XML tool call blocks MUST be present in your response.
You may call multiple tools in one response.

AVAILABLE TOOLS:

1. **list_components** — List available components.
   Args: {"category": "optional filter"}
   Prefer components with priority="recommended" when multiple similar ones exist.

2. **get_component_details** — Get fields and outputs for a component.
   Args: {"type": "component_type_name"}
   Use to check if components can connect (output types must match input types).

3. **build_flow** — Build workflow on canvas.
   Args: {"nodes": [{id, type, values?}], "edges": [{source, source_output, target, target_input}]}
   - "type" MUST be from list_components (e.g. "ChatInput", NOT "OpenAI").
   - "source_output" MUST match output name (e.g. "message").
   - "target_input" MUST match field name (e.g. "input_value").
   - Edges are REQUIRED.

4. **run_flow** — Test the workflow.
   Args: {"input_value": "optional test input"}

5. **save_flow** — Save workflow to database.
   Args: {} (call after testing passes)

WORKFLOW: list_components → get_component_details → build_flow → run_flow → save_flow
If a step fails, retry with adjustments. Max 3 retries per step.

Respond in the same language as the user's message.
"""

# ── Prompt-based tool calling regex ────────────────────────────────────

_TOOL_NAME_RE = re.compile(r"<tool_call_name>(.*?)</tool_call_name>", re.DOTALL)
_TOOL_ARGS_RE = re.compile(r"<tool_call_args>(.*?)</tool_call_args>", re.DOTALL)


def _parse_tool_calls_from_text(text: str) -> list[tuple[str, dict, int, int]]:
    results = []
    for name_match in _TOOL_NAME_RE.finditer(text):
        name = name_match.group(1).strip()
        name_end = name_match.end()
        args_match = _TOOL_ARGS_RE.search(text, name_end)
        if not args_match or args_match.start() > name_end + 200:
            continue
        try:
            args = json.loads(args_match.group(1).strip())
        except json.JSONDecodeError:
            args = {}
        results.append((name, args, name_match.start(), args_match.end()))
    return results


# ── Tool context ───────────────────────────────────────────────────────


class _ToolContext:
    def __init__(self, flow_id: str, user_id: str | None = None):
        self.flow_id = flow_id
        self.user_id = user_id
        self.all_types: dict[str, Any] | None = None
        self.catalog: list[dict] | None = None
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

    async def get_component_details(self, type_name: str) -> dict:
        await self.ensure_types()
        for _category, components in self.all_types.items():  # type: ignore[union-attr]
            if not isinstance(components, dict):
                continue
            comp_data = components.get(type_name)
            if isinstance(comp_data, dict):
                details = _extract_component_details(comp_data)
                return {"content": [{"type": "text", "text": json.dumps(details, ensure_ascii=False)}]}
        return {"content": [{"type": "text", "text": f"Component type '{type_name}' not found."}]}

    async def build_flow(self, nodes: list[dict], edges: list[dict]) -> dict:
        await self.ensure_types()
        compact_data = {"nodes": nodes, "edges": edges}
        logger.warning("[Assistant] build_flow called: nodes=%d edges=%d", len(nodes), len(edges))
        try:
            expanded = expand_compact_flow(compact_data, self.all_types)  # type: ignore[arg-type]
        except ValueError as e:
            logger.warning("[Assistant] expand_compact_flow error: %s", e)
            return {"content": [{"type": "text", "text": f"Error building flow: {e}"}]}

        node_count = len(expanded.get("nodes", []))
        edge_count = len(expanded.get("edges", []))
        logger.warning("[Assistant] expanded: %d nodes, %d edges", node_count, edge_count)
        self.last_canvas_data = expanded
        return {"content": [{"type": "text", "text": f"Canvas updated: {node_count} nodes, {edge_count} edges."}]}

    async def run_flow(self, input_value: str = "") -> dict:
        from langflow.api.v1.schemas import SimplifiedAPIRequest

        try:
            flow = await get_flow_by_id_or_endpoint_name(
                flow_id_or_name=self.flow_id, user_id=self.user_id,
            )
        except (ValueError, RuntimeError) as e:
            return {"content": [{"type": "text", "text": f"Error loading flow: {e}"}]}

        if flow.data is None:
            return {"content": [{"type": "text", "text": "Flow has no data. Please build the flow first."}]}

        input_request = SimplifiedAPIRequest(input_value=input_value or None, input_type="chat")

        try:
            from langflow.api.v1.endpoints import simple_run_flow

            result = await simple_run_flow(flow=flow, input_request=input_request, stream=False)
            outputs = result.outputs if hasattr(result, "outputs") else []
            results: list[dict] = []
            for output in outputs:
                if hasattr(output, "results"):
                    results.extend({"component": k, "result": str(v)} for k, v in output.results.items())
                elif hasattr(output, "messages"):
                    results.extend({"message": str(msg)} for msg in output.messages)
            return {"content": [{"type": "text", "text": json.dumps(results, ensure_ascii=False)}]}
        except ValueError as e:
            return {"content": [{"type": "text", "text": f"Error running flow: {e}"}]}

    async def save_flow(self) -> dict:
        try:
            flow = await get_flow_by_id_or_endpoint_name(
                flow_id_or_name=self.flow_id, user_id=self.user_id,
            )
        except (ValueError, RuntimeError) as e:
            return {"content": [{"type": "text", "text": f"Error loading flow: {e}"}]}

        canvas_data = self.last_canvas_data or flow.data
        if canvas_data is None:
            return {"content": [{"type": "text", "text": "No canvas data to save. Build the flow first."}]}

        try:
            from langflow.services.deps import session_scope

            async with session_scope() as session:
                from sqlmodel import select
                from langflow.services.database.models.flow.model import Flow

                db_flow = (await session.exec(select(Flow).where(Flow.id == self.flow_id))).first()
                if not db_flow:
                    return {"content": [{"type": "text", "text": f"Flow {self.flow_id} not found."}]}

                db_flow.data = canvas_data
                session.add(db_flow)
                await session.commit()

            logger.warning("[Assistant] flow %s saved", self.flow_id)
            return {"content": [{"type": "text", "text": "Workflow saved successfully."}]}
        except Exception as e:  # noqa: BLE001
            logger.exception("[Assistant] save_flow error: %s", e)
            return {"content": [{"type": "text", "text": f"Error saving flow: {e}"}]}


# ── Provider credential resolution ─────────────────────────────────────


def _resolve_provider_credentials(
    selected_model: dict | None,
    user_id: str | None,
) -> tuple[str, str, str]:
    model_name = "claude-sonnet-4-20250514"
    api_key = ""
    base_url = ""

    selected = selected_model
    if selected and isinstance(selected, dict) and selected.get("name"):
        model_name = selected["name"]
        provider = selected.get("provider", "")
        logger.warning("[Assistant] selected model=%s provider=%s", model_name, provider)

        try:
            from lfx.base.models.unified_models import (
                get_all_variables_for_provider,
                get_api_key_for_provider,
            )

            api_key = get_api_key_for_provider(user_id, provider) or ""
            provider_vars = get_all_variables_for_provider(user_id, provider)
            for var_key, value in provider_vars.items():
                if "BASE_URL" in var_key and value:
                    base_url = value
                    break
            logger.warning("[Assistant] resolved api_key=%s base_url=%s", bool(api_key), base_url or "N/A")
        except Exception:  # noqa: BLE001
            logger.warning("Failed to resolve credentials from Model Providers", exc_info=True)

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not base_url:
        base_url = os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("OPENAI_API_BASE", "")

    if not api_key:
        msg = "No API key configured. Please select a model from Model Providers or set the OPENAI_API_KEY / ANTHROPIC_API_KEY environment variable."
        raise ValueError(msg)

    if base_url:
        base_url = base_url.rstrip("/")
        if base_url.endswith("/v1"):
            base_url = base_url[:-3]
        api_url = f"{base_url}/v1/chat/completions"
    else:
        api_url = "https://api.openai.com/v1/chat/completions"

    logger.warning("[Assistant] final api_url=%s model=%s", api_url, model_name)
    return api_key, api_url, model_name


# ── Tool execution ─────────────────────────────────────────────────────


async def _execute_tool(ctx: _ToolContext, tool_name: str, arguments: dict) -> str:
    try:
        if tool_name == "list_components":
            result = await ctx.list_components(arguments.get("category", ""))
        elif tool_name == "get_component_details":
            result = await ctx.get_component_details(arguments.get("type", ""))
        elif tool_name == "build_flow":
            result = await ctx.build_flow(arguments.get("nodes", []), arguments.get("edges", []))
        elif tool_name == "run_flow":
            result = await ctx.run_flow(arguments.get("input_value", ""))
        elif tool_name == "save_flow":
            result = await ctx.save_flow()
        else:
            return f"Unknown tool: {tool_name}"

        content_list = result.get("content", [])
        texts = [item.get("text", "") for item in content_list if isinstance(item, dict) and item.get("type") == "text"]
        return "\n".join(texts)
    except Exception as e:  # noqa: BLE001
        logger.exception("[Assistant] tool execution error: %s", e)
        return f"Error executing {tool_name}: {e}"


# ── Agentic loop (hybrid: native → prompt fallback) ────────────────────

_TOOL_ERROR_PATTERNS = re.compile(r"tool|function|type.*does not match", re.IGNORECASE)


def _is_anthropic_model(model_name: str) -> bool:
    """Check if the model name indicates an Anthropic/Claude model."""
    name_lower = model_name.lower()
    return "claude" in name_lower


async def _run_agentic_loop(
    *,
    api_key: str,
    api_url: str,
    model_name: str,
    user_message: str,
    session_id: str | None,
    ctx: _ToolContext,
    max_turns: int = 25,
) -> AsyncGenerator[tuple[str, dict], None]:
    """Hybrid agentic loop: native tool calling with prompt-based fallback."""

    # Resolve or create session
    sid = session_id or str(uuid.uuid4())
    resolved_sid = sid

    # Restore or create message history
    messages = _get_session_messages(sid)
    if not messages:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_message})

    # Determine mode: skip native for Anthropic models (they don't support OpenAI tool format)
    use_native = not _is_anthropic_model(model_name)
    if use_native:
        logger.warning("[Assistant] starting in native tool calling mode")
    else:
        logger.warning("[Assistant] Anthropic model detected, starting in prompt mode")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        for _turn in range(max_turns):
            text_content = ""
            tool_calls_accum: dict[int, dict] = {}  # native mode only
            finish_reason = None

            # Build payload
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "session_id": resolved_sid,
                "metadata": {"session_id": resolved_sid},
            }
            if use_native:
                payload["tools"] = TOOL_DEFINITIONS

            try:
                async with client.stream("POST", api_url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        error_text = error_body.decode("utf-8", errors="replace")

                        # Check if this is a tool-related 400 → fallback to prompt mode
                        if resp.status_code == 400 and use_native and _TOOL_ERROR_PATTERNS.search(error_text):
                            logger.warning("[Assistant] native tool calling not supported, switching to prompt mode")
                            use_native = False
                            continue  # retry this turn in prompt mode

                        logger.error("[Assistant] API error %d: %s", resp.status_code, error_text[:500])
                        yield ("error", {"error": f"API error {resp.status_code}: {error_text[:300]}"})
                        _save_session_messages(sid, messages)
                        return

                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue
                        delta = choices[0].get("delta", {})
                        finish_reason = choices[0].get("finish_reason") or finish_reason

                        # Stream text
                        content = delta.get("content")
                        if content:
                            text_content += content
                            yield ("token", {"text": content})

                        # Accumulate native tool calls
                        if use_native:
                            tc_list = delta.get("tool_calls")
                            if tc_list:
                                for tc in tc_list:
                                    idx = tc.get("index", 0)
                                    if idx not in tool_calls_accum:
                                        tool_calls_accum[idx] = {"id": "", "name": "", "arguments_str": ""}
                                    entry = tool_calls_accum[idx]
                                    if tc.get("id"):
                                        entry["id"] = tc["id"]
                                    func = tc.get("function", {})
                                    if func.get("name"):
                                        entry["name"] = func["name"]
                                    if func.get("arguments"):
                                        entry["arguments_str"] += func["arguments"]

            except httpx.HTTPError as e:
                logger.error("[Assistant] HTTP error: %s", e)
                yield ("error", {"error": f"HTTP error: {e}"})
                _save_session_messages(sid, messages)
                return

            # Parse tool calls: native or prompt-based
            if use_native and tool_calls_accum:
                tool_calls = []
                for idx in sorted(tool_calls_accum):
                    entry = tool_calls_accum[idx]
                    try:
                        args = json.loads(entry["arguments_str"]) if entry["arguments_str"] else {}
                    except json.JSONDecodeError:
                        args = {}
                    tool_calls.append((entry["name"], args, entry["id"]))
            elif not use_native:
                parsed = _parse_tool_calls_from_text(text_content)
                tool_calls = [(name, args, None) for name, args, _, _ in parsed]
                if not parsed:
                    logger.warning("[Assistant] prompt mode: no tool calls found in response (len=%d)", len(text_content))
                    logger.warning("[Assistant] prompt mode full response: %s", text_content[:2000])
            else:
                tool_calls = []

            # Append assistant message
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if text_content:
                assistant_msg["content"] = text_content
            else:
                assistant_msg["content"] = None

            if not tool_calls:
                messages.append(assistant_msg)
                break

            # Add tool_calls to assistant message for native mode
            if use_native:
                assistant_msg["tool_calls"] = [
                    {"id": tc_id, "type": "function", "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
                    for name, args, tc_id in tool_calls
                ]
            messages.append(assistant_msg)

            # Execute tools and emit events
            tool_results_parts: list[str] = []
            for tool_name, args, tc_id in tool_calls:
                yield ("tool_start", {"name": tool_name, "input": args})

                result_text = await _execute_tool(ctx, tool_name, args)
                logger.warning("[Assistant] tool %s result: %s", tool_name, result_text[:200])

                tool_result_data: dict = {"name": tool_name, "result": result_text[:500]}
                if ctx.last_canvas_data is not None:
                    tool_result_data["canvas_updated"] = True
                yield ("tool_result", tool_result_data)

                if ctx.last_canvas_data is not None:
                    logger.warning(
                        "[Assistant] canvas_update: %d nodes, %d edges",
                        len(ctx.last_canvas_data.get("nodes", [])),
                        len(ctx.last_canvas_data.get("edges", [])),
                    )
                    yield ("canvas_update", ctx.last_canvas_data)
                    ctx.last_canvas_data = None

                # Add to message history in appropriate format
                if use_native and tc_id:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_text})
                else:
                    tool_results_parts.append(
                        f"<tool_result_name>{tool_name}</tool_result_name>\n"
                        f"<tool_result>{result_text}</tool_result>"
                    )

            # For prompt mode, add combined tool results as user message
            if not use_native and tool_results_parts:
                messages.append({
                    "role": "user",
                    "content": "Here are the tool results:\n\n" + "\n\n".join(tool_results_parts),
                })
        else:
            yield ("token", {"text": "\n\n[Max turns reached]"})

    # Save session and emit session_id
    _save_session_messages(sid, messages)
    yield ("session_id", {"session_id": sid})


# ── SSE streaming endpoint ─────────────────────────────────────────────


@router.post("/assistant/chat")
async def assistant_chat(
    request: Request,
    body: AssistantChatRequest,
    current_user: CurrentActiveUser,
):
    """SSE endpoint for AI assistant chat."""
    user_id = str(current_user.id) if current_user else None

    try:
        api_key, api_url, model_name = _resolve_provider_credentials(body.selected_model, user_id)
    except ValueError as e:
        return StreamingResponse(
            iter([_sse_event("error", {"error": str(e)})]),
            media_type="text/event-stream",
        )

    async def event_generator() -> AsyncGenerator[str, None]:
        ctx = _ToolContext(flow_id=body.flow_id, user_id=user_id)
        yield _sse_event("connected", {"status": "ok"})

        try:
            async for event_name, data in _run_agentic_loop(
                api_key=api_key,
                api_url=api_url,
                model_name=model_name,
                user_message=body.message,
                session_id=body.session_id,
                ctx=ctx,
            ):
                if await request.is_disconnected():
                    break
                yield _sse_event(event_name, data)
        except asyncio.CancelledError:
            pass
        except Exception as e:  # noqa: BLE001
            logger.exception("Assistant chat error: %s", e)
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
