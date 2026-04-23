"""AI Assistant endpoint using direct HTTP calls to LLM providers.

Supports any OpenAI-compatible /v1/chat/completions endpoint.
Uses prompt-based tool calling for universal provider compatibility.
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


# ── System prompt with tool instructions ───────────────────────────────

SYSTEM_PROMPT = """\
You are an AI assistant that helps users build workflows in Langflow.
You have access to the following tools:

1. **list_components** — List all available components in the system.
   Arguments: {"category": "optional category filter"}
   Returns: component type, display name, description, category, and outputs.

2. **build_flow** — Build or update a workflow on the canvas.
   Arguments: {"nodes": [{id, type, values?}], "edges": [{source, source_output, target, target_input}]}
   The 'type' field must be a component type from list_components. This replaces the entire canvas.

3. **run_flow** — Run the current workflow on the canvas and return results.
   Arguments: {"input_value": "optional input text"}

IMPORTANT RULES:
- Always use list_components first to get exact type names and output names.
- The 'type' field in build_flow MUST match the exact 'type' value from list_components (e.g. 'ChatInput', 'ChatOutput', 'LanguageModelComponent'). Do NOT use provider names like 'OpenAI' or 'Anthropic' as component types.
- The 'source_output' MUST match the exact output name from list_components (e.g. 'message', 'text_output').
- The 'target_input' MUST match the field name in the target component's template (e.g. 'input_value').
- Edges are REQUIRED for the workflow to work. Always include edges.

When you need to call a tool, output EXACTLY this format (one tool per block):
<tool_call_name>tool_name_here</tool_call_name>
<tool_call_args>{"arg1": "value1"}</tool_call_args>

You may call multiple tools in one response. You may also include explanatory text before or between tool calls.
After receiving tool results, continue the conversation naturally.

Always explain what you're doing before taking actions.
Respond in the same language as the user's message.
"""

# Regex to extract tool calls from model output
_TOOL_NAME_RE = re.compile(r"<tool_call_name>(.*?)</tool_call_name>", re.DOTALL)
_TOOL_ARGS_RE = re.compile(r"<tool_call_args>(.*?)</tool_call_args>", re.DOTALL)


def _parse_tool_calls(text: str) -> list[tuple[str, dict, int, int]]:
    """Parse tool calls from model text output.

    Returns list of (tool_name, arguments, start_pos, end_pos) tuples.
    """
    results = []
    for name_match in _TOOL_NAME_RE.finditer(text):
        name = name_match.group(1).strip()
        name_start = name_match.start()
        name_end = name_match.end()

        # Find the corresponding args block after the name
        args_match = _TOOL_ARGS_RE.search(text, name_end)
        if not args_match or args_match.start() > name_end + 200:
            # No matching args found nearby, skip
            continue

        args_str = args_match.group(1).strip()
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            args = {}

        results.append((name, args, name_start, args_match.end()))
    return results


# ── Tool context ───────────────────────────────────────────────────────


class _ToolContext:
    """Holds per-request state shared across tool invocations."""

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
        result_text = f"Canvas updated: {node_count} nodes, {edge_count} edges."
        logger.warning("[Assistant] expanded: %d nodes, %d edges", node_count, edge_count)

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


# ── Provider credential resolution ─────────────────────────────────────


def _resolve_provider_credentials(
    selected_model: dict | None,
    user_id: str | None,
) -> tuple[str, str, str]:
    """Resolve (api_key, api_url, model_name) from Model Providers or env vars."""
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
            logger.warning(
                "[Assistant] resolved api_key=%s base_url=%s",
                bool(api_key),
                base_url or "N/A",
            )
        except Exception:  # noqa: BLE001
            logger.warning("Failed to resolve credentials from Model Providers, using env vars", exc_info=True)

    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY", "") or os.environ.get("ANTHROPIC_API_KEY", "")
    if not base_url:
        base_url = os.environ.get("OPENAI_BASE_URL", "") or os.environ.get("OPENAI_API_BASE", "")

    if not api_key:
        msg = (
            "No API key configured. "
            "Please select a model from Model Providers "
            "or set the OPENAI_API_KEY / ANTHROPIC_API_KEY environment variable."
        )
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
    """Execute a tool call and return the result text."""
    try:
        if tool_name == "list_components":
            result = await ctx.list_components(arguments.get("category", ""))
        elif tool_name == "build_flow":
            result = await ctx.build_flow(arguments.get("nodes", []), arguments.get("edges", []))
        elif tool_name == "run_flow":
            result = await ctx.run_flow(arguments.get("input_value", ""))
        else:
            return f"Unknown tool: {tool_name}"

        content_list = result.get("content", [])
        texts = []
        for item in content_list:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        return "\n".join(texts)
    except Exception as e:  # noqa: BLE001
        logger.exception("[Assistant] tool execution error: %s", e)
        return f"Error executing {tool_name}: {e}"


# ── Agentic loop (prompt-based tool calling) ───────────────────────────


async def _run_agentic_loop(
    *,
    api_key: str,
    api_url: str,
    model_name: str,
    user_message: str,
    session_id: str | None,
    ctx: _ToolContext,
    max_turns: int = 10,
) -> AsyncGenerator[tuple[str, dict], None]:
    """Run the agentic loop, yielding (event_name, data) tuples.

    Uses prompt-based tool calling: the model outputs tool calls as XML tags
    in its text response, which we parse and execute.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": user_message})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resolved_sid = session_id or str(uuid.uuid4())

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        for _turn in range(max_turns):
            payload: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "stream": True,
                "session_id": resolved_sid,
                "metadata": {"session_id": resolved_sid},
            }

            text_content = ""

            try:
                async with client.stream("POST", api_url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_body = await resp.aread()
                        error_text = error_body.decode("utf-8", errors="replace")
                        logger.error("[Assistant] API error %d: %s", resp.status_code, error_text[:500])
                        yield ("error", {"error": f"API error {resp.status_code}: {error_text[:300]}"})
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
                        content = delta.get("content")
                        if content:
                            text_content += content
                            yield ("token", {"text": content})

            except httpx.HTTPError as e:
                logger.error("[Assistant] HTTP error: %s", e)
                yield ("error", {"error": f"HTTP error: {e}"})
                return

            # Parse tool calls from the text response
            tool_calls = _parse_tool_calls(text_content)

            if not tool_calls:
                # No tool calls — done
                messages.append({"role": "assistant", "content": text_content})
                break

            # Add assistant message to history
            messages.append({"role": "assistant", "content": text_content})

            # Execute each tool call and build result message
            tool_results_parts: list[str] = []
            for tool_name, args, _start, _end in tool_calls:
                yield ("tool_start", {"name": tool_name, "input": args})

                result_text = await _execute_tool(ctx, tool_name, args)
                logger.warning("[Assistant] tool %s result: %s", tool_name, result_text[:200])

                tool_result_data: dict = {
                    "name": tool_name,
                    "result": result_text[:500],
                }
                if ctx.last_canvas_data is not None:
                    tool_result_data["canvas_updated"] = True

                yield ("tool_result", tool_result_data)

                if ctx.last_canvas_data is not None:
                    logger.warning(
                        "[Assistant] sending canvas_update: %d nodes, %d edges",
                        len(ctx.last_canvas_data.get("nodes", [])),
                        len(ctx.last_canvas_data.get("edges", [])),
                    )
                    yield ("canvas_update", ctx.last_canvas_data)
                    ctx.last_canvas_data = None

                tool_results_parts.append(
                    f"<tool_result_name>{tool_name}</tool_result_name>\n"
                    f"<tool_result>{result_text}</tool_result>"
                )

            # Add tool results as a user message for the next turn
            messages.append({
                "role": "user",
                "content": "Here are the tool results:\n\n" + "\n\n".join(tool_results_parts),
            })
        else:
            yield ("token", {"text": "\n\n[Max turns reached]"})


# ── SSE streaming endpoint ─────────────────────────────────────────────


@router.post("/assistant/chat")
async def assistant_chat(
    request: Request,
    body: AssistantChatRequest,
    current_user: CurrentActiveUser,
):
    """SSE endpoint for AI assistant chat using direct HTTP calls."""
    user_id = str(current_user.id) if current_user else None

    try:
        api_key, api_url, model_name = _resolve_provider_credentials(
            body.selected_model, user_id,
        )
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
