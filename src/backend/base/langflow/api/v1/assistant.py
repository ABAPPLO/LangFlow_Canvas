"""AI Assistant endpoint using direct HTTP calls to LLM providers.

Hybrid approach: tries OpenAI native tool calling first, falls back to prompt-based.
Includes session persistence for multi-turn conversations.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
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

# Categories to exclude from list_components (third-party bundles shown in UI "Bundles" section)
_EXCLUDED_CATEGORIES: set[str] = {
    "aiml", "agentics", "agentql", "altk", "languagemodels", "embeddings",
    "memories", "amazon", "anthropic", "apify", "arxiv", "assemblyai", "AssemblyAI",
    "azure", "baidu", "bing", "cassandra", "chroma", "clickhouse",
    "cleanlab", "cloudflare", "cohere", "cometapi", "composio", "confluence",
    "couchbase", "crewai", "cuga", "datastax", "deepseek", "docling",
    "duckduckgo", "elastic", "exa", "FAISS", "firecrawl", "git",
    "glean", "gmail", "google", "groq", "homeassistant", "huggingface",
    "ibm", "icosacomputing", "jigsawstack", "langchain_utilities", "langwatch", "LangWatch",
    "litellm", "lmstudio", "maritalk", "mem0", "milvus", "mistral",
    "mongodb", "needle", "notdiamond", "Notion", "notion", "novita", "nvidia",
    "olivya", "ollama", "openai", "openrouter", "perplexity", "pgvector",
    "pinecone", "qdrant", "redis", "sambanova", "scrapegraph", "searchapi",
    "serpapi", "serper", "supabase", "tavily", "twelvelabs", "unstructured",
    "upstash", "vlmrun", "vectara", "vectorstores", "vllm", "weaviate",
    "vertexai", "wikipedia", "wolframalpha", "xai", "yahoosearch", "youtube", "YouTube",
    "zep",
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
                "input_types, has_options, available_options) and outputs (name, types). "
                "ALWAYS call before build_flow to check available model options."
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
                "Build workflow on canvas. Auto-deduplicates by component type — if a type already "
                "exists on canvas, it will be updated instead of duplicated. "
                "Set clear=true to wipe the canvas before building (for starting over). "
                "Nodes: [{id, type, values?}]. Edges: [{source, source_output, target, target_input}]. "
                "The 'type' must be from list_components. "
                "'values' MUST set key parameters — use get_component_details to check available_options first. "
                "For model fields use {\"model\": [{\"name\": \"dall-e-3\"}]}, for dropdowns use the exact option string. "
                "Edges connect the specified nodes. Existing edges are preserved unless clear=true."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "clear": {
                        "type": "boolean",
                        "description": "If true, clear existing canvas before building. Use when starting a new workflow.",
                    },
                    "nodes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": {"type": "string"},
                                "values": {
                                    "type": "object",
                                    "description": "Component parameters. Check get_component_details for available_options.",
                                },
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
            "description": (
                "Run the current workflow and return results. Auto-saves on success. "
                "On failure, returns diagnosed error with specific fix suggestions."
            ),
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
            "name": "get_current_flow",
            "description": (
                "Get the current canvas state. Returns all nodes (with type, values) and edges "
                "(with source/target connections). Use to inspect before modifying or to check what's on canvas."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_flow",
            "description": (
                "Pre-flight check and auto-fix before run_flow. Validates the flow AND automatically fixes "
                "common issues: wrong model types, empty required fields, missing defaults. "
                "Returns what was fixed and what still needs manual attention. "
                "Call this when ready to run."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_node",
            "description": (
                "Update parameters of a single node on the canvas without rebuilding the entire flow. "
                "Use this to change model, prompt, or any field on an existing node. "
                "For model fields use {\"model\": [{\"name\": \"gpt-4o\"}]}, for dropdowns use the exact option string."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID to update",
                    },
                    "values": {
                        "type": "object",
                        "description": "Parameters to update. Only specified fields are changed, others are preserved.",
                    },
                },
                "required": ["node_id", "values"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_nodes",
            "description": (
                "Delete one or more nodes from the canvas. All edges connected to deleted nodes "
                "are also removed. Use to clean up unwanted nodes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of node IDs to delete",
                    },
                },
                "required": ["node_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_edge",
            "description": (
                "Add a connection between two nodes. Both nodes must exist on the canvas."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source node ID"},
                    "source_output": {"type": "string", "description": "Output field name on source node"},
                    "target": {"type": "string", "description": "Target node ID"},
                    "target_input": {"type": "string", "description": "Input field name on target node"},
                },
                "required": ["source", "source_output", "target", "target_input"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_edge",
            "description": (
                "Delete a connection between two nodes."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Source node ID"},
                    "target": {"type": "string", "description": "Target node ID"},
                },
                "required": ["source", "target"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_component",
            "description": (
                "Run a single component on the canvas and return its output. "
                "Useful for debugging individual components without running the full flow."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {
                        "type": "string",
                        "description": "The node ID to run",
                    },
                },
                "required": ["node_id"],
            },
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


def _build_component_catalog(all_types: dict[str, Any], excluded: set[str] | None = None) -> list[dict]:
    catalog = []
    for _category, components in all_types.items():
        if not isinstance(components, dict):
            continue
        if excluded and _category in excluded:
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
        if field_data.get("_input_type"):
            field_info["_input_type"] = field_data["_input_type"]
        if field_data.get("required"):
            field_info["required"] = True
        if field_data.get("value") is not None:
            field_info["default"] = field_data["value"]
        if field_data.get("input_types"):
            field_info["input_types"] = field_data["input_types"]
        if field_data.get("options"):
            field_info["options"] = field_data["options"]
            # Add readable options summary for the AI
            raw_options = field_data["options"]
            input_type = field_data.get("_input_type", "")
            if input_type == "ModelInput" and isinstance(raw_options, list):
                model_names = []
                for item in raw_options[:15]:
                    if isinstance(item, dict):
                        name = item.get("name", "")
                        provider = item.get("provider", "")
                        model_names.append(f"{name} ({provider})" if provider else name)
                    elif isinstance(item, str):
                        model_names.append(item)
                field_info["has_options"] = True
                field_info["available_options"] = model_names
            elif isinstance(raw_options, list) and raw_options:
                field_info["has_options"] = True
                field_info["available_options"] = [str(o) for o in raw_options[:20]]
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


async def _enrich_model_options(details: dict, user_id: str | None) -> dict:
    """Enrich ModelInput fields with live model options from provider configuration."""
    has_model_field = any(
        f.get("_input_type") == "ModelInput" and not f.get("has_options")
        for f in details.get("fields", [])
    )
    if not has_model_field:
        return details

    try:
        from lfx.base.models.unified_models import get_language_model_options

        options = get_language_model_options(user_id=user_id)
        if not options:
            return details

        # Build compact model list for AI
        model_names = []
        for item in options[:20]:
            if isinstance(item, dict):
                name = item.get("name", "")
                provider = item.get("provider", "")
                model_names.append(f"{name} ({provider})" if provider else name)

        for field in details.get("fields", []):
            if field.get("_input_type") == "ModelInput" and not field.get("has_options"):
                field["has_options"] = True
                field["available_options"] = model_names
    except Exception:  # noqa: BLE001
        logger.debug("[Assistant] Failed to load live model options for details enrichment")

    return details


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

2. **get_component_details** — Get fields, options, and outputs for a component.
   Args: {"type": "component_type_name"}
   ALWAYS call this before build_flow to check available_options for each field.

3. **build_flow** — Build workflow on canvas.
   Args: {"nodes": [{id, type, values?}], "edges": [{source, source_output, target, target_input}]}
   - "type" MUST be from list_components (e.g. "ChatInput", NOT "OpenAI").
   - "source_output" MUST match output name (e.g. "message").
   - "target_input" MUST match field name (e.g. "input_value").
   - "values" MUST set key parameters (especially model fields). Check get_component_details for options.
   - Edges are REQUIRED.

4. **get_current_flow** — Get current canvas state (nodes, edges, values).
   Args: {}

5. **prepare_flow** — Validate + auto-fix before run_flow.
   Args: {}
   Validates AND auto-fixes: wrong models, empty required fields, missing defaults.
   Call this when ready to run.

6. **update_node** — Update a single node's parameters.
   Args: {"node_id": "id", "values": {"field": "value"}}
   Use to fix specific fields without rebuilding the entire flow.

7. **delete_nodes** — Remove nodes and their connected edges.
   Args: {"node_ids": ["id1", "id2"]}

8. **add_edge** — Add a connection between two nodes.
   Args: {"source": "id", "source_output": "name", "target": "id", "target_input": "name"}

9. **delete_edge** — Remove a connection between two nodes.
   Args: {"source": "id", "target": "id"}

10. **run_flow** — Execute the workflow. Auto-saves on success, includes diagnosis on failure.
    Args: {"input_value": "optional test input"}

11. **run_component** — Run a single component for debugging.
    Args: {"node_id": "id"}

WORKFLOW (follow strictly):
  Step 1: list_components → get_component_details (check options for EACH component)
  Step 2: build_flow (set correct values from available_options)
  Step 3: prepare_flow (auto-fix issues)
          → if errors remain: use update_node / add_edge / delete_edge to fix, then retry prepare_flow
  Step 4: run_flow (auto-saves on success)
          → if fails: error response includes diagnosis → use update_node to fix → retry run_flow
  Step 5: Report success to user

PARAMETER GUIDELINES:
  - CRITICAL: Always set "values" in build_flow, especially for model fields.
  - Call get_component_details first. If a field has "has_options": true, pick from "available_options".
  - Match model to task:
    * Image generation → dall-e-3 (NOT gpt-4o or chat models)
    * Chat / Text → gpt-4o, claude-sonnet-4-20250514, etc.
    * Embeddings → text-embedding-3-small
  - ModelInput value format: {"model": [{"name": "dall-e-3"}]}
  - DropdownInput value format: {"model_name": "gpt-4o"}

CRITICAL SUCCESS CRITERIA:
  - build_flow succeeding does NOT mean the task is done.
  - You MUST run run_flow to verify the workflow actually works.
  - If run_flow fails: read the diagnosis in the error response, fix with update_node, then retry run_flow.
  - Max 5 total attempts. If still failing after 5 tries, report the error to the user.

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
        self.last_compact_edges: list[dict] | None = None

    async def ensure_types(self) -> dict[str, Any]:
        if self.all_types is None:
            self.all_types = await _get_all_types_dict()
        return self.all_types

    async def list_components(self, category: str = "") -> dict:
        await self.ensure_types()
        if self.catalog is None:
            self.catalog = _build_component_catalog(self.all_types, _EXCLUDED_CATEGORIES)  # type: ignore[arg-type]
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
                # Enrich ModelInput fields with live model options
                details = await _enrich_model_options(details, self.user_id)
                return {"content": [{"type": "text", "text": json.dumps(details, ensure_ascii=False)}]}
        return {"content": [{"type": "text", "text": f"Component type '{type_name}' not found."}]}

    async def build_flow(self, nodes: list[dict], edges: list[dict], clear: bool = False) -> dict:
        await self.ensure_types()
        logger.warning("[Assistant] build_flow called: nodes=%d edges=%d clear=%s", len(nodes), len(edges), clear)

        # Expand new nodes
        compact_data = {"nodes": nodes, "edges": []}  # edges handled via compact_edges
        try:
            expanded_new = expand_compact_flow(compact_data, self.all_types)  # type: ignore[arg-type]
        except ValueError as e:
            logger.warning("[Assistant] expand_compact_flow error: %s", e)
            return {"content": [{"type": "text", "text": f"Error building flow: {e}"}]}

        new_nodes = expanded_new.get("nodes", [])
        new_node_map = {n["id"]: n for n in new_nodes}

        # Build a type -> existing_node map for deduplication
        new_type_map: dict[str, dict] = {}
        for n in new_nodes:
            ntype = n.get("data", {}).get("type", "")
            if ntype:
                new_type_map[ntype] = n

        # Load existing canvas
        existing_nodes, existing_edges = await self._load_current_canvas()

        if clear:
            # Clear canvas — only keep new nodes
            merged_nodes = list(new_nodes)
            deduped = len(existing_nodes)
        else:
            # Deduplicate by component type: if new node has same type as existing,
            # remap to use existing node's ID and update in-place
            id_remap: dict[str, str] = {}  # new_id -> existing_id
            existing_types: dict[str, dict] = {}
            for existing in existing_nodes:
                etype = existing.get("data", {}).get("type", "")
                if etype:
                    existing_types[etype] = existing

            for n in new_nodes:
                ntype = n.get("data", {}).get("type", "")
                if ntype and ntype in existing_types and n["id"] != existing_types[ntype]["id"]:
                    old_id = n["id"]
                    existing_id = existing_types[ntype]["id"]
                    id_remap[old_id] = existing_id
                    logger.warning("[Assistant] dedup: %s ('%s') -> reuse existing '%s'",
                                   old_id, ntype, existing_id)

            # Remap node IDs in expanded nodes
            for n in new_nodes:
                old_id = n["id"]
                if old_id in id_remap:
                    n["id"] = id_remap[old_id]

            # Remap edge references
            if edges:
                edges = [
                    {
                        **e,
                        "source": id_remap.get(e.get("source", ""), e.get("source", "")),
                        "target": id_remap.get(e.get("target", ""), e.get("target", "")),
                    }
                    for e in edges
                ]

            # Rebuild new_node_map after remapping
            new_node_map = {n["id"]: n for n in new_nodes}

            # Merge: update existing or add new nodes
            merged_nodes = []
            for existing in existing_nodes:
                if existing["id"] in new_node_map:
                    merged_nodes.append(new_node_map.pop(existing["id"]))
                else:
                    merged_nodes.append(existing)
            merged_nodes.extend(new_node_map.values())
            deduped = 0

        # Expand edges against merged nodes
        if edges:
            expanded_new_edges = self._expand_edges(edges, merged_nodes)
            if clear:
                expanded_edges = expanded_new_edges
            else:
                # Build set of (target_node, target_input_name) from new edges
                # to detect conflicts — a target input should only have one source
                new_target_inputs: set[tuple[str, str]] = set()
                for ne in expanded_new_edges:
                    ne_data = ne.get("data", {})
                    th = ne_data.get("targetHandle", {})
                    if isinstance(th, dict):
                        new_target_inputs.add((ne.get("target", ""), th.get("id", "")))

                # Remove old edges that conflict with new target inputs
                if new_target_inputs:
                    kept = []
                    for oe in existing_edges:
                        oe_data = oe.get("data", {})
                        oth = oe_data.get("targetHandle", {})
                        if isinstance(oth, dict):
                            okey = (oe.get("target", ""), oth.get("id", ""))
                            if okey in new_target_inputs:
                                continue  # Old edge conflicts, remove it
                        kept.append(oe)
                    existing_edges = kept

                # Merge with dedup by (source, sourceHandle, target, targetHandle)
                existing_keys = {
                    (e.get("source"), e.get("sourceHandle"), e.get("target"), e.get("targetHandle"))
                    for e in existing_edges
                }
                for ne in expanded_new_edges:
                    key = (ne.get("source"), ne.get("sourceHandle"), ne.get("target"), ne.get("targetHandle"))
                    if key not in existing_keys:
                        existing_edges.append(ne)
                        existing_keys.add(key)
                expanded_edges = existing_edges
        elif clear:
            expanded_edges = []
        else:
            expanded_edges = existing_edges

        node_count = len(merged_nodes)
        edge_count = len(expanded_edges)
        logger.warning("[Assistant] merged canvas: %d nodes, %d edges (cleared=%s, deduped=%d)",
                        node_count, edge_count, clear, deduped)

        # Assign layout positions based on edge topology
        self._assign_layout(merged_nodes, edges)

        self.last_canvas_data = {"nodes": merged_nodes, "edges": expanded_edges}
        self.last_compact_edges = edges

        # Persist to DB immediately
        await self._persist_canvas()

        status = f"Canvas cleared and rebuilt: {node_count} nodes, {edge_count} edges." if clear \
            else f"Canvas updated: {node_count} nodes, {edge_count} edges."
        return {"content": [{"type": "text", "text": status}]}

    async def _load_current_canvas(self) -> tuple[list[dict], list[dict]]:
        """Load existing canvas nodes and edges from DB or last_canvas_data."""
        if self.last_canvas_data:
            return self.last_canvas_data.get("nodes", []), self.last_canvas_data.get("edges", [])
        try:
            flow = await get_flow_by_id_or_endpoint_name(
                flow_id_or_name=self.flow_id, user_id=self.user_id,
            )
            if flow.data:
                return flow.data.get("nodes", []), flow.data.get("edges", [])
        except (ValueError, RuntimeError):
            pass
        return [], []

    def _expand_edges(self, compact_edges: list[dict], merged_nodes: list[dict]) -> list[dict]:
        """Expand compact edges using merged node data."""
        from langflow.processing.expand_flow import (
            _build_source_handle_data,
            _build_target_handle_data,
            _encode_handle,
        )

        # Build expanded node lookup
        expanded_map: dict[str, dict] = {}
        for node in merged_nodes:
            expanded_map[node["id"]] = node

        result_edges = []
        for ce in compact_edges:
            source_node = expanded_map.get(ce.get("source", ""))
            target_node = expanded_map.get(ce.get("target", ""))
            if not source_node or not target_node:
                logger.warning("[Assistant] edge references unknown node: %s", ce)
                continue

            source_data = source_node.get("data", {})
            target_data = target_node.get("data", {})
            source_node_data = source_data.get("node", {})
            target_node_data = target_data.get("node", {})

            # Source handle
            source_type = source_data.get("type", "")
            source_outputs = source_node_data.get("outputs", [])
            source_output_name = ce.get("source_output", "")
            source_output = next((o for o in source_outputs if isinstance(o, dict) and o.get("name") == source_output_name), None)
            output_types = [source_output.get("selected") or (source_output.get("types") or [""])[0]] if source_output else []
            output_name = source_output.get("name", source_output_name) if source_output else source_output_name

            source_handle_data = _build_source_handle_data(
                ce["source"], source_type, output_name, output_types,
            )

            # Target handle
            target_template = target_node_data.get("template", {})
            target_input = ce.get("target_input", "")
            target_field = target_template.get(target_input, {})
            if isinstance(target_field, dict):
                input_types = target_field.get("input_types", [])
                field_type = target_field.get("type", "str")
            else:
                input_types, field_type = [], "str"

            target_handle_data = _build_target_handle_data(
                ce["target"], target_input, input_types or [field_type], field_type,
            )

            source_handle_str = _encode_handle(source_handle_data)
            target_handle_str = _encode_handle(target_handle_data)
            edge_id = f"reactflow__edge-{ce['source']}{source_handle_str}-{ce['target']}{target_handle_str}"

            result_edges.append({
                "source": ce["source"],
                "sourceHandle": source_handle_str,
                "target": ce["target"],
                "targetHandle": target_handle_str,
                "id": edge_id,
                "data": {"sourceHandle": source_handle_data, "targetHandle": target_handle_data},
                "className": "",
                "selected": False,
                "animated": False,
            })

        return result_edges

    def _assign_layout(self, nodes: list[dict], compact_edges: list[dict]) -> None:
        """Assign layout positions to all nodes based on edge topology.

        Uses topological sort to place nodes left-to-right with vertical spacing
        for nodes in the same layer. Always re-layouts all nodes for consistency.
        """
        from collections import deque

        spacing_x = 500
        spacing_y = 350
        start_x = 100
        start_y = 100

        node_ids = {n.get("id") for n in nodes}
        in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
        children: dict[str, list[str]] = {nid: [] for nid in node_ids}

        for ce in compact_edges:
            src = ce.get("source", "")
            tgt = ce.get("target", "")
            if src in node_ids and tgt in node_ids:
                in_degree[tgt] = in_degree.get(tgt, 0) + 1
                children.setdefault(src, []).append(tgt)

        queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
        layers: list[list[str]] = []

        while queue:
            layer = []
            for _ in range(len(queue)):
                nid = queue.popleft()
                layer.append(nid)
                for child in children.get(nid, []):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        queue.append(child)
            if layer:
                layers.append(layer)

        placed = {nid for layer in layers for nid in layer}
        remaining = [nid for nid in node_ids if nid not in placed]
        if remaining:
            layers.append(remaining)

        node_map = {n.get("id"): n for n in nodes}
        for col_idx, layer in enumerate(layers):
            for row_idx, nid in enumerate(layer):
                node = node_map.get(nid)
                if node:
                    node["position"] = {
                        "x": start_x + col_idx * spacing_x,
                        "y": start_y + row_idx * spacing_y,
                    }

    async def run_flow(self, input_value: str = "") -> dict:
        # Auto-save canvas data before running
        await self._persist_canvas()

        # Load flow data
        try:
            flow = await get_flow_by_id_or_endpoint_name(
                flow_id_or_name=self.flow_id, user_id=self.user_id,
            )
        except (ValueError, RuntimeError) as e:
            return {"content": [{"type": "text", "text": f"Error loading flow: {e}"}]}

        if flow.data is None:
            return {"content": [{"type": "text", "text": "Flow has no data. Please build the flow first."}]}

        flow_data = flow.data
        nodes = flow_data.get("nodes", [])
        edges = flow_data.get("edges", [])

        if not nodes:
            return {"content": [{"type": "text", "text": "Canvas is empty. Build the flow first."}]}

        stop_component_id = self._find_output_component(nodes, edges)
        logger.warning("[Assistant] run_flow: stop_component_id=%s", stop_component_id)

        try:
            from langflow.api.utils.core import build_graph_from_data

            graph = await build_graph_from_data(
                flow_id=self.flow_id,
                payload=flow_data,
                user_id=self.user_id,
            )

            inputs_dict = {}
            if input_value:
                inputs_dict["input_value"] = input_value
                inputs_dict["client_request_time"] = str(int(time.time() * 1000))

            first_layer = graph.sort_vertices(stop_component_id=stop_component_id)
            vertices_to_run = list(graph.vertices_to_run)
            logger.warning("[Assistant] run_flow: %d vertices to run", len(vertices_to_run))

            if not vertices_to_run:
                return {"content": [{"type": "text", "text": "No vertices to run. The flow may have connection issues. Check edges with get_current_flow."}]}

            results: list[dict] = []
            all_layers = [first_layer, *graph.vertices_layers]
            for layer in all_layers:
                for vertex_id in layer:
                    vertex = graph.get_vertex(vertex_id)
                    if vertex is None:
                        continue

                    is_first_vertex = first_layer and vertex_id == first_layer[0]
                    if inputs_dict and is_first_vertex:
                        vertex.update_raw_params(inputs_dict)

                    build_result = await graph.build_vertex(
                        vertex_id,
                        inputs_dict=inputs_dict if vertex_id in first_layer else None,
                        user_id=self.user_id,
                        fallback_to_env_vars=True,
                    )
                    result_dict = build_result.result_dict
                    if result_dict:
                        results.append({
                            "component": vertex_id,
                            "result": str(result_dict.get("message", "")),
                            "valid": build_result.valid,
                        })

            logger.warning("[Assistant] run_flow completed: %d results", len(results))

            # Auto-save on success — already persisted before run, just confirm
            return {"content": [{"type": "text", "text": json.dumps(results, ensure_ascii=False)}]}

        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            logger.exception("[Assistant] run_flow error: %s", e)
            # Include diagnosis in the error response
            diagnosis = self._diagnose_error(error_msg, nodes)
            error_response = {
                "error": error_msg[:500],
                "diagnosis": diagnosis,
            }
            return {"content": [{"type": "text", "text": json.dumps(error_response, ensure_ascii=False)}]}

    def _diagnose_error(self, error_message: str, nodes: list[dict]) -> list[dict]:
        """Quick diagnosis of run_flow errors."""
        error_lower = error_message.lower()
        findings: list[dict[str, str]] = []

        # Credential errors
        if any(kw in error_lower for kw in ["api key", "api_key", "unauthorized", "401", "403", "authentication"]):
            findings.append({"fix": "API key missing or invalid. Configure it in Model Providers or check environment variables."})

        # Model errors
        if any(kw in error_lower for kw in ["model not found", "does not exist", "invalid model", "unknown model"]):
            findings.append({"fix": "Model name is incorrect. Use get_component_details to check available models, then fix with update_node."})

        # Network errors
        if any(kw in error_lower for kw in ["connection", "timeout", "refused", "network"]):
            findings.append({"fix": "Network error. Check base_url in Model Providers is correct and accessible."})

        # Type errors
        if any(kw in error_lower for kw in ["type error", "attributeerror", "unexpected type"]):
            findings.append({"fix": "Type mismatch between components. Check that connected outputs match input types."})

        if not findings:
            findings.append({"fix": "Check all component parameters with get_component_details. Use update_node to fix issues."})

        return findings

    def _find_output_component(self, nodes: list[dict], edges: list[dict]) -> str | None:
        """Find the last output component in the flow — the one with no outgoing edges."""
        if not nodes:
            return None

        # Prefer known output types
        _OUTPUT_TYPES = {"ChatOutput", "TextOutput", "ImageOutput", "AudioOutput"}

        # Build set of nodes that are sources of edges (have outgoing connections)
        source_ids = {e.get("source", "") for e in edges}

        # First: look for output-type nodes that are NOT sources (i.e. they are final)
        for node in nodes:
            node_type = node.get("data", {}).get("type", "")
            node_id = node.get("id", "")
            if node_type in _OUTPUT_TYPES and node_id not in source_ids:
                return node_id

        # Fallback: any node that is not a source (has no outgoing edges)
        for node in nodes:
            node_id = node.get("id", "")
            if node_id not in source_ids:
                return node_id

        # Last resort: use the last node
        return nodes[-1].get("id")

    async def get_current_flow(self) -> dict:
        """Return the current canvas state as compact nodes and edges."""
        try:
            flow = await get_flow_by_id_or_endpoint_name(
                flow_id_or_name=self.flow_id, user_id=self.user_id,
            )
        except (ValueError, RuntimeError) as e:
            return {"content": [{"type": "text", "text": f"Error loading flow: {e}"}]}

        flow_data = self.last_canvas_data or flow.data
        if not flow_data:
            return {"content": [{"type": "text", "text": "Canvas is empty. No flow data found."}]}

        await self.ensure_types()

        # Compact node representation
        compact_nodes = []
        for node in flow_data.get("nodes", []):
            node_data = node.get("data", {})
            node_type = node_data.get("type", "unknown")
            template = node_data.get("node", {}).get("template", {})
            values = {}
            for field_name, field_data in template.items():
                if not isinstance(field_data, dict) or field_name.startswith("_"):
                    continue
                val = field_data.get("value")
                if val is not None:
                    values[field_name] = val
            compact_nodes.append({
                "id": node.get("id", ""),
                "type": node_type,
                "values": values,
                "position": node.get("position"),
            })

        # Compact edge representation
        compact_edges = []
        for edge in flow_data.get("edges", []):
            edge_data = edge.get("data", {})
            sh = edge_data.get("sourceHandle", {})
            th = edge_data.get("targetHandle", {})
            compact_edges.append({
                "source": edge.get("source", ""),
                "source_output": sh.get("name", ""),
                "target": edge.get("target", ""),
                "target_input": th.get("fieldName", ""),
            })

        result = {
            "nodes": compact_nodes,
            "edges": compact_edges,
            "node_count": len(compact_nodes),
            "edge_count": len(compact_edges),
        }
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}

    async def prepare_flow(self) -> dict:
        """Pre-flight check and auto-fix before run_flow."""
        # Load canvas
        existing_nodes, existing_edges = await self._load_current_canvas()
        if not existing_nodes:
            return {"content": [{"type": "text", "text": "Canvas is empty. Build the flow first."}]}

        await self.ensure_types()
        issues: list[dict[str, str]] = []
        auto_fixed: list[dict[str, str]] = []
        _CHAT_MODEL_KEYWORDS = {"gpt-4o", "gpt-4", "gpt-3.5", "claude-sonnet", "claude-opus", "claude-haiku", "deepseek"}
        _TASK_RULES: dict[str, dict[str, Any]] = {
            "ImageGeneration": {
                "expected_keywords": ("dall-e", "image", "stable", "flux", "midjourney", "imagen", "seedream"),
                "fix_model": "dall-e-3",
                "message": "Image generation needs an image model, not a chat model.",
            },
            "VideoGeneration": {
                "expected_keywords": ("seedance", "video", "cogvideox", "kling", "sora", "wan"),
                "fix_model": "seedance-2.0",
                "message": "Video generation needs a video model, not a chat model.",
            },
        }

        # Pre-load live model options for ModelInput fields
        live_model_options: list[dict] = []
        try:
            from lfx.base.models.unified_models import get_language_model_options
            live_model_options = get_language_model_options(user_id=self.user_id)
        except Exception:  # noqa: BLE001
            logger.debug("[Assistant] prepare_flow: failed to load live model options")

        modified = False

        for node in existing_nodes:
            node_type = node.get("data", {}).get("type", "")
            node_id = node.get("id", "")
            node_data = node.get("data", {})
            node_node = node_data.get("node", {})
            node_template = node_node.get("template", {})

            comp = self._find_component(node_type)
            if not comp:
                issues.append({"severity": "error", "node_id": node_id, "issue": f"Component type '{node_type}' not found"})
                continue

            comp_template = comp.get("template", {})

            # --- Fix 1: Empty ModelInput → fill first available model ---
            for field_name, field_data in comp_template.items():
                if not isinstance(field_data, dict) or field_name.startswith("_"):
                    continue
                if field_data.get("_input_type") != "ModelInput":
                    continue

                node_field = node_template.get(field_name, {})
                if not isinstance(node_field, dict):
                    continue
                current_value = node_field.get("value")

                is_empty = (current_value is None or current_value == "" or
                            (isinstance(current_value, list) and not current_value))

                if is_empty:
                    # Prefer live model options over static component index
                    model_options = live_model_options or field_data.get("options", [])
                    if model_options and isinstance(model_options, list):
                        first_opt = model_options[0]
                        if isinstance(first_opt, dict) and first_opt.get("name"):
                            new_val = [{"name": first_opt["name"], "provider": first_opt.get("provider", "")}]
                            node_field["value"] = new_val
                            modified = True
                            auto_fixed.append({
                                "node_id": node_id,
                                "node_type": node_type,
                                "field": field_name,
                                "fix": f"Empty model → set to '{first_opt['name']}' ({first_opt.get('provider', '')})",
                            })

            # --- Fix 2: Model-task mismatch → swap to correct model ---
            rule = _TASK_RULES.get(node_type)
            if rule:
                for field_name, fd in node_template.items():
                    if not isinstance(fd, dict) or fd.get("_input_type") != "ModelInput":
                        continue
                    val = fd.get("value")
                    if not val:
                        continue
                    model_name = ""
                    if isinstance(val, list) and val:
                        model_name = val[0].get("name", "") if isinstance(val[0], dict) else str(val[0])
                    elif isinstance(val, str):
                        model_name = val
                    if not model_name:
                        continue
                    name_lower = model_name.lower()
                    is_chat = any(kw in name_lower for kw in _CHAT_MODEL_KEYWORDS)
                    has_expected = any(kw in name_lower for kw in rule["expected_keywords"])
                    if is_chat and not has_expected:
                        # Find a suitable model from live options (fallback to static)
                        model_opts = live_model_options
                        if not model_opts:
                            comp_field = comp_template.get(field_name, {})
                            model_opts = comp_field.get("options", []) if isinstance(comp_field, dict) else []
                        fix_model = None
                        for opt in model_opts:
                            if isinstance(opt, dict):
                                opt_name = opt.get("name", "")
                                if any(kw in opt_name.lower() for kw in rule["expected_keywords"]):
                                    fix_model = opt
                                    break
                        if fix_model:
                            fd["value"] = [{"name": fix_model["name"], "provider": fix_model.get("provider", "")}]
                            modified = True
                            auto_fixed.append({
                                "node_id": node_id,
                                "node_type": node_type,
                                "field": field_name,
                                "fix": f"Model '{model_name}' → '{fix_model['name']}' ({rule['message']})",
                            })
                        else:
                            issues.append({
                                "severity": "error",
                                "node_id": node_id,
                                "node_type": node_type,
                                "issue": f"Model '{model_name}' is wrong for {node_type}. {rule['message']} No suitable model found in options.",
                            })

            # --- Fix 3: Empty required fields with defaults → fill ---
            for field_name, field_data in comp_template.items():
                if not isinstance(field_data, dict) or field_name.startswith("_"):
                    continue
                if not field_data.get("required"):
                    continue

                node_field = node_template.get(field_name, {})
                if not isinstance(node_field, dict):
                    continue
                current_value = node_field.get("value")
                if current_value is not None and current_value != "" and current_value != []:
                    continue

                # Has options? Pick first non-empty
                raw_options = field_data.get("options", [])
                default_val = field_data.get("value")

                if raw_options and isinstance(raw_options, list):
                    first_opt = raw_options[0]
                    if isinstance(first_opt, str) and first_opt:
                        node_field["value"] = first_opt
                        modified = True
                        display_name = field_data.get("display_name", field_name)
                        auto_fixed.append({
                            "node_id": node_id,
                            "node_type": node_type,
                            "field": field_name,
                            "fix": f"Empty required field '{display_name}' → '{first_opt}'",
                        })
                        continue

                if default_val is not None and default_val != "" and default_val != []:
                    node_field["value"] = default_val
                    modified = True
                    display_name = field_data.get("display_name", field_name)
                    auto_fixed.append({
                        "node_id": node_id,
                        "node_type": node_type,
                        "field": field_name,
                        "fix": f"Empty required field '{display_name}' → default '{default_val}'",
                    })

            # --- Fix 4: DropdownInput value not in options ---
            for field_name, field_data in comp_template.items():
                if not isinstance(field_data, dict) or field_name.startswith("_"):
                    continue
                if field_data.get("_input_type") != "DropdownInput":
                    continue
                if field_data.get("combobox"):
                    continue

                raw_options = field_data.get("options", [])
                if not raw_options or not isinstance(raw_options, list):
                    continue

                node_field = node_template.get(field_name, {})
                if not isinstance(node_field, dict):
                    continue
                current_value = node_field.get("value")
                if not isinstance(current_value, str) or not current_value:
                    continue

                valid_strings = [o for o in raw_options if isinstance(o, str)]
                if valid_strings and current_value not in valid_strings:
                    node_field["value"] = valid_strings[0]
                    modified = True
                    display_name = field_data.get("display_name", field_name)
                    auto_fixed.append({
                        "node_id": node_id,
                        "node_type": node_type,
                        "field": field_name,
                        "fix": f"'{display_name}' value '{current_value}' not in options → '{valid_strings[0]}'",
                    })

        # --- Check: Disconnected nodes ---
        if len(existing_nodes) > 1:
            connected_ids: set[str] = set()
            for edge in existing_edges:
                connected_ids.add(edge.get("source", ""))
                connected_ids.add(edge.get("target", ""))
            for node in existing_nodes:
                nid = node.get("id", "")
                if nid and nid not in connected_ids:
                    issues.append({
                        "severity": "warning",
                        "node_id": nid,
                        "issue": f"Node '{nid}' ({node.get('data', {}).get('type', '')}) is not connected",
                    })

        # Save modified canvas
        if modified:
            self.last_canvas_data = {"nodes": existing_nodes, "edges": existing_edges}

        ready = not any(i["severity"] == "error" for i in issues)
        result = {
            "ready": ready,
            "auto_fixed": auto_fixed,
            "remaining_issues": issues,
            "fixed_count": len(auto_fixed),
            "issue_count": len(issues),
        }
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}

    def _find_component(self, type_name: str) -> dict | None:
        """Find component data by type name from all_types."""
        if not self.all_types:
            return None
        for _category, components in self.all_types.items():
            if not isinstance(components, dict):
                continue
            comp = components.get(type_name)
            if isinstance(comp, dict):
                return comp
        return None

    # ── update_node ─────────────────────────────────────────────────────

    async def update_node(self, node_id: str, values: dict) -> dict:
        """Update parameters of a single node without rebuilding the entire flow."""
        existing_nodes, existing_edges = await self._load_current_canvas()
        if not existing_nodes:
            return {"content": [{"type": "text", "text": "Canvas is empty. Build the flow first."}]}

        target_node = None
        for node in existing_nodes:
            if node.get("id") == node_id:
                target_node = node
                break

        if target_node is None:
            available = [n.get("id") for n in existing_nodes]
            return {"content": [{"type": "text", "text": f"Node '{node_id}' not found. Available: {available}"}]}

        node_template = target_node.get("data", {}).get("node", {}).get("template", {})
        updated_fields = []
        for field_name, field_value in values.items():
            if field_name in node_template and isinstance(node_template[field_name], dict):
                node_template[field_name]["value"] = field_value
            else:
                node_template[field_name] = {"value": field_value}
            updated_fields.append(field_name)

        # Save to DB
        self.last_canvas_data = {"nodes": existing_nodes, "edges": existing_edges}
        await self._persist_canvas()

        logger.warning("[Assistant] update_node '%s': updated %s", node_id, updated_fields)
        return {"content": [{"type": "text", "text": f"Node '{node_id}' updated: {updated_fields}"}]}

    # ── delete_nodes ─────────────────────────────────────────────────────

    async def delete_nodes(self, node_ids: list[str]) -> dict:
        """Delete nodes and their connected edges."""
        existing_nodes, existing_edges = await self._load_current_canvas()
        if not existing_nodes:
            return {"content": [{"type": "text", "text": "Canvas is empty. Nothing to delete."}]}

        ids_to_delete = set(node_ids)
        kept_nodes = [n for n in existing_nodes if n.get("id") not in ids_to_delete]
        deleted_count = len(existing_nodes) - len(kept_nodes)

        if deleted_count == 0:
            available = [n.get("id") for n in existing_nodes]
            return {"content": [{"type": "text", "text": f"No matching nodes found. Available: {available}"}]}

        # Remove edges connected to deleted nodes
        kept_edges = [
            e for e in existing_edges
            if e.get("source") not in ids_to_delete and e.get("target") not in ids_to_delete
        ]
        removed_edges = len(existing_edges) - len(kept_edges)

        # Save
        self.last_canvas_data = {"nodes": kept_nodes, "edges": kept_edges}
        self.last_compact_edges = None
        await self._persist_canvas()

        logger.warning("[Assistant] delete_nodes: removed %d nodes, %d edges", deleted_count, removed_edges)
        return {"content": [{"type": "text", "text": f"Deleted {deleted_count} nodes and {removed_edges} connected edges. Canvas: {len(kept_nodes)} nodes, {len(kept_edges)} edges remaining."}]}

    # ── add_edge ─────────────────────────────────────────────────────────

    async def add_edge(self, source: str, source_output: str, target: str, target_input: str) -> dict:
        """Add a connection between two nodes."""
        existing_nodes, existing_edges = await self._load_current_canvas()
        if not existing_nodes:
            return {"content": [{"type": "text", "text": "Canvas is empty. Build the flow first."}]}

        node_map = {n.get("id"): n for n in existing_nodes}
        if source not in node_map:
            return {"content": [{"type": "text", "text": f"Source node '{source}' not found."}]}
        if target not in node_map:
            return {"content": [{"type": "text", "text": f"Target node '{target}' not found."}]}

        # Check if target_input already has a connection (one input can only have one source)
        for e in existing_edges:
            if e.get("target") == target and e.get("targetHandle"):
                # Decode target_handle to extract input field name
                existing_target_input = target_input
                edge_data = e.get("data", {})
                target_handle = edge_data.get("targetHandle", {})
                if isinstance(target_handle, dict):
                    existing_target_input = target_handle.get("id", target_input)
                if existing_target_input == target_input:
                    existing_source = e.get("source", "")
                    return {"content": [{"type": "text", "text": f"Cannot add edge: target input '{target_input}' on node '{target}' already has a connection from '{existing_source}'. Remove the existing edge first."}]}

        new_compact_edge = {"source": source, "source_output": source_output, "target": target, "target_input": target_input}
        expanded_edges = self._expand_edges([new_compact_edge], existing_nodes)
        if not expanded_edges:
            return {"content": [{"type": "text", "text": f"Could not create edge: check that '{source_output}' is a valid output and '{target_input}' is a valid input."}]}

        # Deduplicate
        existing_keys = {
            (e.get("source"), e.get("sourceHandle"), e.get("target"), e.get("targetHandle"))
            for e in existing_edges
        }
        for ne in expanded_edges:
            key = (ne.get("source"), ne.get("sourceHandle"), ne.get("target"), ne.get("targetHandle"))
            if key not in existing_keys:
                existing_edges.append(ne)
                existing_keys.add(key)

        self.last_canvas_data = {"nodes": existing_nodes, "edges": existing_edges}
        await self._persist_canvas()

        logger.warning("[Assistant] add_edge: %s/%s -> %s/%s", source, source_output, target, target_input)
        return {"content": [{"type": "text", "text": f"Edge added: {source}/{source_output} -> {target}/{target_input}. Total edges: {len(existing_edges)}"}]}

    # ── delete_edge ──────────────────────────────────────────────────────

    async def delete_edge(self, source: str, target: str) -> dict:
        """Delete a connection between two nodes."""
        existing_nodes, existing_edges = await self._load_current_canvas()
        if not existing_edges:
            return {"content": [{"type": "text", "text": "No edges on canvas."}]}

        kept_edges = [
            e for e in existing_edges
            if not (e.get("source") == source and e.get("target") == target)
        ]
        removed = len(existing_edges) - len(kept_edges)
        if removed == 0:
            return {"content": [{"type": "text", "text": f"No edge found from '{source}' to '{target}'."}]}

        self.last_canvas_data = {"nodes": existing_nodes, "edges": kept_edges}
        await self._persist_canvas()

        logger.warning("[Assistant] delete_edge: %s -> %s (%d removed)", source, target, removed)
        return {"content": [{"type": "text", "text": f"Deleted {removed} edge(s) from '{source}' to '{target}'. Remaining edges: {len(kept_edges)}"}]}

    # ── run_component ────────────────────────────────────────────────────

    async def run_component(self, node_id: str) -> dict:
        """Run a single component and return its output."""
        from langflow.api.utils.core import build_graph_from_data

        existing_nodes, existing_edges = await self._load_current_canvas()
        if not existing_nodes:
            return {"content": [{"type": "text", "text": "Canvas is empty. Build the flow first."}]}

        node_map = {n.get("id"): n for n in existing_nodes}
        if node_id not in node_map:
            available = list(node_map.keys())
            return {"content": [{"type": "text", "text": f"Node '{node_id}' not found. Available: {available}"}]}

        # Auto-save before running
        if self.last_canvas_data is not None:
            await self._persist_canvas()

        try:
            flow = await get_flow_by_id_or_endpoint_name(
                flow_id_or_name=self.flow_id, user_id=self.user_id,
            )
        except (ValueError, RuntimeError) as e:
            return {"content": [{"type": "text", "text": f"Error loading flow: {e}"}]}

        flow_data = flow.data if flow.data else {"nodes": [], "edges": []}
        try:
            graph = await build_graph_from_data(
                flow_id=self.flow_id,
                payload=flow_data,
                user_id=self.user_id,
            )
            graph.sort_vertices(stop_component_id=node_id)
            vertices_to_run = list(graph.vertices_to_run)
        except Exception as e:  # noqa: BLE001
            logger.exception("[Assistant] run_component graph error: %s", e)
            return {"content": [{"type": "text", "text": f"Error building graph: {e}"}]}

        if node_id not in {v for layer in ([graph.first_layer, *graph.vertices_layers]) for v in layer}:
            return {"content": [{"type": "text", "text": f"Node '{node_id}' is not reachable in the graph. Check connections."}]}

        results = []
        from itertools import chain
        all_layers = [graph.first_layer, *graph.vertices_layers]
        for layer in all_layers:
            for vid in layer:
                try:
                    build_result = await graph.build_vertex(
                        vid,
                        inputs_dict=None,
                        user_id=self.user_id,
                        fallback_to_env_vars=True,
                    )
                    result_dict = build_result.result_dict
                    if result_dict and vid == node_id:
                        results.append({
                            "component": vid,
                            "result": str(result_dict.get("message", "")),
                            "artifacts": str(result_dict.get("artifacts", "")),
                        })
                except Exception as e:  # noqa: BLE001
                    if vid == node_id:
                        results.append({"component": vid, "error": str(e)})
                    logger.warning("[Assistant] run_component vertex %s error: %s", vid, e)

                # Stop after target node
                if vid == node_id:
                    break
            if any(r.get("component") == node_id for r in results):
                break

        logger.warning("[Assistant] run_component '%s': %d results", node_id, len(results))
        return {"content": [{"type": "text", "text": json.dumps(results, ensure_ascii=False, default=str)}]}

    # ── Helper: persist canvas to DB ─────────────────────────────────────

    async def _persist_canvas(self) -> None:
        """Save current last_canvas_data to DB."""
        if not self.last_canvas_data:
            return
        try:
            import uuid

            from langflow.services.deps import session_scope

            async with session_scope() as session:
                from sqlmodel import select
                from langflow.services.database.models.flow.model import Flow

                flow_uuid = uuid.UUID(self.flow_id) if isinstance(self.flow_id, str) else self.flow_id
                db_flow = (await session.exec(select(Flow).where(Flow.id == flow_uuid))).first()
                if db_flow:
                    db_flow.data = self.last_canvas_data
                    session.add(db_flow)
                    await session.commit()
                    logger.warning("[Assistant] canvas persisted to DB")
        except Exception as e:  # noqa: BLE001
            logger.warning("[Assistant] _persist_canvas failed: %s", e)


# ── Provider credential resolution ─────────────────────────────────────


async def _resolve_provider_credentials(
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

        # Strategy 1: Try unified_models helpers
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
            logger.warning("[Assistant] strategy1 api_key=%s base_url=%s", bool(api_key), base_url or "N/A")
        except Exception:  # noqa: BLE001
            logger.warning("Strategy 1 failed, trying direct DB lookup", exc_info=True)

        # Strategy 2: Direct DB lookup via session_scope
        if not api_key and user_id and provider:
            try:
                from lfx.services.deps import get_settings_service, session_scope
                from sqlmodel import select

                from langflow.services.database.models.variable.model import Variable

                # Build expected variable names from provider
                provider_upper = re.sub(r"[^A-Za-z0-9]", "", provider).upper()
                api_key_var = f"{provider_upper}_API_KEY"
                base_url_var = f"{provider_upper}_BASE_URL"

                async with session_scope() as session:
                    stmt = select(Variable).where(
                        Variable.user_id == user_id,
                        Variable.name.in_([api_key_var, base_url_var]),
                    )
                    rows = (await session.exec(stmt)).all()
                    for row in rows:
                        if row.name == api_key_var and row.value:
                            api_key = row.value
                        elif row.name == base_url_var and row.value:
                            base_url = row.value

                logger.warning("[Assistant] strategy2 api_key=%s base_url=%s", bool(api_key), base_url or "N/A")
            except Exception:  # noqa: BLE001
                logger.warning("Strategy 2 (direct DB) failed", exc_info=True)

    # Fallback: environment variables
    if not api_key:
        api_key = (
            os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
        )
    if not base_url:
        base_url = (
            os.environ.get("OPENAI_BASE_URL", "")
            or os.environ.get("OPENAI_API_BASE", "")
            or os.environ.get("ANTHROPIC_BASE_URL", "")
        )

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
            result = await ctx.build_flow(arguments.get("nodes", []), arguments.get("edges", []), arguments.get("clear", False))
        elif tool_name == "run_flow":
            result = await ctx.run_flow(arguments.get("input_value", ""))
        elif tool_name == "get_current_flow":
            result = await ctx.get_current_flow()
        elif tool_name == "prepare_flow":
            result = await ctx.prepare_flow()
        elif tool_name == "update_node":
            result = await ctx.update_node(arguments.get("node_id", ""), arguments.get("values", {}))
        elif tool_name == "delete_nodes":
            result = await ctx.delete_nodes(arguments.get("node_ids", []))
        elif tool_name == "add_edge":
            result = await ctx.add_edge(
                arguments.get("source", ""),
                arguments.get("source_output", ""),
                arguments.get("target", ""),
                arguments.get("target_input", ""),
            )
        elif tool_name == "delete_edge":
            result = await ctx.delete_edge(arguments.get("source", ""), arguments.get("target", ""))
        elif tool_name == "run_component":
            result = await ctx.run_component(arguments.get("node_id", ""))
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


async def _get_canvas_context(ctx: _ToolContext) -> str:
    """Get current canvas state as a compact context string for the system prompt."""
    try:
        flow = await get_flow_by_id_or_endpoint_name(
            flow_id_or_name=ctx.flow_id, user_id=ctx.user_id,
        )
    except (ValueError, RuntimeError):
        return ""

    flow_data = flow.data
    if not flow_data:
        return ""

    await ctx.ensure_types()

    compact_nodes = []
    for node in flow_data.get("nodes", []):
        node_data = node.get("data", {})
        node_type = node_data.get("type", "unknown")
        template = node_data.get("node", {}).get("template", {})
        values = {}
        for field_name, field_data in template.items():
            if not isinstance(field_data, dict) or field_name.startswith("_"):
                continue
            val = field_data.get("value")
            if val is not None and val != "" and val != []:
                values[field_name] = val
        compact_nodes.append({"id": node.get("id", ""), "type": node_type, "values": values})

    compact_edges = []
    for edge in flow_data.get("edges", []):
        edge_data = edge.get("data", {})
        sh = edge_data.get("sourceHandle", {})
        th = edge_data.get("targetHandle", {})
        compact_edges.append({
            "source": edge.get("source", ""),
            "source_output": sh.get("name", ""),
            "target": edge.get("target", ""),
            "target_input": th.get("fieldName", ""),
        })

    if not compact_nodes:
        return ""

    context = {"nodes": compact_nodes, "edges": compact_edges}
    return json.dumps(context, ensure_ascii=False)


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
        # New session — inject canvas context into system prompt
        canvas_ctx = await _get_canvas_context(ctx)
        system_content = SYSTEM_PROMPT
        if canvas_ctx:
            system_content += f"\n\nCURRENT CANVAS STATE (do NOT duplicate these nodes):\n{canvas_ctx}"
        messages = [{"role": "system", "content": system_content}]
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

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0), trust_env=False) as client:
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
                        "[Assistant] canvas_update: %d nodes, %d compact_edges",
                        len(ctx.last_canvas_data.get("nodes", [])),
                        len(ctx.last_compact_edges or []),
                    )
                    canvas_payload = {
                        "nodes": ctx.last_canvas_data.get("nodes", []),
                        "compact_edges": ctx.last_compact_edges or [],
                    }
                    yield ("canvas_update", canvas_payload)
                    # Do NOT clear last_canvas_data here — run_flow needs it
                    # for auto-save. It will be cleared after all tools in this
                    # turn have finished executing.

                # Add to message history in appropriate format
                if use_native and tc_id:
                    messages.append({"role": "tool", "tool_call_id": tc_id, "content": result_text})
                else:
                    tool_results_parts.append(
                        f"<tool_result_name>{tool_name}</tool_result_name>\n"
                        f"<tool_result>{result_text}</tool_result>"
                    )

            # Clear canvas data after ALL tools in this turn have executed,
            # so that run_flow can still auto-save from a prior build_flow.
            ctx.last_canvas_data = None
            ctx.last_compact_edges = None

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
        api_key, api_url, model_name = await _resolve_provider_credentials(body.selected_model, user_id)
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
