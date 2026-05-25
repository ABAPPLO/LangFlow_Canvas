"""Common MCP handler functions shared between mcp.py and mcp_projects.py.

This module serves as the single source of truth for MCP functionality.
"""

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4

from lfx.base.mcp.constants import MAX_MCP_TOOL_NAME_LENGTH
from lfx.base.mcp.util import get_flow_snake_case, get_unique_name, sanitize_mcp_name
from lfx.log.logger import logger
from lfx.utils.helpers import build_content_type_from_extension
from mcp import types
from sqlmodel import select

from langflow.api.v1.endpoints import simple_run_flow
from langflow.api.v1.schemas import SimplifiedAPIRequest
from langflow.helpers.flow import get_mcp_input_parameters, json_schema_from_flow
from langflow.schema.data import Data
from langflow.schema.message import Message
from langflow.serialization.serialization import serialize
from langflow.services.database.models import Flow
from langflow.services.database.models.file.model import File as UserFile
from langflow.services.database.models.user.model import User
from langflow.services.deps import get_settings_service, get_storage_service, session_scope

T = TypeVar("T")
P = ParamSpec("P")

MCP_SERVERS_FILE = "_mcp_servers"

# Create context variables
current_user_ctx: ContextVar[User] = ContextVar("current_user_ctx")
# Carries per-request variables injected via HTTP headers (e.g., X-Langflow-Global-Var-*)
current_request_variables_ctx: ContextVar[dict[str, str] | None] = ContextVar(
    "current_request_variables_ctx", default=None
)


class MCPToolInputError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def build_tweaks_from_mcp_arguments(flow: Flow, arguments: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    parameters = get_mcp_input_parameters(flow)
    arguments = arguments or {}
    allowed_names = {
        parameter.get("parameter_name")
        for parameter in parameters
        if isinstance(parameter, dict) and parameter.get("parameter_name")
    }

    for parameter in parameters:
        if not isinstance(parameter, dict) or not parameter.get("parameter_name"):
            continue
        name = parameter["parameter_name"]
        if parameter.get("required", False) and (name not in arguments or arguments[name] is None):
            code = "missing_required_parameter"
            raise MCPToolInputError(
                code,
                f"Missing required parameter: {name}",
            )
        if parameter.get("required", False) and isinstance(arguments.get(name), str) and not arguments[name].strip():
            code = "empty_required_parameter"
            raise MCPToolInputError(
                code,
                f"Required parameter cannot be empty: {name}",
            )

    for name in arguments:
        if name not in allowed_names:
            code = "unknown_parameter"
            raise MCPToolInputError(
                code,
                f"Unknown parameter: {name}",
            )

    tweaks: dict[str, dict[str, str]] = {}
    for parameter in parameters:
        if not isinstance(parameter, dict):
            continue
        name = parameter.get("parameter_name")
        component_id = parameter.get("component_id")
        if not name or not component_id or name not in arguments:
            continue
        value = arguments[name]
        if not isinstance(value, str):
            code = "invalid_parameter_type"
            raise MCPToolInputError(
                code,
                f"Parameter {name} must be a string. Serialize complex values to JSON strings before calling the tool.",
            )
        field = parameter.get("field") or "input_value"
        tweaks.setdefault(component_id, {})[field] = value
    return tweaks


def build_mcp_error_content(
    *,
    flow: Flow | None,
    tool_name: str,
    code: str,
    message: str,
) -> types.TextContent:
    envelope = {
        "flow_id": str(flow.id) if flow else None,
        "flow_name": flow.name if flow else None,
        "tool_name": tool_name,
        "success": False,
        "outputs": {},
        "error": {
            "code": code,
            "message": message,
        },
    }
    return types.TextContent(type="text", text=json.dumps(envelope, ensure_ascii=False))


def _json_content(value: Any) -> Any:
    if isinstance(value, Message):
        return value.get_text()
    if isinstance(value, Data):
        return serialize(value.data)
    if isinstance(value, dict):
        return {key: _json_content(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_content(item) for item in value]
    return serialize(value)


def _result_data_content(result_data: Any) -> Any:
    results = getattr(result_data, "results", None)
    if results:
        if isinstance(results, dict):
            if len(results) == 1:
                return _json_content(next(iter(results.values())))
            return {key: _json_content(value) for key, value in results.items()}
        return _json_content(results)

    messages = getattr(result_data, "messages", None) or []
    message_texts = [message.message for message in messages if getattr(message, "message", None)]
    if len(message_texts) == 1:
        return message_texts[0]
    if message_texts:
        return message_texts

    outputs = getattr(result_data, "outputs", None)
    if outputs:
        return _json_content(outputs)
    return None


def _mcp_output_type(component_id: str, result_data: Any, output_vertices: dict[str, Any]) -> str:
    vertex = output_vertices.get(component_id)
    vertex_type = getattr(vertex, "vertex_type", None)
    if vertex_type == "DataOutput":
        return "data"
    if vertex_type == "ChatOutput":
        return "message"
    if vertex_type == "TextOutput":
        return "text"

    results = getattr(result_data, "results", None)
    if isinstance(results, dict):
        if any(isinstance(value, Data) for value in results.values()):
            return "data"
        if any(isinstance(value, Message) for value in results.values()):
            return "message"
    return str(vertex_type or "unknown").lower()


def build_mcp_tool_output_envelope(flow: Flow, run_response: Any, tool_name: str | None = None) -> dict[str, Any]:
    from lfx.graph.graph.base import Graph

    graph = Graph.from_payload(flow.data or {})
    output_vertices = {vertex.id: vertex for vertex in graph.vertices if vertex.is_output}
    result_data_by_component_id: dict[str, Any] = {}

    if run_response.outputs:
        for run_output in run_response.outputs:
            for result_data in getattr(run_output, "outputs", []) or []:
                component_id = getattr(result_data, "component_id", None)
                if component_id:
                    result_data_by_component_id[component_id] = result_data

    outputs: dict[str, dict[str, Any]] = {}
    component_ids = list(output_vertices) or list(result_data_by_component_id)
    for component_id in component_ids:
        result_data = result_data_by_component_id.get(component_id)
        vertex = output_vertices.get(component_id)
        display_name = (
            getattr(result_data, "component_display_name", None)
            or getattr(vertex, "display_name", None)
            or component_id
        )
        outputs[component_id] = {
            "display_name": display_name,
            "type": _mcp_output_type(component_id, result_data, output_vertices) if result_data else "unknown",
            "content": _result_data_content(result_data) if result_data else None,
        }

    resolved_tool_name = tool_name or (
        sanitize_mcp_name(flow.action_name) if flow.action_name else sanitize_mcp_name(flow.name)
    )
    return {
        "flow_id": str(flow.id),
        "flow_name": flow.name,
        "tool_name": resolved_tool_name,
        "success": True,
        "outputs": outputs,
        "error": None,
    }


def handle_mcp_errors(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
    """Decorator to handle MCP endpoint errors consistently."""

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            msg = f"Error in {func.__name__}: {e!s}"
            await logger.aexception(msg)
            raise

    return wrapper


async def with_db_session(operation: Callable[[Any], Awaitable[T]]) -> T:
    """Execute an operation within a database session context."""
    async with session_scope() as session:
        return await operation(session)


class MCPConfig:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.enable_progress_notifications = None
        return cls._instance


def get_mcp_config():
    return MCPConfig()


async def handle_list_resources(project_id=None):
    """Handle listing resources for MCP.

    Args:
        project_id: Optional project ID to filter resources by project
    """
    resources = []
    try:
        storage_service = get_storage_service()
        settings_service = get_settings_service()

        # Build full URL from settings
        host = getattr(settings_service.settings, "host", "localhost")
        port = getattr(settings_service.settings, "port", 3000)

        base_url = f"http://{host}:{port}".rstrip("/")
        try:
            current_user = current_user_ctx.get()
        except Exception as e:  # noqa: BLE001
            msg = f"Error getting current user: {e!s}"
            await logger.aexception(msg)
            current_user = None
        async with session_scope() as session:
            # Build query based on whether project_id is provided
            flows_query = select(Flow).where(Flow.folder_id == project_id) if project_id else select(Flow)

            flows = (await session.exec(flows_query)).all()

            for flow in flows:
                if flow.id:
                    try:
                        files = await storage_service.list_files(flow_id=str(flow.id))
                        for file_name in files:
                            # URL encode the filename
                            safe_filename = quote(file_name)
                            resource = types.Resource(
                                uri=f"{base_url}/api/v1/files/download/{flow.id}/{safe_filename}",
                                name=file_name,
                                description=f"File in flow: {flow.name}",
                                mimeType=build_content_type_from_extension(file_name),
                            )
                            resources.append(resource)
                    except FileNotFoundError as e:
                        msg = f"Error listing files for flow {flow.id}: {e}"
                        await logger.adebug(msg)
                        continue
            ####################################################
            # When a user uploads a file inside a flow
            # (e.g., via the File Read component),
            # it hits /api/v2/files (POST),
            # which saves files at the user-level.
            # So the above query for flow files is not enough.
            # So we list all user files for the current user.
            # This is not good. We need to fix this for 1.8.0.
            ###################################################
            if current_user:
                user_files_stmt = select(UserFile).where(UserFile.user_id == current_user.id)
                user_files = (await session.exec(user_files_stmt)).all()
                for user_file in user_files:
                    stored_path = getattr(user_file, "path", "") or ""
                    stored_filename = Path(stored_path).name if stored_path else user_file.name
                    safe_filename = quote(stored_filename)
                    if stored_filename.startswith(f"{MCP_SERVERS_FILE}_{current_user.id}"):
                        # reserved file name for langflow MCP server config file(s)
                        continue
                    description = getattr(user_file, "provider", None) or "User file uploaded via File Manager"
                    resource = types.Resource(
                        uri=f"{base_url}/api/v1/files/download/{current_user.id}/{safe_filename}",
                        name=stored_filename,
                        description=description,
                        mimeType=build_content_type_from_extension(stored_filename),
                    )
                    resources.append(resource)
    except Exception as e:
        msg = f"Error in listing resources: {e!s}"
        await logger.aexception(msg)
        raise
    return resources


async def handle_read_resource(uri: str) -> bytes:
    """Handle resource read requests."""
    try:
        # Parse the URI properly
        parsed_uri = urlparse(str(uri))
        # Path will be like /api/v1/files/download/{namespace}/{filename}
        path_parts = parsed_uri.path.split("/")
        # Remove empty strings from split
        path_parts = [p for p in path_parts if p]

        # The flow_id and filename should be the last two parts
        two = 2
        if len(path_parts) < two:
            msg = f"Invalid URI format: {uri}"
            raise ValueError(msg)

        flow_id = path_parts[-2]
        filename = unquote(path_parts[-1])  # URL decode the filename

        storage_service = get_storage_service()

        # Read the file content
        content = await storage_service.get_file(flow_id=flow_id, file_name=filename)
        if not content:
            msg = f"File {filename} not found in flow {flow_id}"
            raise ValueError(msg)

        # Ensure content is base64 encoded
        if isinstance(content, str):
            content = content.encode()
        return base64.b64encode(content)
    except Exception as e:
        msg = f"Error reading resource {uri}: {e!s}"
        await logger.aexception(msg)
        raise


async def handle_call_tool(
    name: str, arguments: dict, server, project_id=None, *, is_action=False
) -> list[types.TextContent]:
    """Handle tool execution requests.

    Args:
        name: Tool name
        arguments: Tool arguments
        server: MCP server instance
        project_id: Optional project ID to filter flows by project
        is_action: Whether to use action name for flow lookup
    """
    mcp_config = get_mcp_config()
    if mcp_config.enable_progress_notifications is None:
        settings_service = get_settings_service()
        mcp_config.enable_progress_notifications = settings_service.settings.mcp_server_enable_progress_notifications

    current_user = current_user_ctx.get()
    # Build execution context with request-level variables if present
    request_variables = current_request_variables_ctx.get()
    exec_context = {"request_variables": request_variables} if request_variables else None

    async def execute_tool(session):
        # Get flow id from name
        flow = await get_flow_snake_case(name, current_user.id, session, is_action=is_action)
        if not flow:
            msg = f"Flow with name '{name}' not found"
            raise ValueError(msg)

        # If project_id is provided, verify the flow belongs to the project
        if project_id and flow.folder_id != project_id:
            msg = f"Flow '{name}' not found in project {project_id}"
            raise ValueError(msg)

        try:
            tweaks = build_tweaks_from_mcp_arguments(flow, arguments)
        except MCPToolInputError as e:
            return [build_mcp_error_content(flow=flow, tool_name=name, code=e.code, message=e.message)]

        # Initial progress notification
        if mcp_config.enable_progress_notifications and (progress_token := server.request_context.meta.progressToken):
            await server.request_context.session.send_progress_notification(
                progress_token=progress_token, progress=0.0, total=1.0
            )

        conversation_id = str(uuid4())
        input_request = SimplifiedAPIRequest(
            input_value=None,
            tweaks=tweaks,
            output_type="any",
            session_id=conversation_id,
        )

        async def send_progress_updates(progress_token):
            try:
                progress = 0.0
                while True:
                    await server.request_context.session.send_progress_notification(
                        progress_token=progress_token, progress=min(0.9, progress), total=1.0
                    )
                    progress += 0.1
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                if mcp_config.enable_progress_notifications:
                    await server.request_context.session.send_progress_notification(
                        progress_token=progress_token, progress=1.0, total=1.0
                    )
                raise

        try:
            progress_task = None
            if mcp_config.enable_progress_notifications and server.request_context.meta.progressToken:
                progress_task = asyncio.create_task(send_progress_updates(server.request_context.meta.progressToken))

            try:
                try:
                    result = await simple_run_flow(
                        flow=flow,
                        input_request=input_request,
                        stream=False,
                        api_key_user=current_user,
                        context=exec_context,
                    )
                    envelope = build_mcp_tool_output_envelope(flow, result, tool_name=name)
                    return [types.TextContent(type="text", text=json.dumps(envelope, ensure_ascii=False))]
                except Exception as e:  # noqa: BLE001
                    error_msg = f"Error Executing the {flow.name} tool. Error: {e!s}"
                    return [
                        build_mcp_error_content(
                            flow=flow,
                            tool_name=name,
                            code="tool_execution_failed",
                            message=error_msg,
                        )
                    ]
            finally:
                if progress_task:
                    progress_task.cancel()
                    await asyncio.gather(progress_task, return_exceptions=True)

        except Exception:
            if mcp_config.enable_progress_notifications and (
                progress_token := server.request_context.meta.progressToken
            ):
                await server.request_context.session.send_progress_notification(
                    progress_token=progress_token, progress=1.0, total=1.0
                )
            raise

    try:
        return await with_db_session(execute_tool)
    except Exception as e:
        msg = f"Error executing tool {name}: {e!s}"
        await logger.aexception(msg)
        raise


async def handle_list_tools(project_id=None, *, mcp_enabled_only=False):
    """Handle listing tools for MCP.

    Args:
        project_id: Optional project ID to filter tools by project
        mcp_enabled_only: Whether to filter for MCP-enabled flows only
    """
    tools = []
    try:
        async with session_scope() as session:
            # Build query based on parameters
            if project_id:
                # Filter flows by project and optionally by MCP enabled status
                flows_query = select(Flow).where(Flow.folder_id == project_id, Flow.is_component == False)  # noqa: E712
                if mcp_enabled_only:
                    flows_query = flows_query.where(Flow.mcp_enabled == True)  # noqa: E712
            else:
                # Get all flows
                flows_query = select(Flow)

            flows = (await session.exec(flows_query)).all()

            existing_names = set()
            for flow in flows:
                if flow.user_id is None:
                    continue

                # For project-specific tools, use action names if available
                if project_id:
                    base_name = (
                        sanitize_mcp_name(flow.action_name) if flow.action_name else sanitize_mcp_name(flow.name)
                    )
                    name = get_unique_name(base_name, MAX_MCP_TOOL_NAME_LENGTH, existing_names)
                    description = flow.action_description or (
                        flow.description if flow.description else f"Tool generated from flow: {name}"
                    )
                else:
                    # For global tools, use simple sanitized names
                    base_name = sanitize_mcp_name(flow.name)
                    name = base_name[:MAX_MCP_TOOL_NAME_LENGTH]
                    if name in existing_names:
                        i = 1
                        while True:
                            suffix = f"_{i}"
                            truncated_base = base_name[: MAX_MCP_TOOL_NAME_LENGTH - len(suffix)]
                            candidate = f"{truncated_base}{suffix}"
                            if candidate not in existing_names:
                                name = candidate
                                break
                            i += 1
                    description = (
                        f"{flow.id}: {flow.description}" if flow.description else f"Tool generated from flow: {name}"
                    )

                try:
                    tool = types.Tool(
                        name=name,
                        description=description,
                        inputSchema=json_schema_from_flow(flow),
                    )
                    tools.append(tool)
                    existing_names.add(name)
                except Exception as e:  # noqa: BLE001
                    msg = f"Error in listing tools: {e!s} from flow: {base_name}"
                    await logger.awarning(msg)
                    continue
    except Exception as e:
        msg = f"Error in listing tools: {e!s}"
        await logger.aexception(msg)
        raise
    return tools
