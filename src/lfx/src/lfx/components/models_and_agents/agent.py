from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from lfx.components.models_and_agents.memory import MemoryComponent

if TYPE_CHECKING:
    from langchain_core.tools import Tool

from lfx.base.agents.agent import LCToolsAgentComponent
from lfx.base.agents.callback import ToolCallRecorder
from lfx.base.agents.events import ExceptionWithMessageError
from lfx.base.models.unified_models import (
    apply_provider_variable_config_to_build_config,
    get_language_model_options,
    get_llm,
    get_provider_for_model_name,
    update_model_options_in_build_config,
)
from lfx.base.models.watsonx_constants import IBM_WATSONX_URLS
from lfx.components.helpers import CurrentDateComponent
from lfx.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from lfx.custom.custom_component.component import get_component_toolkit
from lfx.field_typing.range_spec import RangeSpec
from lfx.helpers.base_model import build_model_from_schema
from lfx.inputs.inputs import BoolInput, DropdownInput, ModelInput, StrInput
from lfx.io import IntInput, MessageTextInput, MultilineInput, Output, SecretStrInput, TableInput
from lfx.log.logger import logger
from lfx.schema.data import Data
from lfx.schema.dotdict import dotdict
from lfx.schema.message import Message
from lfx.schema.table import EditMode

TOOL_RESULT_MODE_DIRECT_SUMMARY = "Direct Tool Summary"
TOOL_RESULT_MODE_AGENT_LOOP = "Agent Loop"
DEFAULT_AGENT_SYSTEM_PROMPT = """你是一个严谨的工作流调度 Agent。你的职责是根据用户需求选择最合适的工具，并严格按照工具的 inputSchema 构造参数后调用工具。

你通常会收到一个 JSON 字符串：

{
  "requirement": "本次要完成的需求",
  "parameters": {
    "参数名": "参数值"
  }
}

执行规则：

1. 只根据当前请求的 requirement、parameters、可用工具的名称、描述、inputSchema，以及同一 session 的最近历史用户请求来选择工具和准备参数。
2. 默认只调用一个最匹配的工具；不要调用无关工具。
3. 工具参数名必须严格来自目标工具的 inputSchema，不要编造 schema 中不存在的参数，不要传入 schema 外参数。
4. 当前请求中的非空 parameters 优先级最高；如果当前请求提供了非空参数，必须使用当前请求的值，不要用历史值覆盖。
5. 如果目标工具的 required 参数在当前请求中缺失、为空字符串、纯空白字符串，或语义上没有有效内容，允许从同一 session 最近历史用户请求中查找同名或语义等价的非空参数进行补齐。
6. 历史补齐只允许用于 required 参数；非 required 参数如果当前请求没有提供有效值，不要从历史补齐，也不要传入工具。
7. 只能使用历史中的用户请求参数补齐，不要使用历史 AI 回答、Full Response、工具输出、错误提示或示例内容作为参数来源。
8. 如果历史中存在多个候选参数，使用距离当前请求最近且非空的用户参数。
9. 必填 string 参数最终必须存在，且不能为空字符串或纯空白字符串。
10. 如果 required 参数在当前请求中缺失或为空，并且历史请求中也找不到可用的非空值，直接说明缺少哪些参数，不要调用工具。
11. 当前工作流输入节点只支持 string；如果参数值是对象、数组或其他复杂结构，必须先序列化为 JSON 字符串后再传给工具。
12. 调用工具前，逐项确认最终参数名与 inputSchema 完全一致，required 参数都已满足，非 required 参数没有被历史补齐，且没有传入 schema 外参数。
13. 返回结果时，用用户能理解的语言简洁说明执行结果；不要暴露内部工具调用细节，除非用户明确要求。"""
DEFAULT_TOOL_SUCCESS_SUMMARY_PROMPT = (
    "你是一个面向终端用户的结果总结助手。\n\n"
    "工具已经执行成功。请根据用户本次请求、工具名称、工具入参和工具返回结果，"
    "生成一段清晰、准确、用户友好的中文回复。\n\n"
    "要求：\n"
    "1. 只总结工具返回结果中真实存在的信息，不要编造。\n"
    "2. 不要暴露内部字段名、组件 ID、tool_calls、raw_output、JSON 信封等技术细节。\n"
    "3. 如果工具返回了多个输出，优先总结与用户需求最相关的内容。\n"
    "4. 如果工具结果本身已经是完整文本，可以在保持原意的基础上适度整理表达。\n"
    "5. 回复要简洁，但不能遗漏关键结论。"
)
DEFAULT_TOOL_FAILURE_SUMMARY_PROMPT = (
    "你是一个面向终端用户的错误说明助手。\n\n"
    "工具调用失败了。请根据用户本次请求、工具名称、工具入参和工具错误信息，"
    "生成一段清晰、准确、可操作的中文回复。\n\n"
    "要求：\n"
    "1. 明确说明本次没有成功完成工具执行，不要说“执行成功”。\n"
    "2. 如果错误来自缺少必填参数、参数为空或参数格式不正确，请明确指出需要补充或修改的参数。\n"
    "3. 不要暴露内部字段名、组件 ID、tool_calls、raw_output、JSON 信封等技术细节，"
    "除非参数名本身就是用户需要填写的业务参数。\n"
    "4. 不要编造工具没有返回的信息。\n"
    "5. 回复要给出下一步操作建议，让用户知道应该如何修正后重试。"
)


def _agent_input_text(input_value) -> str:
    if isinstance(input_value, Message):
        return input_value.get_text()
    return str(input_value or "")


def _parse_agent_task(input_value) -> dict:
    text = _agent_input_text(input_value)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _message_text(message: Message) -> str:
    if hasattr(message, "get_text"):
        return message.get_text()
    return getattr(message, "text", "") or str(message)


def _lc_message_text(message: Any) -> str:
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if text:
                    text_parts.append(str(text))
            elif item is not None:
                text_parts.append(str(item))
        return "\n".join(text_parts)
    if content is not None:
        return str(content)
    return str(message or "")


def _parse_tool_args(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {"input": value}
        return parsed if isinstance(parsed, dict) else {"input": value}
    return {}


def _tool_output_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return json.dumps(output, ensure_ascii=False)
    content = getattr(output, "content", None)
    if isinstance(content, str):
        return content
    if content is not None:
        text_parts = []
        for item in content:
            text = getattr(item, "text", None)
            text_parts.append(text if isinstance(text, str) else str(item))
        return "\n".join(text_parts)
    if hasattr(output, "model_dump"):
        return json.dumps(output.model_dump(), ensure_ascii=False)
    return str(output)


def _extract_final_outputs(tool_calls: list[dict]) -> dict:
    data_outputs: dict[str, object] = {}
    for tool_call in tool_calls:
        raw_output = tool_call.get("raw_output")
        if not isinstance(raw_output, dict):
            continue
        outputs = raw_output.get("outputs")
        if not isinstance(outputs, dict):
            continue
        for component_id, output in outputs.items():
            if isinstance(output, dict) and output.get("type") == "data":
                data_outputs[component_id] = output.get("content")

    if len(data_outputs) == 1:
        only_output = next(iter(data_outputs.values()))
        return only_output if isinstance(only_output, dict) else {"result": only_output}
    return data_outputs


def _extract_errors(tool_calls: list[dict]) -> list[dict]:
    errors: list[dict] = []
    for tool_call in tool_calls:
        error = tool_call.get("error")
        raw_output = tool_call.get("raw_output")
        if not error and isinstance(raw_output, dict):
            error = raw_output.get("error")
        if error:
            if isinstance(error, dict):
                errors.append({"tool_name": tool_call.get("tool_name"), **error})
            else:
                errors.append(
                    {
                        "tool_name": tool_call.get("tool_name"),
                        "code": "tool_execution_failed",
                        "message": str(error),
                    }
                )
    return errors


def _latest_tool_error(recorder: ToolCallRecorder | None) -> dict | None:
    if not recorder or not recorder.tool_calls:
        return None
    latest_call = recorder.tool_calls[-1]
    if latest_call.get("success") is False:
        error = latest_call.get("error")
        if isinstance(error, dict):
            return error
        if error:
            return {"code": "tool_execution_failed", "message": str(error)}
    return None


def build_full_response(
    result: Message,
    tool_calls: list[dict],
    input_value,
    request_variables: dict | None = None,
) -> dict:
    task = _parse_agent_task(input_value)
    request_id = task.get("request_id")
    if not request_id and request_variables:
        request_id = request_variables.get("TASK-ID") or request_variables.get("TASK_ID")
    selected_call = next((call for call in tool_calls if call.get("success")), tool_calls[0] if tool_calls else None)
    return {
        "request_id": request_id,
        "answer": _message_text(result),
        "selected_tool": selected_call.get("tool_name") if selected_call else None,
        "tool_calls": tool_calls,
        "final_outputs": _extract_final_outputs(tool_calls),
        "errors": _extract_errors(tool_calls),
    }


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


class AgentComponent(ToolCallingAgentComponent):
    display_name: str = "Agent"
    description: str = "Define the agent's instructions, then enter a task to complete using tools."
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "Agent"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        ModelInput(
            name="model",
            display_name="Language Model",
            info="Select your model provider",
            real_time_refresh=True,
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="Model Provider API key",
            real_time_refresh=True,
            advanced=True,
        ),
        DropdownInput(
            name="base_url_ibm_watsonx",
            display_name="watsonx API Endpoint",
            info="The base URL of the API (IBM watsonx.ai only)",
            options=IBM_WATSONX_URLS,
            value=IBM_WATSONX_URLS[0],
            show=False,
            real_time_refresh=True,
        ),
        StrInput(
            name="project_id",
            display_name="watsonx Project ID",
            info="The project ID associated with the foundation model (IBM watsonx.ai only)",
            show=False,
            required=False,
        ),
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value=DEFAULT_AGENT_SYSTEM_PROMPT,
            advanced=False,
        ),
        DropdownInput(
            name="tool_result_mode",
            display_name="Tool Result Mode",
            info=(
                "Agent Loop keeps the original LangChain tool loop. Direct Tool Summary runs one tool call, "
                "stores the full tool result, then calls the same model once more to summarize it."
            ),
            options=[TOOL_RESULT_MODE_DIRECT_SUMMARY, TOOL_RESULT_MODE_AGENT_LOOP],
            value=TOOL_RESULT_MODE_DIRECT_SUMMARY,
            real_time_refresh=True,
            advanced=False,
        ),
        MultilineInput(
            name="tool_success_summary_prompt",
            display_name="Tool Success Summary Prompt",
            info="Instructions used when Direct Tool Summary mode receives a successful tool result.",
            value=DEFAULT_TOOL_SUCCESS_SUMMARY_PROMPT,
            show=True,
            advanced=False,
        ),
        MultilineInput(
            name="tool_failure_summary_prompt",
            display_name="Tool Failure Summary Prompt",
            info="Instructions used when Direct Tool Summary mode receives a failed tool result.",
            value=DEFAULT_TOOL_FAILURE_SUMMARY_PROMPT,
            show=True,
            advanced=False,
        ),
        MessageTextInput(
            name="context_id",
            display_name="Context ID",
            info="The context ID of the chat. Adds an extra layer to the local memory.",
            value="",
            advanced=True,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Chat History Messages",
            value=0,
            info="Number of chat history messages to retrieve.",
            advanced=False,
            show=True,
            range_spec=RangeSpec(min=0, step=1, step_type="int"),
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="Maximum number of tokens to generate. Field name varies by provider.",
            advanced=True,
            range_spec=RangeSpec(min=1, max=128000, step=1, step_type="int"),
        ),
        MultilineInput(
            name="format_instructions",
            display_name="Output Format Instructions",
            info="Generic Template for structured output formatting. Valid only with Structured response.",
            value=(
                "You are an AI that extracts structured JSON objects from unstructured text. "
                "Use a predefined schema with expected types (str, int, float, bool, dict). "
                "Extract ALL relevant instances that match the schema - if multiple patterns exist, capture them all. "
                "Fill missing or ambiguous values with defaults: null for missing values. "
                "Remove exact duplicates but keep variations that have different field values. "
                "Always return valid JSON in the expected format, never throw errors. "
                "If multiple objects can be extracted, return them all in the structured format."
            ),
            advanced=True,
        ),
        TableInput(
            name="output_schema",
            display_name="Output Schema",
            info=(
                "Schema Validation: Define the structure and data types for structured output. "
                "No validation if no output schema."
            ),
            advanced=True,
            required=False,
            value=[],
            table_schema=[
                {
                    "name": "name",
                    "display_name": "Name",
                    "type": "str",
                    "description": "Specify the name of the output field.",
                    "default": "field",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "description",
                    "display_name": "Description",
                    "type": "str",
                    "description": "Describe the purpose of the output field.",
                    "default": "description of field",
                    "edit_mode": EditMode.POPOVER,
                },
                {
                    "name": "type",
                    "display_name": "Type",
                    "type": "str",
                    "edit_mode": EditMode.INLINE,
                    "description": ("Indicate the data type of the output field (e.g., str, int, float, bool, dict)."),
                    "options": ["str", "int", "float", "bool", "dict"],
                    "default": "str",
                },
                {
                    "name": "multiple",
                    "display_name": "As List",
                    "type": "boolean",
                    "description": "Set to True if this output field should be a list of the specified type.",
                    "default": "False",
                    "edit_mode": EditMode.INLINE,
                },
            ],
        ),
        *LCToolsAgentComponent.get_base_inputs(),
        # removed memory inputs from agent component
        # *memory_inputs,
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
    ]
    outputs = [
        Output(name="response", display_name="Response", method="message_response", group_outputs=True),
        Output(
            name="full_response",
            display_name="Full Response",
            method="full_response",
            group_outputs=True,
            tool_mode=False,
        ),
    ]

    def _get_max_tokens_value(self):
        """Return the user-supplied max_tokens or None when unset/zero."""
        val = getattr(self, "max_tokens", None)
        if val in {"", 0}:
            return None
        return val

    def _get_llm(self):
        """Override parent to include Agent fields and request-level model headers."""
        wallet_id = getattr(self, "user_wallet_id", None)
        task_id = getattr(self, "task_id", None)
        if not wallet_id or not task_id:
            request_variables = None
            if hasattr(self, "graph") and self.graph and hasattr(self.graph, "context"):
                request_variables = self.graph.context.get("request_variables")
            if request_variables:
                if not wallet_id:
                    wallet_id = request_variables.get("USER-WALLET-ID")
                if not task_id:
                    task_id = request_variables.get("TASK-ID")

        self.log(f"user_wallet_id={wallet_id}, task_id={task_id}")
        return get_llm(
            model=self.model,
            user_id=self.user_id,
            api_key=getattr(self, "api_key", None),
            max_tokens=self._get_max_tokens_value(),
            watsonx_url=getattr(self, "base_url_ibm_watsonx", None),
            watsonx_project_id=getattr(self, "project_id", None),
            user_wallet_id=wallet_id,
            task_id=task_id,
        )

    async def get_agent_requirements(self):
        """Get the agent requirements for the agent."""
        from langchain_core.tools import StructuredTool

        llm_model = self._get_llm()
        if llm_model is None:
            msg = "No language model selected. Please choose a model to proceed."
            raise ValueError(msg)

        # Get memory data
        self.chat_history = await self.get_memory_data()
        await logger.adebug(f"Retrieved {len(self.chat_history)} chat history messages")
        if isinstance(self.chat_history, Message):
            self.chat_history = [self.chat_history]

        # Add current date tool if enabled
        if self.add_current_date_tool:
            if not isinstance(self.tools, list):  # type: ignore[has-type]
                self.tools = []
            current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)

            if not isinstance(current_date_tool, StructuredTool):
                msg = "CurrentDateComponent must be converted to a StructuredTool"
                raise TypeError(msg)
            self.tools.append(current_date_tool)

        # Set shared callbacks for tracing the tools used by the agent
        self.set_tools_callbacks(self.tools, self._get_shared_callbacks())

        return llm_model, self.chat_history, self.tools

    def _build_direct_tool_messages(self, input_text: str):
        messages = []
        system_prompt = getattr(self, "system_prompt", "") or ""
        if system_prompt.strip():
            messages.append(SystemMessage(content=system_prompt))

        if hasattr(self, "chat_history") and self.chat_history:
            if isinstance(self.chat_history, Data):
                messages.extend(self._data_to_messages_skip_empty([self.chat_history]))
            elif all(isinstance(message, Message) for message in self.chat_history):
                messages.extend(self._data_to_messages_skip_empty([message.to_data() for message in self.chat_history]))
            elif all(isinstance(message, Data) for message in self.chat_history):
                messages.extend(self._data_to_messages_skip_empty(self.chat_history))

        messages.append(HumanMessage(content=input_text.strip() or "Continue the conversation."))
        return messages

    async def _summarize_direct_tool_result(
        self,
        llm_model,
        input_text: str,
        tool_name: str | None,
        tool_args: dict,
        tool_result: str,
        summary_prompt: str,
        result_label: str,
    ) -> str:
        summary_messages = [
            SystemMessage(content=summary_prompt),
            HumanMessage(
                content=(
                    "User request:\n"
                    f"{input_text}\n\n"
                    "Tool name:\n"
                    f"{tool_name or ''}\n\n"
                    "Tool arguments:\n"
                    f"{json.dumps(tool_args, ensure_ascii=False)}\n\n"
                    f"{result_label}:\n"
                    f"{tool_result}"
                )
            ),
        ]
        summary = await llm_model.ainvoke(summary_messages, config={"callbacks": self.get_langchain_callbacks()})
        return _lc_message_text(summary).strip()

    async def _run_direct_tool_summary(self) -> Message:
        try:
            self._tool_call_recorder = ToolCallRecorder()
            callbacks = [self._tool_call_recorder, *self.get_langchain_callbacks()]
            self.shared_callbacks = callbacks
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            self.set_tools_callbacks(self.tools, callbacks)

            if not self.tools:
                msg = "Direct Tool Summary mode requires at least one tool."
                raise ValueError(msg)
            if not hasattr(llm_model, "bind_tools"):
                msg = "The selected language model does not support tool calling."
                raise NotImplementedError(msg)

            input_text = _agent_input_text(self.input_value)
            tool_selector = llm_model.bind_tools(self.tools or [])
            tool_selection = await tool_selector.ainvoke(
                self._build_direct_tool_messages(input_text),
                config={"callbacks": self.get_langchain_callbacks()},
            )
            tool_calls = getattr(tool_selection, "tool_calls", None) or []

            if not tool_calls:
                result = Message(text=_lc_message_text(tool_selection).strip())
                self._agent_result = result
                self.status = result
                return result

            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            tool_args = _parse_tool_args(tool_call.get("args"))
            selected_tool = next((tool for tool in self.tools if getattr(tool, "name", None) == tool_name), None)
            if selected_tool is None:
                msg = f"Tool '{tool_name}' was selected by the model but is not available."
                raise ValueError(msg)

            tool_output = await selected_tool.ainvoke(tool_args, config={"callbacks": callbacks})
            tool_output_text = _tool_output_text(tool_output)
            tool_error = _latest_tool_error(self._tool_call_recorder)

            if tool_error:
                error_text = tool_error.get("message") or tool_output_text or "Tool execution failed."
                failure_prompt = getattr(self, "tool_failure_summary_prompt", "") or ""
                try:
                    summary_text = await self._summarize_direct_tool_result(
                        llm_model,
                        input_text,
                        tool_name,
                        tool_args,
                        error_text,
                        failure_prompt,
                        "Tool error",
                    )
                except Exception as e:  # noqa: BLE001
                    await logger.aerror(f"Direct tool failure summary failed: {e!s}")
                    summary_text = f"Tool execution failed: {error_text}"
                result = Message(text=summary_text)
                self._agent_result = result
                self.status = result
                return result

            try:
                success_prompt = getattr(self, "tool_success_summary_prompt", "") or ""
                summary_text = await self._summarize_direct_tool_result(
                    llm_model,
                    input_text,
                    tool_name,
                    tool_args,
                    tool_output_text,
                    success_prompt,
                    "Tool result",
                )
            except Exception as e:  # noqa: BLE001
                await logger.aerror(f"Direct tool summary failed: {e!s}")
                summary_text = "The tool completed successfully, but the summary could not be generated."

            result = Message(text=summary_text)
            self._agent_result = result
            self.status = result
        except (ValueError, TypeError, KeyError) as e:
            await logger.aerror(f"{type(e).__name__}: {e!s}")
            raise
        except ExceptionWithMessageError as e:
            await logger.aerror(f"ExceptionWithMessageError occurred: {e}")
            raise
        except Exception as e:
            await logger.aerror(f"Unexpected error: {e!s}")
            raise
        else:
            return result

    async def _run_agent_loop(self) -> Message:
        try:
            self._tool_call_recorder = ToolCallRecorder()
            callbacks = [self._tool_call_recorder, *self.get_langchain_callbacks()]
            self.shared_callbacks = callbacks
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            self.set_tools_callbacks(self.tools, callbacks)
            # Set up and run agent
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )
            agent = self.create_agent_runnable()
            result = await self.run_agent(agent)

            # Store result for potential JSON output
            self._agent_result = result

        except (ValueError, TypeError, KeyError) as e:
            await logger.aerror(f"{type(e).__name__}: {e!s}")
            raise
        except ExceptionWithMessageError as e:
            await logger.aerror(f"ExceptionWithMessageError occurred: {e}")
            raise
        # Avoid catching blind Exception; let truly unexpected exceptions propagate
        except Exception as e:
            await logger.aerror(f"Unexpected error: {e!s}")
            raise
        else:
            return result

    async def _run_agent_once(self) -> Message:
        mode = getattr(self, "tool_result_mode", TOOL_RESULT_MODE_DIRECT_SUMMARY) or TOOL_RESULT_MODE_DIRECT_SUMMARY
        if mode == TOOL_RESULT_MODE_AGENT_LOOP:
            return await self._run_agent_loop()
        return await self._run_direct_tool_summary()

    async def _ensure_agent_result(self) -> Message:
        if hasattr(self, "_agent_result"):
            return self._agent_result
        return await self._run_agent_once()

    async def message_response(self) -> Message:
        return await self._ensure_agent_result()

    async def full_response(self) -> Data:
        result = await self._ensure_agent_result()
        recorder = getattr(self, "_tool_call_recorder", None)
        tool_calls = recorder.tool_calls if recorder else []
        request_variables = None
        if hasattr(self, "graph") and self.graph and hasattr(self.graph, "context"):
            request_variables = self.graph.context.get("request_variables")
        return Data(data=build_full_response(result, tool_calls, self.input_value, request_variables))

    def _preprocess_schema(self, schema):
        """Preprocess schema to ensure correct data types for build_model_from_schema."""
        processed_schema = []
        for field in schema:
            processed_field = {
                "name": str(field.get("name", "field")),
                "type": str(field.get("type", "str")),
                "description": str(field.get("description", "")),
                "multiple": field.get("multiple", False),
            }
            # Ensure multiple is handled correctly
            if isinstance(processed_field["multiple"], str):
                processed_field["multiple"] = processed_field["multiple"].lower() in [
                    "true",
                    "1",
                    "t",
                    "y",
                    "yes",
                ]
            processed_schema.append(processed_field)
        return processed_schema

    async def build_structured_output_base(self, content: str):
        """Build structured output with optional BaseModel validation."""
        json_pattern = r"\{.*\}"
        schema_error_msg = "Try setting an output schema"

        # Try to parse content as JSON first
        json_data = None
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(json_pattern, content, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"content": content, "error": schema_error_msg}
            else:
                return {"content": content, "error": schema_error_msg}

        # If no output schema provided, return parsed JSON without validation
        if not hasattr(self, "output_schema") or not self.output_schema or len(self.output_schema) == 0:
            return json_data

        # Use BaseModel validation with schema
        try:
            processed_schema = self._preprocess_schema(self.output_schema)
            output_model = build_model_from_schema(processed_schema)

            # Validate against the schema
            if isinstance(json_data, list):
                # Multiple objects
                validated_objects = []
                for item in json_data:
                    try:
                        validated_obj = output_model.model_validate(item)
                        validated_objects.append(validated_obj.model_dump())
                    except ValidationError as e:
                        await logger.aerror(f"Validation error for item: {e}")
                        # Include invalid items with error info
                        validated_objects.append({"data": item, "validation_error": str(e)})
                return validated_objects

            # Single object
            try:
                validated_obj = output_model.model_validate(json_data)
                return [validated_obj.model_dump()]  # Return as list for consistency
            except ValidationError as e:
                await logger.aerror(f"Validation error: {e}")
                return [{"data": json_data, "validation_error": str(e)}]

        except (TypeError, ValueError) as e:
            await logger.aerror(f"Error building structured output: {e}")
            # Fallback to parsed JSON without validation
            return json_data

    async def json_response(self) -> Data:
        """Convert agent response to structured JSON Data output with schema validation."""
        # Always use structured chat agent for JSON response mode for better JSON formatting
        try:
            system_components = []

            # 1. Agent Instructions (system_prompt)
            agent_instructions = getattr(self, "system_prompt", "") or ""
            if agent_instructions:
                system_components.append(f"{agent_instructions}")

            # 2. Format Instructions
            format_instructions = getattr(self, "format_instructions", "") or ""
            if format_instructions:
                system_components.append(f"Format instructions: {format_instructions}")

            # 3. Schema Information from BaseModel
            if hasattr(self, "output_schema") and self.output_schema and len(self.output_schema) > 0:
                try:
                    processed_schema = self._preprocess_schema(self.output_schema)
                    output_model = build_model_from_schema(processed_schema)
                    schema_dict = output_model.model_json_schema()
                    schema_info = (
                        "You are given some text that may include format instructions, "
                        "explanations, or other content alongside a JSON schema.\n\n"
                        "Your task:\n"
                        "- Extract only the JSON schema.\n"
                        "- Return it as valid JSON.\n"
                        "- Do not include format instructions, explanations, or extra text.\n\n"
                        "Input:\n"
                        f"{json.dumps(schema_dict, indent=2)}\n\n"
                        "Output (only JSON schema):"
                    )
                    system_components.append(schema_info)
                except (ValidationError, ValueError, TypeError, KeyError) as e:
                    await logger.aerror(f"Could not build schema for prompt: {e}", exc_info=True)

            # Combine all components
            combined_instructions = "\n\n".join(system_components) if system_components else ""
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=combined_instructions,
            )

            # Create and run structured chat agent
            try:
                structured_agent = self.create_agent_runnable()
            except (NotImplementedError, ValueError, TypeError) as e:
                await logger.aerror(f"Error with structured chat agent: {e}")
                raise
            try:
                result = await self.run_agent(structured_agent)
            except (
                ExceptionWithMessageError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:
                await logger.aerror(f"Error with structured agent result: {e}")
                raise
            # Extract content from structured agent result
            if hasattr(result, "content"):
                content = result.content
            elif hasattr(result, "text"):
                content = result.text
            else:
                content = str(result)

        except (
            ExceptionWithMessageError,
            ValueError,
            TypeError,
            NotImplementedError,
            AttributeError,
        ) as e:
            await logger.aerror(f"Error with structured chat agent: {e}")
            msg = f"Error with structured chat agent: {e}"
            raise RuntimeError(msg) from e

        # Process with structured output validation
        try:
            structured_output = await self.build_structured_output_base(content)

            # Handle different output formats
            if isinstance(structured_output, list) and structured_output:
                if len(structured_output) == 1:
                    return Data(data=structured_output[0])
                return Data(data={"results": structured_output})
            if isinstance(structured_output, dict):
                return Data(data=structured_output)
            return Data(data={"content": content})

        except (ValueError, TypeError) as e:
            await logger.aerror(f"Error in structured output processing: {e}")
            msg = f"Error in structured output processing: {e}"
            raise ValueError(msg) from e

    async def get_memory_data(self):
        # TODO: This is a temporary fix to avoid message duplication. We should develop a function for this.
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(
                session_id=self.graph.session_id,
                context_id=self.context_id,
                order="Ascending",
                n_messages=self.n_messages,
            )
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

    def update_input_types(self, build_config: dotdict) -> dotdict:
        """Update input types for all fields in build_config."""
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
        return build_config

    def update_direct_tool_prompt_visibility(
        self, build_config: dotdict, field_value, field_name: str | None
    ) -> dotdict:
        current_mode = (
            field_value
            if field_name == "tool_result_mode"
            else build_config.get("tool_result_mode", {}).get("value", TOOL_RESULT_MODE_DIRECT_SUMMARY)
        )
        if isinstance(current_mode, list) and current_mode:
            current_mode = current_mode[0]

        show_summary_prompts = current_mode == TOOL_RESULT_MODE_DIRECT_SUMMARY
        for prompt_field in ("tool_success_summary_prompt", "tool_failure_summary_prompt"):
            if prompt_field in build_config:
                build_config[prompt_field]["show"] = show_summary_prompts

        return build_config

    async def update_build_config(
        self,
        build_config: dotdict,
        field_value: list[dict],
        field_name: str | None = None,
    ) -> dotdict:
        # Update model options with caching (for all field changes)
        # Agents require tool calling, so filter for only tool-calling capable models
        def get_tool_calling_model_options(user_id=None):
            return get_language_model_options(user_id=user_id, tool_calling=True)

        build_config = update_model_options_in_build_config(
            component=self,
            build_config=dict(build_config),
            cache_key_prefix="language_model_options_tool_calling",
            get_options_func=get_tool_calling_model_options,
            field_name=field_name,
            field_value=field_value,
        )
        build_config = dotdict(build_config)
        build_config = self.update_direct_tool_prompt_visibility(build_config, field_value, field_name)

        if field_name == "model":
            build_config = self.update_input_types(build_config)

        current_model_value = field_value if field_name == "model" else build_config.get("model", {}).get("value")
        provider = ""
        if isinstance(current_model_value, list) and current_model_value:
            selected_model = current_model_value[0]
            provider = (selected_model.get("provider") or "").strip()
            if not provider and selected_model.get("name"):
                provider = get_provider_for_model_name(str(selected_model["name"]))

        if provider:
            build_config = apply_provider_variable_config_to_build_config(build_config, provider)

        if field_name == "model":
            default_keys = [
                "code",
                "_type",
                "model",
                "tools",
                "input_value",
                "add_current_date_tool",
                "system_prompt",
                "agent_description",
                "max_iterations",
                "handle_parsing_errors",
                "verbose",
            ]
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list[Tool]:
        component_toolkit = get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        # TODO: Agent Description Depreciated Feature to be removed
        description = f"{agent_description}{tools_names}"

        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_Agent",
            tool_description=description,
            # here we do not use the shared callbacks as we are exposing the agent as a tool
            callbacks=self.get_langchain_callbacks(),
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)

        return tools
