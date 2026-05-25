# Langflow Agent 工具完整返回执行方案

## 目标

后端通过 `/api/v2/workflows` 调用 Simple Agent 工作流后，必须拿到结构化完整结果。

结果必须包含：

```text
Agent 最终回答
Agent 选择的工具
工具调用入参
工具原始完整输出
目标工作流多个输出组件
错误信息
后端真正消费的 final_outputs
```

固定链路：

```text
MCP Tool 执行业务工作流
-> MCP Server 返回完整工具输出信封
-> Agent 记录 tool_calls
-> Agent 暴露 full_response
-> 后端读取结构化输出
```

## 相关文件

- 总链路方案：`langflow-agent-backend-e2e-plan.md`
- MCP 多输入参数方案：`langflow-mcp-multi-input-plan.md`
- 请求头透传方案：`langflow-agent-header-forwarding-plan.md`
- 后端调用模拟脚本：`temp_run_agent_workflow.py`

## 当前代码验证结论

当前代码还没有完整返回能力，缺口分成 MCP Server 和 Agent 两层。

MCP Server 当前问题：

```text
src/backend/base/langflow/api/v1/mcp_utils.py
  handle_call_tool()
```

当前执行目标工作流后，会遍历 `result.outputs`，把 `messages` 和 `results` 转成文本后追加到 `TextContent`。这会导致：

```text
1. 输出组件 ID 丢失。
2. 输出组件 display_name 丢失。
3. Data / Message 类型信息丢失。
4. 多个输出组件被压成多个文本片段或被去重。
5. 后端无法稳定读取 raw_output.outputs。
```

Agent 当前问题：

```text
src/lfx/src/lfx/components/models_and_agents/agent.py
  outputs 当前只有 Response。

src/lfx/src/lfx/base/agents/events.py
  process_agent_events() 能处理 on_tool_start / on_tool_end，
  但只更新 Message.content_blocks，不会产出后端可消费的 tool_calls。
```

可复用的现有能力：

```text
src/lfx/src/lfx/components/models_and_agents/agent.py
  message_response() 已把 agent 执行结果保存到 self._agent_result。

src/lfx/src/lfx/base/agents/callback.py
  AgentAsyncHandler 已能接收工具 start/end 事件。
```

因此本方案不需要让模型复述 trace，而是要在执行层记录工具调用，并新增结构化输出。

## 代码落地方案

### 1. MCP Server 返回完整输出信封

修改文件：

```text
src/backend/base/langflow/api/v1/mcp_utils.py
```

新增序列化函数：

```python
def build_mcp_tool_output_envelope(flow: Flow, run_response: RunResponse) -> dict:
    ...
```

输出结构固定为：

```json
{
  "flow_id": "target-flow-id",
  "flow_name": "目标业务工作流",
  "tool_name": "blockbuster_recreate",
  "success": true,
  "outputs": {
    "DataOutput-abc": {
      "display_name": "Data Output",
      "type": "data",
      "content": {}
    },
    "ChatOutput-def": {
      "display_name": "Chat Output",
      "type": "message",
      "content": "自然语言输出"
    }
  },
  "error": null
}
```

序列化要求：

```text
1. 外层保留 flow_id、flow_name、tool_name、success、error。
2. outputs 的 key 使用输出组件 ID。
3. 每个输出组件保留 display_name、type、content。
4. Message 使用 get_text() 或 message 字段写入 content。
5. Data / dict / list 保持结构化 content，不转成不可解析字符串。
6. 其他对象使用 Langflow 现有 serialize 工具转成 JSON 兼容结构。
```

`handle_call_tool()` 不再把结果逐条压成文本并去重，而是只返回一个 JSON 信封：

```python
return [
    types.TextContent(
        type="text",
        text=json.dumps(envelope, ensure_ascii=False),
    )
]
```

失败时也返回同样结构：

```json
{
  "flow_id": "target-flow-id",
  "flow_name": "目标业务工作流",
  "tool_name": "blockbuster_recreate",
  "success": false,
  "outputs": {},
  "error": {
    "code": "tool_execution_failed",
    "message": "目标工作流执行失败"
  }
}
```

### 2. Agent 增加工具调用记录器

修改文件：

```text
src/lfx/src/lfx/components/models_and_agents/agent.py
src/lfx/src/lfx/base/agents/callback.py
```

新增一个执行层 recorder，而不是让模型生成 trace：

```python
class ToolCallRecorder(AsyncCallbackHandler):
    def __init__(self):
        self.tool_calls = []

    async def on_tool_start(...):
        记录 tool_name、arguments、start_time

    async def on_tool_end(...):
        解析 output，写入 raw_output、elapsed_ms、success=true

    async def on_tool_error(...):
        写入 elapsed_ms、success=false、error
```

Agent 执行前创建 recorder，并写入现有 shared callbacks：

```python
self._tool_call_recorder = ToolCallRecorder()
callbacks = [
    self._tool_call_recorder,
    *self.get_langchain_callbacks(),
]
self.shared_callbacks = callbacks
```

这组 callbacks 必须挂到工具上，并由现有 `run_agent()` 自动复用：

```python
self.set_tools_callbacks(self.tools, callbacks)
result = await self.run_agent(agent)
```

唯一实现方式：不修改 `run_agent()` 方法签名。当前 `run_agent()` 内部会调用 `_get_shared_callbacks()` 组装 Agent 执行 callbacks，因此 recorder 必须通过 `self.shared_callbacks` 注入；同时调用 `set_tools_callbacks()`，保证工具执行也使用同一组 callbacks。

不要新增 `run_agent(agent, callbacks=...)` 这种调用方式；当前代码不支持该签名。

注意：`ToolCallRecorder` 必须保存原始工具入参和工具输出，不从 `Message.content_blocks` 反推。

### 3. Agent 新增 Full Response 输出且避免重复执行

修改文件：

```text
src/lfx/src/lfx/components/models_and_agents/agent.py
```

新增输出：

```python
outputs = [
    Output(name="response", display_name="Response", method="message_response", group_outputs=True),
    Output(name="full_response", display_name="Full Response", method="full_response", group_outputs=True),
]
```

`Response` 和 `Full Response` 必须同时作为独立端点输出，不再通过下拉框二选一。
其中 `Response` 保持 Message 类型，用于用户友好展示；`Full Response` 保持 Data/JSON 类型，用于后端结构化消费。
为了避免 Agent 作为 Tool 暴露时出现“只能有一个 tool output”的兼容问题，`Response` 可以保留 `tool_mode=True`，`Full Response` 不作为 tool output 暴露。

新增统一执行函数：

```python
async def _ensure_agent_result(self) -> Message:
    if hasattr(self, "_agent_result"):
        return self._agent_result
    return await self.message_response()
```

`message_response()` 和 `full_response()` 必须共享同一次 Agent 执行结果，不能因为有两个输出就重复调用工具。

`full_response()` 返回 `Data`：

```python
async def full_response(self) -> Data:
    result = await self._ensure_agent_result()
    tool_calls = self._tool_call_recorder.tool_calls
    return Data(data=build_full_response(result, tool_calls, self.input_value))
```

### 4. final_outputs 由执行层提取

`build_full_response()` 固定执行：

```text
1. 从 Agent 输入 JSON 中读取 request_id。
2. 从 Agent 最终 Message 中读取 answer。
3. 从 recorder 中读取 tool_calls。
4. selected_tool 使用第一个成功工具调用，失败时使用第一个调用过的工具。
5. 从 tool_calls[].raw_output.outputs 中优先提取 Data 类型输出。
6. 如果只有一个 Data 输出，展开为 final_outputs。
7. 如果有多个 Data 输出，用组件 ID 作为 key 保留。
8. 所有失败写入 errors。
```

这样后端读取的是确定结构，不依赖模型自由发挥。

## 唯一执行原则

完整返回能力由两层共同保证：

```text
MCP Server 层
  保证工具工作流原始输出完整。

Agent 层
  保证工具调用记录和原始输出被暴露给后端。
```

不依赖模型在最终自然语言回答中复述工具结果。

不把 MCP Tools 节点作为最终输出节点。

最终由 Agent 节点新增结构化输出：

```text
Full Response
```

现有：

```text
Response
```

继续保留，用于 UI 对话展示。

## Simple Agent 工作流输出结构

工作流连接方式固定为：

```text
Agent.response
  -> ChatOutput.input_value

Agent.full_response
  -> DataOutput / 结构化输出节点
```

后端读取结构化输出节点，不再只依赖 `ChatOutput` 的自然语言内容。

通过 `/api/v2/workflows` 读取结果时，`outputs` 的 key 固定使用结构化输出组件 ID，不使用组件显示名。后端和 `temp_run_agent_workflow.py` 都必须通过该组件 ID 读取 `Agent.full_response` 的落点；不要把 `Full Response` 这类 display_name 当作稳定 key。

## Agent 工具结果处理模式

Agent 需要支持两种工具结果处理模式：

```text
Direct Tool Summary
Agent Loop
```

`Agent Loop` 保留原有 LangChain Agent 循环，不改变底层循环逻辑。
该模式适合模型供应商完整支持工具调用循环、消息历史和 reasoning 相关字段的场景。

`Direct Tool Summary` 是新增默认模式，用于规避部分 NewAPI 兼容层在 Agent 循环中不支持 `reasoning_content` 等字段导致的失败。
该模式固定执行三步：

```text
1. 第一次模型调用绑定 MCP tools，只负责选择工具和生成工具入参。
2. 执行被选中的 MCP tool 一次，并由 ToolCallRecorder 记录 raw_output。
3. 第二次独立模型调用只接收用户原始请求、工具名、工具入参、工具原始返回或工具错误，以及对应的总结提示词，生成用户友好的 Response。
```

第二次总结调用不能再绑定 tools，不能继续进入 Agent 循环，也不能携带 agent_scratchpad / reasoning_content / full chat history。
它只发送：

```text
SystemMessage(content=tool_success_summary_prompt 或 tool_failure_summary_prompt)
HumanMessage(content="User request:\n{input_text}\n\nTool name:\n{tool_name}\n\nTool arguments:\n{tool_args}\n\nTool result/error:\n{tool_output_or_error}")
```

第二次总结调用使用同一个 Agent 配置出的 `llm_model` 和 callbacks，仍然走同一个模型供应商配置；差异只在输入消息更小、没有 tools、没有 agent loop。

两种模式都必须生成同一套 `Full Response` 结构。
`Direct Tool Summary` 中，`Full Response` 直接来自工具原始返回和 recorder；`Response` 来自第二次模型总结。工具成功时使用 `tool_success_summary_prompt`，工具失败时使用 `tool_failure_summary_prompt`。
`Agent Loop` 中，`Full Response` 仍按原有执行层 recorder 生成。

`tool_success_summary_prompt` 与 `tool_failure_summary_prompt` 只在 `Direct Tool Summary` 模式下显示；切回 `Agent Loop` 时这两个输入项必须隐藏。不要保留旧字段 `tool_summary_prompt` 的兼容逻辑。

这两个字段的默认值使用中文提示词，分别面向“工具成功后的用户友好总结”和“工具失败后的可操作错误说明”：

```text
tool_success_summary_prompt 默认值：
你是一个面向终端用户的结果总结助手。

工具已经执行成功。请根据用户本次请求、工具名称、工具入参和工具返回结果，生成一段清晰、准确、用户友好的中文回复。

要求：
1. 只总结工具返回结果中真实存在的信息，不要编造。
2. 不要暴露内部字段名、组件 ID、tool_calls、raw_output、JSON 信封等技术细节。
3. 如果工具返回了多个输出，优先总结与用户需求最相关的内容。
4. 如果工具结果本身已经是完整文本，可以在保持原意的基础上适度整理表达。
5. 回复要简洁，但不能遗漏关键结论。

tool_failure_summary_prompt 默认值：
你是一个面向终端用户的错误说明助手。

工具调用失败了。请根据用户本次请求、工具名称、工具入参和工具错误信息，生成一段清晰、准确、可操作的中文回复。

要求：
1. 明确说明本次没有成功完成工具执行，不要说“执行成功”。
2. 如果错误来自缺少必填参数、参数为空或参数格式不正确，请明确指出需要补充或修改的参数。
3. 不要暴露内部字段名、组件 ID、tool_calls、raw_output、JSON 信封等技术细节，除非参数名本身就是用户需要填写的业务参数。
4. 不要编造工具没有返回的信息。
5. 回复要给出下一步操作建议，让用户知道应该如何修正后重试。
```

`n_messages` 不再放在高级配置中，默认值固定为 `0`，避免默认携带大量历史消息进入 Agent 调用。该字段允许配置任意非负整数，例如 `0`、`1`、`20`、`50`。

## MCP Server 工具输出信封

当 Project MCP Server 执行某个工作流工具时，必须收集目标工作流的完整输出组件。

MCP Tool 返回给 Agent 的内容固定为 JSON 信封：

```json
{
  "flow_id": "target-flow-id",
  "flow_name": "目标业务工作流",
  "tool_name": "blockbuster_recreate",
  "outputs": {
    "ChatOutput-xxx": {
      "display_name": "Chat Output",
      "type": "message",
      "content": "自然语言输出"
    },
    "DataOutput-yyy": {
      "display_name": "Data Output",
      "type": "data",
      "content": {
        "strategy": {},
        "visual_spec": {},
        "content_spec": {}
      }
    }
  },
  "success": true,
  "error": null
}
```

要求：

- 保留目标工作流 `flow_id`。
- 保留目标工作流 `flow_name`。
- 保留每个输出组件 ID。
- 保留每个输出组件 `display_name`。
- 保留每个输出组件类型。
- 保留每个输出组件原始结构化内容。
- 如果输出是 `Data`，不得压成不可解析的普通文本。
- 如果输出是 `Message`，保留文本内容。

## MCP Server 修改位置

目标文件：

```text
src/backend/base/langflow/api/v1/mcp_utils.py
```

需要在工具执行逻辑中完成：

```text
handle_call_tool
-> simple_run_flow
-> 收集 result.outputs
-> 构造完整 JSON 信封
-> 作为 MCP TextContent 返回给 Agent
```

MCP 标准返回通常是 content list，因此 JSON 信封可以作为 `TextContent.text` 承载：

```json
{
  "type": "text",
  "text": "{\"flow_id\":\"...\",\"outputs\":{...}}"
}
```

Agent 侧收到后再解析为结构化 `raw_output`。

## Agent 工具调用记录

Agent 执行过程中必须由执行层记录工具调用，而不是让模型自己总结。

每次工具调用记录固定结构：

```json
{
  "tool_name": "blockbuster_recreate",
  "arguments": {
    "hot_video_info": "爆款视频信息",
    "business_info": "业务信息"
  },
  "raw_output": {
    "flow_id": "target-flow-id",
    "flow_name": "目标业务工作流",
    "outputs": {}
  },
  "success": true,
  "elapsed_ms": 12000,
  "error": null
}
```

记录时机：

```text
工具调用开始
  记录 tool_name、arguments、start_time

工具调用成功
  记录 raw_output、elapsed_ms、success=true

工具调用失败
  记录 elapsed_ms、success=false、error
```

## Agent Full Response 输出结构

Agent 节点新增输出：

```text
Full Response
```

类型建议：

```text
Data
```

固定结构：

```json
{
  "request_id": "backend-task-001",
  "answer": "已完成复刻执行方案生成。",
  "selected_tool": "blockbuster_recreate",
  "tool_calls": [
    {
      "tool_name": "blockbuster_recreate",
      "arguments": {
        "hot_video_info": "爆款视频信息",
        "business_info": "业务信息"
      },
      "raw_output": {
        "flow_id": "target-flow-id",
        "flow_name": "目标业务工作流",
        "outputs": {}
      },
      "success": true,
      "elapsed_ms": 12000,
      "error": null
    }
  ],
  "final_outputs": {
    "strategy": {},
    "visual_spec": {},
    "content_spec": {},
    "output_format": {},
    "constraints": {}
  },
  "errors": []
}
```

字段说明：

```text
request_id
  与后端输入任务 JSON 中的 request_id 保持一致。

answer
  Agent 的最终自然语言摘要。

selected_tool
  本次实际选中的工具名。

tool_calls
  所有工具调用记录，按调用顺序排列。

final_outputs
  从工具 raw_output 中提取出的后端业务结果。

errors
  结构化错误列表。
```

## final_outputs 提取规则

`final_outputs` 由执行层从 `tool_calls[].raw_output.outputs` 中提取。

固定规则：

```text
1. 优先读取 Data 类型输出组件。
2. 如果存在多个 Data 输出，保留为对象，key 为组件 ID。
3. 如果只有一个主要 Data 输出，允许展开为业务结构。
4. ChatOutput 只作为 answer 候选，不作为 final_outputs 的主要来源。
5. 提取失败时，final_outputs 为空对象，并在 errors 中写入原因。
```

示例：

工具 raw_output：

```json
{
  "outputs": {
    "DataOutput-abc": {
      "type": "data",
      "content": {
        "strategy": {},
        "visual_spec": {}
      }
    }
  }
}
```

Agent full_response：

```json
{
  "final_outputs": {
    "strategy": {},
    "visual_spec": {}
  }
}
```

## 错误结构

工具调用失败时，`tool_calls` 和 `errors` 必须同时记录。

示例：

```json
{
  "request_id": "backend-task-001",
  "answer": "工具调用失败，未能完成任务。",
  "selected_tool": "weather",
  "tool_calls": [
    {
      "tool_name": "weather",
      "arguments": {
        "city": "厦门"
      },
      "raw_output": null,
      "success": false,
      "elapsed_ms": 1200,
      "error": {
        "code": "tool_execution_failed",
        "message": "目标工作流执行失败"
      }
    }
  ],
  "final_outputs": {},
  "errors": [
    {
      "code": "tool_execution_failed",
      "message": "目标工作流执行失败",
      "tool_name": "weather"
    }
  ]
}
```

## Agent 节点修改位置

目标文件范围：

```text
src/lfx/src/lfx/components/models_and_agents/agent.py
```

执行要求：

```text
1. 保留现有 Response 输出，并让它继续作为 Message 端点。
2. 新增 Full Response 输出，并让它作为 Data/JSON 端点与 Response 同时输出。
3. 在工具调用 wrapper / callback 中记录 tool_calls。
4. 将最终模型回答或 Direct Tool Summary 的总结结果写入 answer。
5. 将工具调用记录写入 tool_calls。
6. 从 raw_output 提取 final_outputs。
7. 将失败信息写入 errors。
8. 新增 tool_result_mode，支持 Direct Tool Summary 与 Agent Loop 双模式。
9. `tool_success_summary_prompt` 和 `tool_failure_summary_prompt` 只在 Direct Tool Summary 下展示。
10. n_messages 取消高级配置、默认为 0，并允许任意非负整数。
```

`Full Response` 的生成不能依赖模型自由发挥。

模型可以生成 `answer`，但 `tool_calls`、`raw_output`、`elapsed_ms`、`success`、`error` 必须由执行层记录。

## 实施步骤

1. 修改 `mcp_utils.py`，让 Project MCP Server 返回目标工作流完整输出 JSON 信封。
2. 新增 MCP 工具失败时的统一 JSON 错误信封。
3. 新增 Agent 工具调用 recorder，记录工具名、入参、耗时、成功状态、原始输出和错误。
4. 修改 Agent 节点，新增 `Full Response` 输出。
5. 将 `Response` 与 `Full Response` 改为同时输出的 group outputs，不再二选一。
6. 抽出 `_ensure_agent_result()`，保证 `Response` 和 `Full Response` 不会重复执行 Agent。
7. 解析 MCP Tool 返回的 JSON 信封，写入 `tool_calls[].raw_output`。
8. 从 `raw_output.outputs` 提取 `final_outputs`。
9. 新增 `tool_result_mode`，默认 `Direct Tool Summary`，并保留 `Agent Loop` 原逻辑。
10. 新增 `tool_success_summary_prompt` / `tool_failure_summary_prompt` 可见性联动，只在 `Direct Tool Summary` 模式展示，不做旧字段兼容。
11. 将 `n_messages` 取消高级配置并默认设为 0，同时修复前端整数输入控件对 `n_messages` 的最大值限制。
12. 连接 `Agent.full_response` 到结构化输出节点。
13. 更新 `temp_run_agent_workflow.py` 的结果读取说明。
14. 增加 MCP tool 多输出、Agent 双输出、工具失败、Direct Tool Summary 四类测试。

## 验收标准

- 后端调用 Simple Agent 后可以读取结构化 `Full Response`。
- `Response` 旧输出仍可用于 UI 对话展示。
- `tool_calls` 中包含工具名、入参、耗时、成功状态、错误信息。
- `raw_output` 包含目标工作流所有输出组件。
- 多个输出组件不会被压缩成单一自然语言文本。
- Data 输出能作为结构化对象被后端消费。
- 工具失败时后端能拿到结构化错误。
- 模型没有复述 trace 时，后端仍能拿到完整工具调用记录。
- `Response` 和 `Full Response` 在同一次运行中同时存在，且不会导致工具重复执行。
- `Direct Tool Summary` 模式下，第二次模型调用只做总结，不再进入工具循环。
- 切换到 `Agent Loop` 时，原有 Agent 循环行为保持不变。

## 2026-05-25 实现同步补充

### 工具错误识别与失败总结

对应文件：

```text
src/lfx/src/lfx/base/agents/callback.py
src/lfx/src/lfx/components/models_and_agents/agent.py
```

实现要求：

```text
1. `ToolCallRecorder` 不能只看 callback 是否抛异常，还必须识别工具返回值中的失败信封。
2. 当 raw_output 是 dict 且 `success=false` 或 `error` 非空时，`tool_calls[].success=false`。
3. 当 raw_output 是字符串且以 `Input validation error:`、`Invalid input:`、`Tool execution failed:` 开头时，按工具失败处理，错误码为 `tool_input_validation_error`。
4. `Direct Tool Summary` 模式下，工具失败后仍允许第二次模型调用生成用户友好的失败说明。
5. 失败总结使用用户填写的 `tool_failure_summary_prompt`；成功总结使用 `tool_success_summary_prompt`。
6. 第二次总结失败时才允许返回最小兜底文案；正常路径不做内部提示词拼接。
```

### `n_messages` 与历史消息边界

对应文件：

```text
src/lfx/src/lfx/components/models_and_agents/agent.py
src/frontend/src/components/core/parameterRenderComponent/components/intComponent/index.tsx
```

实现要求：

```text
1. `n_messages=0` 时不向第一次模型调用传入历史消息。
2. `n_messages>0` 时，从同一 `session_id` / `context_id` 读取最近历史消息，并转换为 HumanMessage / AIMessage 传给第一次模型调用。
3. Direct Tool Summary 的第二次总结模型调用不携带完整历史消息，只携带用户当前请求、工具名、工具入参、工具结果或错误。
4. 前端整数输入控件对 `n_messages` 不设置最大值，允许配置 `0`、`1`、`20`、`50` 等非负整数。
```
