# Langflow Agent MCP 代理工作流实施任务书

## 目标

把当前“后端直接调用单条业务工作流”的方式，升级为：

```text
后端
-> /api/v2/workflows
-> Simple Agent 工作流
-> MCP Tools
-> Project MCP Server
-> 业务工作流工具
-> Agent Full Response
-> 后端结构化消费
```

后端长期只维护：

```text
1. Simple Agent 工作流 ID
2. Simple Agent 输入组件 ID
3. Simple Agent 结构化输出组件 ID
```

后端不再维护每条业务工作流的 flow_id、输入组件 ID、输出组件 ID、跨环境 ID 映射。

## 参考文档

实施时必须同时参考以下文档，但以本文的执行顺序和禁止事项为准：

```text
langflow-agent-backend-e2e-plan.md
langflow-mcp-multi-input-plan.md
langflow-agent-header-forwarding-plan.md
langflow-agent-tool-full-response-plan.md
temp_run_agent_workflow.py
```

## 当前代码前提

当前代码已经具备：

```text
1. /api/v2/workflows 可以把平铺 inputs 转为 tweaks。
2. /api/v2/workflows 可以提取 X-LANGFLOW-GLOBAL-VAR-* 请求头到 graph.context.request_variables。
3. simple_run_flow() 支持 input_request.tweaks。
4. Project MCP Server 可以从请求头提取 X-LANGFLOW-GLOBAL-VAR-*。
```

当前代码不具备：

```text
1. MCP 工具多输入参数配置。
2. MCP call_tool 按组件 ID 注入多个输入节点。
3. MCP tool 完整原始输出信封。
4. Agent Full Response 结构化输出。
5. MCP Tools 对内部 Project MCP Server 自动透传 USER-WALLET-ID / TASK-ID。
6. 涉及请求级 header 时绕过 MCPSessionManager persistent session cache。
```

## 实施顺序

必须按以下顺序实施，不要跳步。

### 阶段 1：MCP 工具多输入参数配置

目标：让每个 Project MCP Server 工具拥有稳定的语义化 `inputSchema`，并能映射到目标工作流多个输入节点。

必须修改：

```text
src/backend/base/langflow/services/database/models/flow/model.py
src/backend/base/langflow/alembic/versions/<new_revision>.py
src/backend/base/langflow/api/v1/schemas/__init__.py
src/backend/base/langflow/api/v1/mcp_projects.py
src/backend/base/langflow/helpers/flow.py
src/backend/base/langflow/api/v1/mcp_utils.py
src/frontend/src/types/mcp/index.ts
src/frontend/src/pages/MainPage/pages/homePage/hooks/useMcpServer.ts
src/frontend/src/pages/MainPage/pages/homePage/utils/mcpServerUtils.tsx
src/frontend/src/pages/MainPage/pages/homePage/components/McpFlowsSection.tsx
src/frontend/src/modals/toolsModal/components/toolsTable/index.tsx
```

必须实现：

```text
1. Flow 增加 mcp_input_parameters JSON 字段，并创建 Alembic 迁移。
2. MCPSettings / MCPProjectUpdateRequest 增加 mcp_input_parameters。
3. update_project_mcp_settings() 保存 mcp_input_parameters。
4. Edit Tools 右侧面板增加 Input parameters 配置区。
5. 首次配置时扫描 ChatInput / TextInput 生成默认参数配置。
6. json_schema_from_flow() 优先用 mcp_input_parameters 生成 inputSchema。
7. handle_call_tool() 把 arguments 转成 SimplifiedAPIRequest.tweaks。
8. required 缺失时返回结构化错误，不执行目标工作流。
9. unknown parameter 出现时返回结构化错误，不执行目标工作流。
```

唯一执行规则：

```text
所有输入节点都通过 mcp_input_parameters 映射。
即使是单输入工作流，也走同一套参数配置映射。
不要继续长期依赖 arguments["input_value"]。
```

### 阶段 2：MCP Tools 自动透传关键请求头

目标：后端传给 Simple Agent 工作流的 `USER-WALLET-ID` / `TASK-ID`，能继续到达被代理业务工作流。

必须修改：

```text
src/lfx/src/lfx/components/models_and_agents/mcp_component.py
src/lfx/src/lfx/base/mcp/util.py
```

必须实现：

```text
1. 新增 is_internal_project_mcp_server(server_config)。
2. 新增 normalize_headers(headers)，只做 list/dict 格式归一化和 header name 小写化，不解析 request_variables。
3. 删除/判断 USER-WALLET-ID / TASK-ID header 时必须大小写不敏感。
4. MCP Tools 连接内部 Project MCP Server 时：
   - 当前 request_variables 有真实 USER-WALLET-ID，才写入 X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID。
   - 当前 request_variables 有真实 TASK-ID，才写入 X-LANGFLOW-GLOBAL-VAR-TASK-ID。
   - 当前 request_variables 没有真实值时，移除同名手填 header。
5. 最终 headers 涉及 USER-WALLET-ID / TASK-ID 时：
   - 禁用 MCPToolsComponent 整包 cache。
   - 通过 disable_session_cache=True 绕过 MCPSessionManager persistent session cache。
   - 新建一次性 MCP 连接，并在本次工具加载结束后关闭/清理。
6. disable_session_cache 必须从 MCPToolsComponent.update_tool_list() 传到 update_tools()，再传到 MCPStdioClient / MCPStreamableHttpClient。
7. disable_session_cache=True 时，MCP client 不得调用 MCPSessionManager.get_session()。
```

唯一执行规则：

```text
只要最终 headers 涉及：
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID
X-LANGFLOW-GLOBAL-VAR-TASK-ID

就不得复用：
tools
tool_names
tool_cache
config
MCP client session
MCPSessionManager persistent session
```

第一版不做 schema-only cache。

### 阶段 3：MCP Server 返回完整工具输出信封

目标：Project MCP Server 执行业务工作流后，返回完整结构化原始输出，不再压成普通文本。

必须修改：

```text
src/backend/base/langflow/api/v1/mcp_utils.py
```

必须实现：

```text
1. 新增 build_mcp_tool_output_envelope(flow, run_response)。
2. 收集目标工作流所有输出组件。
3. outputs key 使用输出组件 ID。
4. 每个输出保留 display_name、type、content。
5. Data / dict / list 保持结构化 content。
6. Message 使用文本 content。
7. 失败也返回统一 JSON 错误信封。
8. MCP Tool 返回一个 TextContent，text 是 JSON 信封字符串。
```

唯一执行规则：

```text
不得继续把多个输出组件逐条转字符串并去重。
不得丢失输出组件 ID。
不得把 Data 输出压成不可解析文本。
```

### 阶段 4：Agent 新增 Full Response

目标：后端调用 Simple Agent 后，能稳定读取工具调用记录、原始输出和业务结构化结果。

必须修改：

```text
src/lfx/src/lfx/components/models_and_agents/agent.py
src/lfx/src/lfx/base/agents/callback.py
```

必须实现：

```text
1. 新增 ToolCallRecorder。
2. recorder 记录 tool_name、arguments、start_time、elapsed_ms、raw_output、success、error。
3. recorder 通过 self.shared_callbacks 注入，不修改 run_agent() 方法签名。
4. 同一组 callbacks 必须同时传给 set_tools_callbacks(self.tools, callbacks)。
5. Agent outputs 增加 Full Response。
6. 新增 _ensure_agent_result()，避免 Response 和 Full Response 重复执行 Agent。
7. full_response 返回 Data。
8. build_full_response() 从 recorder 读取 tool_calls。
9. 从 tool_calls[].raw_output.outputs 提取 final_outputs。
```

唯一执行规则：

```text
不要新增 run_agent(agent, callbacks=...)。
当前 run_agent() 不支持 callbacks 参数。

tool_calls / raw_output / elapsed_ms / success / error 必须由执行层记录。
不要依赖模型在自然语言中复述。
```

### 阶段 5：Simple Agent 工作流与后端调用协议

目标：后端通过固定 JSON 任务协议调用 Simple Agent。

Agent 输入必须是结构化 JSON 字符串，但只保留 Agent 选工具和填参真正需要的信息：

```json
{
  "requirement": "本次要完成的需求",
  "parameters": {}
}
```

唯一执行规则：

```text
Agent 根据 requirement、parameters 和 MCP Tool 的 name/description/inputSchema 选择最匹配工具。
不要把 request_id、output_requirement 等后端消费字段放入 Agent 输入。
任务关联使用请求头 TASK-ID；Full Response 由 Agent 执行层生成。
工具参数必须严格遵守 inputSchema；缺少必填参数时不调用工具。
```

需要同步：

```text
1. Simple Agent instructions。
2. temp_run_agent_workflow.py 的调用示例和结果读取逻辑。
```

## 追加需求同步

以下内容是在初始五阶段方案落地过程中追加确认的实现要求，不改变前面阶段顺序，只补充到对应阶段内执行。

### 阶段 1 补充：Edit Tools 参数默认值与同步边界

```text
1. Input parameters 卡片顶部标题只展示工作流输入节点 display_name，并随工作流节点名更新。
2. 参数名输入框不再从工作流节点名或 description 自动生成，默认使用英文占位名 default_parameter / default_parameter_2。
3. 参数描述输入框不再从工作流节点 description 自动继承，默认使用中文引导文案。
4. 用户在 Edit Tools 保存后的 parameter_name、parameter_description、required 以保存值为准。
5. Required 有实际意义，会写入 MCP inputSchema.required，并参与 call_tool 缺失参数校验。
6. parameter_name 必须非空；清空后前端归一化会过滤该参数项。
```

详细规则见 `langflow-mcp-multi-input-plan.md`。

### 阶段 2 补充：MCP Tools 长耗时工具配置

后续业务工具可能连续调用视频生成等长耗时能力，单次工具执行可能持续几十分钟。
因此 MCP Tools 组件必须提供非高级配置项：

```text
tool_call_timeout_minutes
  单位分钟，UI 中直接配置分钟数，代码内部转换为秒传给 MCP client run_tool。
  默认 30 分钟。

tool_call_max_retries
  最大重试次数，默认 0。
  该配置属于 MCP Tools 组件对工具调用的控制，不是 Agent 自身的重试循环。
```

底层只修改 Langflow 内部 MCP client 的 `run_tool` 调用超时和重试参数。
外部第三方 Agent 一般使用自己的 MCP client，直接调用 Langflow Project MCP Server 时不受这个配置影响。
这两个工具调用选项需要进入 MCP Tools cache key；当 timeout/retry 配置变化时，不能复用旧工具配置。

### 阶段 2 补充：Trace Task ID 关联

为观察 Simple Agent 调用业务工作流的完整链路，所有工作流 trace 需要保存请求级 `TASK-ID`。

```text
1. trace 表增加 task_id 字段和索引。
2. tracing service / native tracer 从 graph.context.request_variables["TASK-ID"] 读取并写入 trace。
3. traces API 返回 task_id，并支持按 task_id 过滤和搜索。
4. Traces 页面在 Trace ID 后新增 Task ID 列。
```

这样同一次请求中 Simple Agent 工作流和被代理业务工作流的 trace 会展示同一个 `Task ID`。
详细规则见 `langflow-agent-header-forwarding-plan.md`。

### 阶段 4 补充：Agent 双模式与双输出端点

Agent 必须同时输出：

```text
Response      Message，用于用户友好展示
Full Response Data/JSON，用于后端结构化消费
```

这两个输出是同时存在的端点，不再二选一。

Agent 工具结果处理新增双模式：

```text
Direct Tool Summary
  默认模式。
  第一次模型调用选择工具并生成入参。
  执行工具一次。
  第二次独立模型调用只根据用户请求和工具原返回生成简短总结。
  Full Response 直接使用工具原始返回和 recorder 生成。

Agent Loop
  保留原有 Agent 循环，不修改底层循环。
```

`tool_success_summary_prompt` 和 `tool_failure_summary_prompt` 只在 `Direct Tool Summary` 下显示；切换到 `Agent Loop` 时隐藏。`n_messages` 取消高级配置、默认设为 0，并允许任意非负整数。
详细规则见 `langflow-agent-tool-full-response-plan.md`。

### 阶段 5 补充：temp_run_agent_workflow.py 验证输出

`temp_run_agent_workflow.py` 除了模拟后端调用，还必须打印：

```text
1. 完整工作流响应。
2. 完整 workflow outputs。
3. Agent Response 输出组件的完整结果和解析结果。
4. Agent Full Response 输出组件的完整结果和解析结果。
```

脚本读取输出时必须按组件 ID 读取，不按 display_name 读取。
脚本注释使用中文，并说明每段逻辑在后端链路中的作用。

## 禁止事项

不要做以下实现：

```text
1. 不要只设置 MCPToolsComponent.use_cache=False 就认为已经禁用 session cache。
2. 不要复用 MCPSessionManager 中可能带旧 headers 的 session。
3. 不要实现 run_agent(agent, callbacks=...)。
4. 不要让模型自由生成 tool_calls / raw_output。
5. 不要继续只读取 arguments["input_value"]。
6. 不要把多个输出组件压成字符串数组。
7. 不要把 Data 输出转成不可解析文本。
8. 不要对外部 MCP Server 自动透传内部 USER-WALLET-ID / TASK-ID。
9. 不要在没有真实 request_variables 时向内部 Project MCP Server 发送 USER-WALLET-ID / TASK-ID 占位字符串。
10. 不要在第一版实现 schema-only cache。
```

## 验收用例

必须至少覆盖以下用例：

```text
1. 单输入业务工作流仍可作为 MCP Tool 调用。
2. 两个 TextInput 能分别收到不同参数。
3. TextInput + ChatInput 混合输入能分别注入。
4. 缺少 required 参数时不执行业务工作流，并返回结构化错误。
5. 出现 unknown parameter 时不执行业务工作流，并返回结构化错误。
6. 后端传 USER-WALLET-ID / TASK-ID 时，工具工作流能读取到同一组值。
7. UI 手动执行 Simple Agent 且无 request_variables 时，不发送 USER-WALLET-ID / TASK-ID。
8. 第一次请求带 headers、第二次请求不带 headers 时，第二次不会复用第一次的 headers/config/session。
9. 连续两个不同用户/任务请求不会串 USER-WALLET-ID / TASK-ID。
10. 业务工作流有多个输出组件时，raw_output.outputs 保留所有组件 ID 和内容。
11. Data 输出能进入 final_outputs。
12. 工具失败时，tool_calls[].error 和 errors 都有结构化错误。
13. Response 和 Full Response 同时存在时，Agent 不重复调用工具。
14. temp_run_agent_workflow.py 可以模拟后端调用 Simple Agent，并读取结构化 Full Response。
15. MCP Tools 的 tool_call_timeout_minutes 可以支持几十分钟级长耗时工具。
16. tool_call_max_retries 默认 0，且只影响 Langflow 内部 MCP client 的工具调用。
17. Direct Tool Summary 模式下，工具只执行一次，第二次模型调用只做用户友好总结。
18. Traces 页面可以通过 Task ID 关联 Simple Agent trace 和业务工具 trace。
19. Edit Tools 中工作流输入节点展示名会跟随工作流更新，但参数名和参数描述保持用户保存值。
```

## 最终交付标准

完成后必须满足：

```text
1. 后端只需要调用 Simple Agent 工作流。
2. 后端不需要知道被代理业务工作流 ID。
3. 后端不需要知道被代理业务工作流组件 ID。
4. 后端可以读取 request_id、answer、selected_tool、tool_calls、final_outputs、errors。
5. 请求头 USER-WALLET-ID / TASK-ID 可以全链路到达业务工具工作流。
6. 多输入、多输出、工具失败都能结构化处理。
7. Response 和 Full Response 可以同时被后端读取。
8. Task ID 可以在所有工作流 trace 中展示并用于链路关联。
```

## 2026-05-25 最终实现同步清单

以下条目是初始五阶段之外追加落地的实现，必须作为本方案的一部分维护。

### MCP 参数与校验

```text
src/backend/base/langflow/helpers/flow.py
  - required string 参数在 inputSchema 中增加 `minLength: 1`。
  - 默认参数名固定为 `default_parameter` 系列。
  - 默认参数描述固定为中文引导文案。
  - 工作流节点改名只同步 `component_display_name`。

src/backend/base/langflow/api/v1/mcp_utils.py
  - `build_tweaks_from_mcp_arguments()` 增加服务端硬校验。
  - required 缺失 -> `missing_required_parameter`。
  - required string 为空 -> `empty_required_parameter`。
  - schema 外参数 -> `unknown_parameter`。
  - 非 string 参数 -> `invalid_parameter_type`。
```

### MCP Tools 长耗时与缓存

```text
src/lfx/src/lfx/components/models_and_agents/mcp_component.py
src/lfx/src/lfx/base/mcp/util.py
  - MCP Tools 新增 `tool_call_timeout_minutes`，单位为分钟，默认 30。
  - MCP Tools 新增 `tool_call_max_retries`，默认 0。
  - 底层 MCP client `run_tool()` 接收 timeout/retry 配置。
  - timeout/retry 进入 MCP tools cache key，配置变化时不复用旧工具配置。
```

### Agent 双模式与用户友好失败总结

```text
src/lfx/src/lfx/base/agents/callback.py
  - ToolCallRecorder 识别 JSON 错误信封和 `Input validation error:` 等字符串错误。
  - 工具返回校验错误时 `tool_calls[].success=false`。

src/lfx/src/lfx/components/models_and_agents/agent.py
  - `Direct Tool Summary` 与 `Agent Loop` 双模式共存。
  - `Agent Instructions` 默认值固定为严谨工作流调度提示词，要求只按 requirement、parameters、工具名称、描述、inputSchema 和同 session 最近历史用户请求选择工具。
  - required 参数缺失、空字符串、纯空白或语义无效时，允许只从历史用户请求补齐；非 required 参数不从历史补齐。
  - `Response` 与 `Full Response` 同时作为输出端点存在。
  - `tool_success_summary_prompt` 和 `tool_failure_summary_prompt` 分别控制成功/失败总结。
  - 两个 summary prompt 的默认值均为中文：成功提示词面向终端用户总结真实工具结果；失败提示词明确说明未完成执行并给出参数修正建议。
  - 不保留旧字段 `tool_summary_prompt` 兼容逻辑。
  - Direct Tool Summary 工具失败后仍调用第二次模型生成用户友好失败说明。
  - `_get_llm()` 从 `graph.context.request_variables` 读取 USER-WALLET-ID / TASK-ID 并传给模型供应商。
```

### 历史消息和前端整数输入

```text
src/lfx/src/lfx/components/models_and_agents/agent.py
  - `n_messages=0` 时不读取历史消息。
  - `n_messages>0` 时读取同一 session/context 的最近历史消息供第一次模型调用使用。

src/frontend/src/components/core/parameterRenderComponent/components/intComponent/index.tsx
  - `n_messages` 不设置最大值，允许填写 0、1、20、50 等非负整数。
```

### Trace 可靠落库和 Task ID 关联

```text
src/backend/base/langflow/services/tracing/service.py
  - native tracer 优先初始化。
  - 外部 tracer 初始化失败只记录 debug 日志，不能阻断 native trace 落库。

src/backend/base/langflow/alembic/versions/b7c9d2e4f6a8_add_task_id_to_trace.py
src/backend/base/langflow/services/database/models/traces/model.py
src/backend/base/langflow/api/v1/traces.py
src/backend/base/langflow/services/tracing/native.py
src/backend/base/langflow/services/tracing/repository.py
src/lfx/src/lfx/graph/graph/base.py
src/lfx/src/lfx/services/tracing/base.py
src/lfx/src/lfx/services/tracing/service.py
src/frontend/src/controllers/API/queries/traces/helpers.ts
src/frontend/src/controllers/API/queries/traces/types.ts
src/frontend/src/pages/FlowPage/components/TraceComponent/config/flowTraceColumns.tsx
src/frontend/src/pages/FlowPage/components/TraceComponent/types.ts
  - trace 表、API、前端 Traces 表格均支持 `task_id`。
```

### 前端静态 HTML 缓存控制

```text
src/backend/base/langflow/main.py
  - 非 `/api` 的 HTML 响应增加 no-cache headers。
  - `index.html` FileResponse 增加 no-cache headers。
  - 目的：完整重启后浏览器不继续使用旧 HTML 或旧资源入口，避免组件字段和前端可见性逻辑看起来没有更新。
```

### 临时测试脚本

```text
temp_run_agent_workflow.py
  - 测试 Simple Agent 代理链路。
  - 打印完整工作流响应、完整 outputs、Response 和 Full Response 两个输出组件。

temp_run_legacy_direct_workflow.py
  - 保留旧直连业务工作流测试，仅用于对比和排查。
```
