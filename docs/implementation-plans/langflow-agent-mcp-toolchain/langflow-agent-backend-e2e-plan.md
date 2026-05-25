# Langflow Agent 后端统一调用执行方案

## 目标

后端只对接一条固定的 Agent 工作流，不再直接对接每一条业务工作流。

最终链路固定为：

```text
后端
-> /api/v2/workflows
-> Simple Agent 工作流
-> MCP Tools
-> Langflow Project MCP Server
-> 被代理的业务工作流工具
-> Agent Full Response
-> 后端结构化消费
```

后端长期只需要稳定维护：

```text
Agent 工作流 ID
Agent 输入组件 ID
Agent 结构化输出组件 ID
```

业务工作流新增、修改、导入正式环境后产生的 flow_id / component_id 变化，全部收敛在 Langflow / MCP / Agent 层内部处理。

## 相关文件

- `langflow-mcp-multi-input-plan.md`
  - 规定 MCP Tool 的 inputSchema 如何生成，以及 Agent 参数如何精确注入到多个输入节点。
- `langflow-agent-header-forwarding-plan.md`
  - 规定 `USER-WALLET-ID` / `TASK-ID` 如何从后端请求头全链路透传到被代理工作流。
- `langflow-agent-tool-full-response-plan.md`
  - 规定 Agent 如何返回最终回答、工具调用记录、工具原始输出和结构化业务结果。
- `temp_run_agent_workflow.py`
  - 模拟后端调用 Agent 工作流的真实请求形态。

## 当前代码验证结论

当前代码已经支持后端统一调用 Agent 工作流的第一跳：

```text
src/backend/base/langflow/api/v2/converters.py
  parse_flat_inputs()
  支持把 ChatInput-QHaDe.input_value 转成 tweaks。

src/backend/base/langflow/api/v2/workflow.py
  execute_workflow_sync()
  execute_workflow_background()
  支持从 X-LANGFLOW-GLOBAL-VAR-* 请求头提取 request_variables。
```

当前代码尚未支持 Agent 代理业务工作流的完整闭环：

```text
1. MCP list_tools 不能稳定表达多个输入节点。
2. MCP call_tool 只读取 arguments["input_value"]，不能按组件 ID 注入多个输入节点。
3. MCP tool 输出会被压成文本，丢失组件 ID 和结构化 Data。
4. Agent 组件当前没有 Full Response 输出。
5. MCP Tools 不会自动把 Agent 工作流收到的关键 headers 继续透传给内部 Project MCP Server。
```

因此本总方案依赖三份子方案全部落地后才成立：

```text
多输入参数映射
请求头自动透传
工具完整返回 + Agent Full Response
```

在这些改造完成前，`temp_run_agent_workflow.py` 只能验证后端调用 Agent 的入口形态，不能证明完整代理链路已经可生产使用。

## 分问题执行方案

### 问题一：多个输入节点无法准确传参

对应文档：

```text
langflow-mcp-multi-input-plan.md
```

改法固定为：

```text
Flow.mcp_input_parameters
-> MCP inputSchema
-> MCP call_tool arguments
-> SimplifiedAPIRequest.tweaks
```

后端不再让 Agent 知道目标工作流组件 ID。Agent 只按工具 schema 传语义参数，MCP Server 按保存配置映射到组件 ID。

### 问题二：工具完整原始输出丢失

对应文档：

```text
langflow-agent-tool-full-response-plan.md
```

改法固定为：

```text
Project MCP Server 返回 JSON 输出信封
-> Agent recorder 记录 raw_output
-> Agent.full_response 暴露 tool_calls / final_outputs
```

不能继续把目标工作流多个输出组件压成普通文本。

### 问题三：请求级上下文不能自动传到被代理工作流

对应文档：

```text
langflow-agent-header-forwarding-plan.md
```

改法固定为：

```text
MCP Tools 判断内部 Project MCP Server
-> 从 graph.context.request_variables 读取 USER-WALLET-ID / TASK-ID
-> 自动写入 X-LANGFLOW-GLOBAL-VAR-* headers
-> 只要最终 headers 涉及 USER-WALLET-ID / TASK-ID，就禁用连接/config/session cache
```

这部分不需要重写 Project MCP Server 接收逻辑，因为服务端已经能提取这些 headers。

## 固定调用入口

后端统一调用：

```http
POST /api/v2/workflows
```

请求头必须携带：

```http
Content-Type: application/json
accept: application/json
x-api-key: <LANGFLOW_API_KEY>
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID: <user_wallet_id>
X-LANGFLOW-GLOBAL-VAR-TASK-ID: <task_id>
```

请求体固定形态：

```json
{
  "flow_id": "f316c44e-21df-4d24-a651-3a9fc9cf291a",
  "background": true,
  "stream": false,
  "inputs": {
    "ChatInput-QHaDe.input_value": "{\"requirement\":\"...\",\"parameters\":{...}}"
  }
}
```

其中：

```text
flow_id
  Simple Agent 工作流 ID。

ChatInput-QHaDe.input_value
  Agent 工作流统一入口。值必须是结构化 JSON 字符串。
```

## 后端传入 Agent 的任务协议

后端写入 `ChatInput-QHaDe.input_value` 的 JSON 字符串必须保持精简，只包含 Agent 选工具和填参需要的信息：

```json
{
  "requirement": "根据爆款视频和用户业务信息生成复刻执行方案规格书",
  "parameters": {
    "hot_video_info": "爆款视频链接或解析信息",
    "business_info": "用户业务信息",
    "direction": "{\"direction_num\":1,\"title\":\"最简产品置换\",\"summary\":\"直接替换核心产品，测试原爆款叙事框架适用性\"}"
  }
}
```

字段定义：

```text
requirement
  本次任务的自然语言需求。Agent 用它理解目标并选择合适工具。

parameters
  后端传入的结构化业务信息。
  Agent 不直接关心目标工作流组件 ID，只按 Tool inputSchema 填参。
  当前工作流输入节点只支持 string；如果某个业务字段本身是对象或数组，后端必须先序列化为 JSON 字符串，再放入 parameters。
```

不要把 `request_id`、`output_requirement`、目标工作流 ID、目标组件 ID 放入 Agent 输入。
任务关联统一使用请求头 `X-LANGFLOW-GLOBAL-VAR-TASK-ID`，Agent `Full Response.request_id` 可由执行层从该请求头兜底填充。

## Simple Agent 工作流结构

当前 Agent 工作流应保持这个结构：

```text
ChatInput
  -> Agent.input_value

MCP Tools
  -> Agent.tools

Agent.response
  -> ChatOutput.input_value

Agent.full_response
  -> DataOutput / 结构化输出节点
```

要求：

- `ChatInput` 是后端唯一输入入口。
- `MCP Tools` 连接内部 Langflow Project MCP Server。
- `Agent.response` 保留给 UI 对话展示。
- `Agent.full_response` 是后端稳定消费的结构化输出。

## Agent 调度规则

Agent instructions 固定要求：

```text
你是一个严谨的工作流调度 Agent。你的职责是根据用户需求选择最合适的工具，并严格按照工具的 inputSchema 构造参数后调用工具。

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
13. 返回结果时，用用户能理解的语言简洁说明执行结果；不要暴露内部工具调用细节，除非用户明确要求。
```

### Agent Instructions 历史补齐规则

在启用 `Number of Chat History Messages` 时，第一次模型调用会收到同一 `session_id` / `context_id` 下的最近历史用户消息和 AI 消息。Simple Agent 的提示词必须明确历史补齐边界：

```text
1. 当前请求中的非空 parameters 优先级最高；如果当前请求提供了非空参数，必须使用当前请求的值，不要用历史值覆盖。
2. 如果目标工具的 required 参数在当前请求中缺失、为空字符串、纯空白字符串，或语义上没有有效内容，允许从同一 session 最近历史用户请求中查找同名或语义等价的非空参数进行补齐。
3. 历史补齐只允许用于 required 参数；非 required 参数如果当前请求没有提供有效值，不要从历史补齐，也不要传入工具。
4. 只能使用历史中的用户请求参数补齐，不要使用历史 AI 回答、Full Response、工具输出、错误提示或示例内容作为参数来源。
5. 如果历史中存在多个候选参数，使用距离当前请求最近且非空的用户参数。
6. required string 参数最终必须存在，且不能为空字符串或纯空白字符串。
7. 如果 required 参数在当前请求中缺失或为空，并且历史请求中也找不到可用的非空值，直接说明缺少哪些参数，不要调用工具。
```

因此：

```text
n_messages=0
  等同单次调用，Agent 不会收到历史消息，不能补齐历史参数。

n_messages>0 且 session_id 相同
  Agent 可以用历史用户请求补齐 required 参数。

当前请求显式传了非 required 空值或缺失非 required 参数
  不从历史补齐，也不传给工具。
```

## MCP Tool 参数执行规则

Agent 看到的是 MCP Tool 的语义化 `inputSchema`。

示例：

```json
{
  "hot_video_info": "爆款视频信息",
  "business_info": "业务信息",
  "direction": "复刻方向"
}
```

Langflow MCP 执行层负责把这些语义化参数映射到目标业务工作流的具体输入节点：

```python
tweaks = {
    "TextInput-video": {
        "input_value": "爆款视频信息"
    },
    "TextInput-business": {
        "input_value": "业务信息"
    },
    "TextInput-direction": {
        "input_value": "复刻方向"
    }
}
```

该能力按 `langflow-mcp-multi-input-plan.md` 执行。

## 请求头透传规则

后端传给 `/api/v2/workflows` 的关键请求头必须全链路携带：

```text
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID
X-LANGFLOW-GLOBAL-VAR-TASK-ID
```

第一跳：

```text
后端 -> Agent 工作流
```

由 `/api/v2/workflows` 现有逻辑提取到：

```json
{
  "request_variables": {
    "USER-WALLET-ID": "user_001",
    "TASK-ID": "task_001"
  }
}
```

第二跳：

```text
Agent 工作流 -> MCP Tools -> 内部 Project MCP Server -> 工具工作流
```

由 MCP Tools 组件自动透传完成。

执行要求：

```text
当 MCP Tools 连接内部 Langflow Project MCP Server 时，
自动透传 USER-WALLET-ID / TASK-ID 到 X-LANGFLOW-GLOBAL-VAR-* headers。
```

该能力按 `langflow-agent-header-forwarding-plan.md` 执行。

## Agent 结构化输出协议

后端最终消费 `Agent.full_response`。

固定输出结构：

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
        "business_info": "业务信息",
        "direction": "复刻方向"
      },
      "raw_output": {
        "flow_id": "target-flow-id",
        "flow_name": "业务工作流名称",
        "outputs": {
          "ChatOutput-xxx": {
            "type": "message",
            "content": "自然语言输出"
          },
          "DataOutput-yyy": {
            "type": "data",
            "content": {
              "strategy": {},
              "visual_spec": {},
              "content_spec": {}
            }
          }
        }
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

该能力按 `langflow-agent-tool-full-response-plan.md` 执行。

## temp_run_agent_workflow.py 的角色

`temp_run_agent_workflow.py` 用来模拟后端调用 Simple Agent 链路，不参与生产链路。

它必须展示：

```text
1. 后端如何组装 /api/v2/workflows 请求。
2. 后端如何传递 USER-WALLET-ID / TASK-ID 请求头。
3. 后端如何把结构化任务 JSON 写入 Agent ChatInput。
4. 后端如何轮询 job_id。
5. 后端如何读取结构化输出。
```

脚本默认使用 Agent 模式：

```text
LANGFLOW_WORKFLOW_MODE=agent
```

注意：V2 workflow 的 `outputs` key 使用输出组件 ID，而不是组件显示名。完整返回能力落地后，脚本必须通过 `LANGFLOW_AGENT_FULL_RESPONSE_COMPONENT_ID` 配置 Agent 结构化输出组件 ID。

脚本还必须用于调试双输出链路，运行后打印：

```text
1. 完整工作流响应。
2. 完整 workflow outputs。
3. Agent Response 输出组件完整结果。
4. Agent Response 输出组件解析结果。
5. Agent Full Response 输出组件完整结果。
6. Agent Full Response 输出组件解析结果。
```

当前默认组件 ID：

```text
LANGFLOW_AGENT_RESPONSE_COMPONENT_ID=ChatOutput-XtDPp
LANGFLOW_AGENT_FULL_RESPONSE_COMPONENT_ID=ChatOutput-YUG9M
```

如果 Simple Agent 工作流重新连线或输出节点 ID 变化，必须只更新脚本/后端配置中的组件 ID，不按 `Response` / `Full Response` display_name 查找。

脚本注释使用中文，说明它如何模拟后端组装请求、传 headers、轮询 job、读取两个输出端点。

### 临时测试脚本拆分

当前测试脚本按调用模式拆分，避免 Agent 代理链路和旧直连链路混在一个文件里：

```text
temp_run_agent_workflow.py
  模拟后端调用 Simple Agent 工作流。
  负责验证 ChatInput 入参、session_id、USER-WALLET-ID / TASK-ID headers、Response / Full Response 两个输出组件。

temp_run_legacy_direct_workflow.py
  保留旧的直接调用业务工作流方式。
  只用于对比旧链路或排查业务工作流本身，不代表最终后端接入方式。
```

所有临时脚本必须以 `temp_` 开头，避免被误认为生产入口。

## 本地启动与端到端验证

本项目本地验证固定使用 7850 端口：

```bash
ALL_PROXY= all_proxy= LANGFLOW_DEVELOPER_API_ENABLED=true LFX_DEV=1 PYTHONIOENCODING=utf-8 uv run langflow run \
  --frontend-path src/frontend/build \
  --log-level debug \
  --host 0.0.0.0 \
  --port 7850 \
  --env-file .env \
  --no-open-browser
```

访问地址：

```text
http://localhost:7850
```

验证前需要确认：

```text
1. 前端 build 已包含最新前端代码；否则 UI 改动需要重新构建并重启。
2. Alembic migration 文件存在并可升级数据库。
3. Simple Agent 工作流使用 Agent 的 Response 和 Full Response 两个输出端点。
4. MCP Tools 连接内部 Project MCP Server。
5. MCP Tools 的 tool_call_timeout_minutes 足够覆盖目标业务工作流耗时。
```

重启时必须完整重启 Langflow 进程，不能只依赖组件热更新：

```text
1. 检查 7850 端口占用。
2. 停止旧 Langflow 进程。
3. 使用上面的启动命令重新启动，并等待 `/health_check` 正常。
```

### 前端 HTML 缓存控制

对应文件：

```text
src/backend/base/langflow/main.py
```

实现要求：

```text
1. 非 `/api` 的 `text/html` 响应必须带 no-cache headers。
2. `index.html` 的 FileResponse 也必须带 no-cache headers。
3. headers 使用：
   Cache-Control: no-store, no-cache, must-revalidate, max-age=0
   Pragma: no-cache
   Expires: 0
```

原因是本项目经常修改组件定义和前端配置。如果浏览器继续使用旧 HTML 或旧资源入口，即使后端已经完整重启，页面上仍可能看不到最新字段或最新可见性逻辑。
该能力只用于本地和部署后的前端更新一致性，不改变 API 响应缓存策略。

数据库迁移文件必须随代码提交。
Langflow 启动时会执行数据库初始化和 Alembic upgrade；如果新增字段只改模型、不提供 migration 文件，别人拉取项目部署到旧数据库时会缺列或迁移链断裂。

本地调试时，`src/backend/base/langflow/initial_setup/starter_projects/*.json` 已经被 Git 跟踪，即使 `.gitignore` 包含该路径，也不会自动忽略已有 tracked 文件。
如果只是本地运行导致 starter project JSON 变化，可以在本机使用 `skip-worktree` 降噪；这是本地 Git index 状态，不属于项目代码能力，也不会随提交同步给别人。

## Agent 双模式端到端要求

Agent 节点必须支持：

```text
Direct Tool Summary
Agent Loop
```

端到端生产验证优先使用 `Direct Tool Summary`。
该模式下，工具执行完成后：

```text
1. Full Response 直接保留工具 raw_output、tool_calls、final_outputs、errors。
2. Response 由第二次独立模型调用总结工具原返回。
3. 第二次模型调用不再绑定 tools，不再进入 Agent 循环。
```

`Agent Loop` 作为兼容模式保留原有循环，不在该模式中强行处理 NewAPI `reasoning_content` 兼容问题。

## Trace 关联验证

每次后端调用应传入：

```http
X-LANGFLOW-GLOBAL-VAR-TASK-ID: <task_id>
```

验证要求：

```text
1. Simple Agent 工作流 trace 中显示同一个 Task ID。
2. 被代理业务工作流 trace 中显示同一个 Task ID。
3. Traces 页面在 Trace ID 后有 Task ID 列。
4. traces API 可以按 task_id 过滤或搜索。
```

这用于确认一次后端请求触发的 Agent trace 和业务工具 trace 属于同一条链路。

## 实施顺序

必须按以下顺序落地：

1. 保持 `/api/v2/workflows` 后端统一调用 Agent 的请求形态，并用 `temp_run_agent_workflow.py` 验证第一跳。
2. 落地 MCP 多输入参数配置存储和 Edit Tools 参数配置 UI。
3. 改造 `json_schema_from_flow()`，让 MCP `list_tools` 使用参数配置生成 inputSchema。
4. 改造 `handle_call_tool()`，让 MCP `call_tool` 把 arguments 映射成 `SimplifiedAPIRequest.tweaks`。
5. 改造 MCP Tools，对内部 Project MCP Server 自动透传关键 headers，并处理缓存。
6. 改造 MCP Server 工具执行返回，输出完整 JSON 信封。
7. 改造 Agent，新增 `Full Response`，并由执行层记录 tool_calls。
8. 改造 Agent 双模式，默认使用 `Direct Tool Summary`，保留 `Agent Loop`。
9. 更新 Simple Agent instructions，明确工具选择、参数填写、双模式和输出规则。
10. 更新 `temp_run_agent_workflow.py`，打印完整响应、完整 outputs，以及 Response / Full Response 两个输出组件的完整结果。
11. 为 trace 增加 Task ID 链路显示，并用真实请求验证 Agent trace 与业务工作流 trace 可关联。
12. 用真实业务工作流完成端到端验证。

## 验收标准

### 后端对接

- 后端只调用 Simple Agent 工作流。
- 后端不需要知道被代理业务工作流 ID。
- 后端不需要知道被代理业务工作流组件 ID。
- 后端请求头中的 `USER-WALLET-ID` / `TASK-ID` 可以全链路到达工具工作流。

### Agent 调度

- Agent 可以根据 `requirement`、`parameters` 和 Tool inputSchema 命中正确 MCP Tool。
- Agent 可以根据 Tool inputSchema 正确填参。
- 必填参数缺失时，Agent 返回结构化错误。
- `Direct Tool Summary` 模式下，工具只执行一次，第二次模型调用只做总结。
- `Agent Loop` 模式下，保留原有循环行为。

### 工具执行

- 多个 `TextInput` / `ChatInput` 输入节点可以分别收到正确参数。
- 工具工作流多个输出组件可以被完整收集。
- 工具调用失败时，错误进入 `tool_calls[].error` 和 `errors`。

### 后端输出

- 后端可以读取 `request_id`。
- 后端可以读取 `answer`。
- 后端可以读取 `selected_tool`。
- 后端可以读取完整 `tool_calls`。
- 后端可以读取业务结构化 `final_outputs`。
- 后端可以同时读取用户友好 `Response` 和结构化 `Full Response`。
- `temp_run_agent_workflow.py` 能打印完整 workflow 响应和两个输出组件的完整结果。

### Trace 观测

- Simple Agent 工作流和被代理业务工作流的 trace 都保存同一个 `Task ID`。
- Traces 页面可以直接通过 `Task ID` 关联一次请求中的多条 trace。

## 最终状态

完成后，后端统一对接 Agent 工作流。

业务工作流作为 Project MCP Server 中的 Tool 被 Agent 调用。

后端不再维护每条业务工作流的输入组件 ID、输出组件 ID 和跨环境 ID 映射。
