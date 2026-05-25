# Langflow Agent 代理工作流请求头透传执行方案

## 目标

后端通过 `/api/v2/workflows` 调用 Simple Agent 工作流时，以下两个请求级上下文必须全链路传递到被 Agent 代理调用的业务工作流：

```text
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID
X-LANGFLOW-GLOBAL-VAR-TASK-ID
```

固定链路：

```text
后端请求头
-> /api/v2/workflows
-> Agent 工作流 request_variables
-> MCP Tools 自动透传
-> Langflow Project MCP Server 请求头
-> 工具工作流 request_variables
```

## 相关文件

- 总链路方案：`langflow-agent-backend-e2e-plan.md`
- MCP 多输入参数方案：`langflow-mcp-multi-input-plan.md`
- Agent 完整返回方案：`langflow-agent-tool-full-response-plan.md`
- 后端调用模拟脚本：`temp_run_agent_workflow.py`

## 当前代码验证结论

第一跳已经成立：

```text
src/backend/base/langflow/api/v2/workflow.py
  execute_workflow_sync()
  execute_workflow_background()
```

当前会调用 `extract_global_variables_from_headers(http_request.headers)`，并把结果放入：

```python
context = {"request_variables": request_variables}
```

因此后端调用 Agent 工作流时，`USER-WALLET-ID` / `TASK-ID` 已经能进入 Agent 工作流的 `graph.context`。

Project MCP Server 接收逻辑也已经成立：

```text
src/backend/base/langflow/api/v1/mcp_projects.py
  _dispatch_project_streamable_http()
```

当前会从 MCP 请求头中提取 `X-LANGFLOW-GLOBAL-VAR-*`，并写入 `current_request_variables_ctx`，后续 `handle_call_tool()` 会把它传给工具工作流执行上下文。

当前缺口在 MCP Tools：

```text
src/lfx/src/lfx/components/models_and_agents/mcp_component.py
  update_tool_list()
```

当前逻辑能读取 `self.graph.context["request_variables"]`，也能把 headers 传给 `update_tools()`，但它不会自动判断内部 Project MCP Server，也不会自动补齐关键 headers。

同时当前 headers 合并规则是：

```python
merged_headers = {**existing_headers, **component_headers_dict}
```

也就是 UI 手填 headers 覆盖 server config。对于 `USER-WALLET-ID` / `TASK-ID` 这两个系统级 header，这个优先级必须反过来：内部 Project MCP Server 场景下必须由当前请求变量覆盖手填值。

缓存风险也已经确认：

```text
src/lfx/src/lfx/components/models_and_agents/mcp_component.py
  use_cache 分支可能提前返回 cached tools / config。
```

一旦自动透传请求级 header，不能复用可能带旧 headers 的 MCP 连接。

## 代码落地方案

### 1. MCP Tools 中新增内部 Project MCP Server 判断

修改文件：

```text
src/lfx/src/lfx/components/models_and_agents/mcp_component.py
```

新增函数：

```python
def is_internal_project_mcp_server(server_config: dict) -> bool:
    candidates = []
    if server_config.get("url"):
        candidates.append(str(server_config["url"]))
    candidates.extend(str(x) for x in server_config.get("args", []) or [])
    return any("/api/v1/mcp/project/" in x for x in candidates)
```

第一版只按路径判断，不按域名判断。原因是本地、测试、正式环境域名可能不同，但内部 Project MCP Server 路径固定。

### 2. MCP Tools 自动写入关键 headers

修改位置：

```text
MCPToolsComponent.update_tool_list()
```

执行顺序必须严格调整为：

```text
1. 先解析 server_config。
2. 判断 server_config 是否指向内部 Project MCP Server。
3. 读取当前 Agent 工作流 graph.context.request_variables。
4. 合并普通 headers。
5. 如果是内部 Project MCP Server，只有当前 request_variables 中存在真实值时，才写入对应关键 header。
6. 如果是内部 Project MCP Server，但当前 request_variables 没有对应真实值，则移除同名关键 header，避免 UI 手动执行时传入固定旧值或占位字符串。
7. 检查最终 headers 是否涉及 USER-WALLET-ID / TASK-ID。
8. 只要涉及这两个请求级 header，本次禁止使用连接/config/session cache。
9. 此时才允许读取普通工具缓存；禁止在步骤 1-8 之前直接 cached return。
10. 再调用 update_tools()。
```

伪代码：

```python
REQUEST_CONTEXT_HEADER_NAMES = {
    "x-langflow-global-var-user-wallet-id",
    "x-langflow-global-var-task-id",
}


def remove_request_context_headers(headers: dict) -> dict:
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in REQUEST_CONTEXT_HEADER_NAMES
    }


def headers_involve_request_context(headers: dict) -> bool:
    return any(key.lower() in REQUEST_CONTEXT_HEADER_NAMES for key in headers)


def normalize_headers(headers: Any) -> dict:
    """只做格式归一化，不解析 request_variables。"""
    result = {}
    if isinstance(headers, dict):
        items = headers.items()
    elif isinstance(headers, list):
        items = (
            (item.get("key"), item.get("value"))
            for item in headers
            if isinstance(item, dict) and "key" in item and "value" in item
        )
    else:
        return result

    for key, value in items:
        if isinstance(key, str) and isinstance(value, str):
            result[key.lower()] = value
    return result


server_config = resolve_mcp_config(...)
is_internal = is_internal_project_mcp_server(server_config)

request_variables = {}
if hasattr(self, "graph") and self.graph and hasattr(self.graph, "context"):
    request_variables = self.graph.context.get("request_variables") or {}

server_config["headers"] = merge_normal_headers(server_config, component_headers)

if is_internal:
    headers = normalize_headers(server_config.get("headers", {}))
    headers = remove_request_context_headers(headers)

    if request_variables.get("USER-WALLET-ID"):
        headers["X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID"] = "USER-WALLET-ID"

    if request_variables.get("TASK-ID"):
        headers["X-LANGFLOW-GLOBAL-VAR-TASK-ID"] = "TASK-ID"

    server_config["headers"] = headers

headers = normalize_headers(server_config.get("headers", {}))
has_request_context_headers = headers_involve_request_context(headers)

if has_request_context_headers:
    use_cache = False
    disable_session_cache = True

# 注意：cache 检查只能发生在最终 headers 判断之后。
# 不能在函数开头因为 cached tools/config 存在就直接 return。
```

这里 header value 保持变量名，继续复用已有解析逻辑：

```text
src/lfx/src/lfx/base/mcp/util.py
  _process_headers()
  _resolve_global_variables_in_headers()
```

`update_tools(..., request_variables=request_variables)` 会把变量名解析为当前请求真实值。

### 3. 系统关键 header 优先级

内部 Project MCP Server 场景下，优先级固定为：

```text
当前请求 request_variables
> MCP Tools UI 手填 headers
> MCP Server 配置 headers
```

只对以下两个 header 生效：

```text
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID
X-LANGFLOW-GLOBAL-VAR-TASK-ID
```

其他 headers 仍保持现有逻辑：组件手填值可以覆盖 server config。

### 4. 缓存处理

自动透传关键 headers 时必须关闭请求级连接缓存：

```text
1. 不能在解析 server_config 之前使用 cached server_config / tools 直接返回。
2. 必须先得到最终 headers，再判断是否涉及 USER-WALLET-ID / TASK-ID。
3. 只要最终 headers 涉及这两个请求级上下文，本次 use_cache=False。
4. 不复用可能携带旧 headers 的 MCP session。
5. 不复用可能携带旧 headers 的 cached server_config。
6. 不复用当前整包 MCP cache，包括 tools、tool_names、tool_cache、config。
7. 不能只设置 `MCPToolsComponent.use_cache=False`；还必须通过 `disable_session_cache=True` 绕过 `MCPSessionManager` 的 persistent session cache。
```

第一版执行策略：

```text
只要最终 headers 中存在以下任一请求级上下文 header，
本次 update_tool_list() 强制 use_cache=False：

X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID
X-LANGFLOW-GLOBAL-VAR-TASK-ID

内部 Project MCP Server 场景下，如果当前请求没有真实 `USER-WALLET-ID` / `TASK-ID`，不得注入这两个 header，也不得保留 UI 手填的同名旧值。

原因是避免：
1. UI 手动执行 Simple Agent 时没有真实请求变量，却把 USER-WALLET-ID / TASK-ID 字符串原样传给 MCP Server。
2. 第一次请求带 headers、第二次请求未带 headers 时，第二次误用第一次缓存下来的 config/session。
```

第一版不做 schema-only cache。当前代码的 MCP cache 是整包缓存，尚未把工具 schema 与连接/config/session 拆开，因此涉及这两个请求级 header 时必须整包禁用。

注意：当前代码还有 `MCPSessionManager` 级别的 persistent session cache。它独立于 `MCPToolsComponent.use_cache`，会按 server key 复用已有 MCP session。因此实现时必须新增唯一开关 `disable_session_cache`，并按固定链路传递：

```text
MCPToolsComponent.update_tool_list()
-> update_tools(..., disable_session_cache=True)
-> MCPStdioClient / MCPStreamableHttpClient
-> 连接时绕过 MCPSessionManager.get_session()
-> 创建一次性 MCP 连接
-> 本次工具加载结束后关闭/清理该一次性连接
```

不要采用“先清理共享 session 再连接”的方案，避免误删同一进程中其他并发请求正在使用的 session。

## 唯一执行原则

正式能力由 MCP Tools 组件自动完成。

规则固定为：

```text
当 MCP Tools 连接内部 Langflow Project MCP Server 时，
如果当前 Agent 工作流上下文中存在真实 USER-WALLET-ID / TASK-ID，
自动透传对应值到 X-LANGFLOW-GLOBAL-VAR-* headers。

如果当前上下文中没有真实 USER-WALLET-ID / TASK-ID，
不注入这两个 header，也不保留 UI 手填的同名旧值。
```

不依赖用户在 UI 中手动配置 headers。

不对外部 MCP Server 默认透传内部请求变量。

## 旧模式逻辑

旧模式中，后端直接调用业务工作流：

```text
后端
-> /api/v2/workflows
-> 业务工作流
```

后端请求头：

```http
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID: user_001
X-LANGFLOW-GLOBAL-VAR-TASK-ID: task_001
```

`/api/v2/workflows` 会提取为：

```json
{
  "request_variables": {
    "USER-WALLET-ID": "user_001",
    "TASK-ID": "task_001"
  }
}
```

业务工作流中的组件可以读取：

```text
USER-WALLET-ID
TASK-ID
```

## Agent 代理模式逻辑

新模式中，后端调用 Agent 工作流：

```text
后端
-> /api/v2/workflows
-> Simple Agent 工作流
-> MCP Tools
-> Project MCP Server
-> 工具工作流
```

第一跳：

```text
后端 -> Agent 工作流
```

仍由 `/api/v2/workflows` 提取请求变量。

第二跳：

```text
Agent 工作流 -> MCP Tools -> Project MCP Server -> 工具工作流
```

由 MCP Tools 自动透传关键请求变量。

## 自动透传范围

自动透传只包含以下关键变量：

```text
USER-WALLET-ID
TASK-ID
```

对应请求头：

```text
USER-WALLET-ID -> X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID
TASK-ID        -> X-LANGFLOW-GLOBAL-VAR-TASK-ID
```

## 内部 Project MCP Server 判断规则

MCP Tools 只在连接内部 Langflow Project MCP Server 时自动透传。

第一版判断规则：

```text
server_config.url 或 server_config.args 中包含：
/api/v1/mcp/project/
```

示例：

```text
http://localhost:7850/api/v1/mcp/project/d8109a46-eb65-4a7c-b34e-e85be499defc/streamable
```

这个地址属于内部 Project MCP Server，必须自动透传。

外部 MCP Server 示例：

```text
https://third-party.example.com/mcp/streamable
```

这种地址不自动透传内部请求变量。

## 代码落点

主要修改文件：

```text
src/lfx/src/lfx/components/models_and_agents/mcp_component.py
```

辅助逻辑涉及：

```text
src/lfx/src/lfx/base/mcp/util.py
```

服务端接收逻辑已有：

```text
src/backend/base/langflow/api/v1/mcp_projects.py
```

## MCP Tools 执行规则

在 `MCPToolsComponent.update_tool_list()` 中执行以下逻辑：

```text
1. 解析 MCP Server 配置。
2. 读取当前图上下文 request_variables。
3. 判断 MCP Server 是否为内部 Langflow Project MCP Server。
4. 如果是内部 Project MCP Server，只有当前 request_variables 中存在真实值时，才写入对应关键 header。
5. 如果当前 request_variables 没有对应真实值，则移除同名关键 header，不允许旧值或手动错误值继续传递。
6. 只要最终 headers 涉及 USER-WALLET-ID / TASK-ID，就关闭可能携带旧 headers 的缓存连接、缓存 config、缓存 session。
7. 调用 update_tools() 连接 MCP Server。
```

伪代码：

```python
request_variables = self.graph.context.get("request_variables") or {}

if is_internal_project_mcp_server(server_config):
    headers = normalize_headers(server_config.get("headers", {}))
    headers = remove_request_context_headers(headers)

    if request_variables.get("USER-WALLET-ID"):
        headers["X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID"] = "USER-WALLET-ID"

    if request_variables.get("TASK-ID"):
        headers["X-LANGFLOW-GLOBAL-VAR-TASK-ID"] = "TASK-ID"

    server_config["headers"] = headers

headers = normalize_headers(server_config.get("headers", {}))
if headers_involve_request_context(headers):
    use_cache = False
```

这里 header value 写变量名：

```text
USER-WALLET-ID
TASK-ID
```

随后 `update_tools()` 会把变量名解析为当前请求真实值：

```http
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID: user_001
X-LANGFLOW-GLOBAL-VAR-TASK-ID: task_001
```

## Header 优先级

对这两个关键 header，系统自动值优先。

如果用户在 MCP Tools headers 中手动写了同名 header，且当前是内部 Project MCP Server：

```text
系统自动值覆盖用户手动值。
```

原因：

```text
USER-WALLET-ID / TASK-ID 是计费、日志、错误归因链路的关键上下文。
内部 Project MCP Server 场景下必须保证它们来自当前后端请求，而不是 UI 中的固定旧值。
```

## Transport 处理

### Streamable HTTP / SSE

MCP Tools 连接内部 Project MCP Server 时，把解析后的 headers 传给 HTTP MCP client。

实际请求：

```http
POST /api/v1/mcp/project/{project_id}/streamable
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID: user_001
X-LANGFLOW-GLOBAL-VAR-TASK-ID: task_001
```

### Stdio + mcp-proxy

如果 MCP Server 配置是 stdio，并通过 `mcp-proxy` 转发到 Project MCP Server，则把 headers 注入为 proxy 参数：

```text
--headers X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID user_001
--headers X-LANGFLOW-GLOBAL-VAR-TASK-ID task_001
```

## 缓存处理

只要启用了关键请求变量自动透传，MCP Tools 不得复用可能携带旧 headers 的缓存连接。

原因：

```text
用户 A -> USER-WALLET-ID=user_a, TASK-ID=task_a
用户 B -> USER-WALLET-ID=user_b, TASK-ID=task_b
```

如果复用旧 session，可能导致：

```text
用户 B 的工具工作流拿到用户 A 的请求变量。
```

执行规则：

```text
1. 自动透传关键变量时，禁用 MCP Tools server/tool/cache 整包缓存。
2. 自动透传关键变量时，通过 `disable_session_cache=True` 绕过 MCPSessionManager persistent session cache。
3. 自动透传关键变量时，直接新建 MCP 连接。
4. 第一版不缓存工具 schema；后续如需缓存，必须先新增独立的 schema-only cache 结构。
```

## MCP Server 接收规则

Langflow Project MCP Server 接收到请求后，继续使用现有 header 提取逻辑。

收到：

```http
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID: user_001
X-LANGFLOW-GLOBAL-VAR-TASK-ID: task_001
```

提取为：

```json
{
  "request_variables": {
    "USER-WALLET-ID": "user_001",
    "TASK-ID": "task_001"
  }
}
```

然后执行工具工作流时，将该上下文传入图执行。

## Trace Task ID 关联

为了排查 Simple Agent 工作流调用业务工作流的完整链路，`TASK-ID` 还必须进入 Trace 数据。
该能力不是只针对某一条工作流，而是对所有工作流 trace 生效。

固定链路：

```text
X-LANGFLOW-GLOBAL-VAR-TASK-ID
-> graph.context.request_variables["TASK-ID"]
-> Graph tracing context
-> NativeTracer
-> trace.task_id
-> Traces 页面 Task ID 列
```

数据库层需要在 `trace` 表增加可空字段和索引：

```text
trace.task_id
ix_trace_task_id
```

API 层需要：

```text
1. TraceRead / TraceSummaryRead / TraceCreate 暴露 task_id。
2. traces 列表接口支持按 task_id 过滤。
3. search 同时支持匹配 task_id。
```

前端 Traces 表格需要在 `Trace ID` 后增加 `Task ID` 列。
这样同一次后端请求中的 Simple Agent trace 和被代理业务工作流 trace 会显示相同 `Task ID`，可以直接关联。

注意：Trace 的 `task_id` 来自请求级 `TASK-ID`，不能从模型输出、工具返回或 display_name 反推。

## 外部 Agent 调用 Langflow MCP Server

外部自研 Agent 如果直接作为 MCP Client 调用 Langflow Project MCP Server，也可以主动传这两个 header。

示例：

```http
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID: user_001
X-LANGFLOW-GLOBAL-VAR-TASK-ID: task_001
```

Langflow MCP Server 会接收并提取。

路径判断只发生在 Langflow 内部 MCP Tools 自动透传逻辑中，用于避免把内部变量自动发给外部 MCP Server。

## 鉴权关系

MCP Server 鉴权和请求变量透传是两件事。

如果 Project MCP Server 使用 `None (public)`：

```text
不需要鉴权 header。
仍然需要请求变量 header 透传。
```

如果 Project MCP Server 使用 `API Key`：

```text
需要鉴权 header。
也需要请求变量 header 透传。
```

鉴权 header 示例：

```http
x-api-key: <LANGFLOW_API_KEY>
```

请求变量 header 示例：

```http
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID: user_001
X-LANGFLOW-GLOBAL-VAR-TASK-ID: task_001
```

## 安全边界

不允许对所有 MCP Server 默认透传 `X-LANGFLOW-GLOBAL-VAR-*`。

原因：

```text
USER-WALLET-ID / TASK-ID 属于内部业务上下文。
外部第三方 MCP Server 不应默认收到这些值。
```

自动透传只对内部 Project MCP Server 生效。

## 实施步骤

1. 在 `mcp_component.py` 中新增 `is_internal_project_mcp_server()`。
2. 在 `update_tool_list()` 中先解析 `server_config`，再判断是否为内部 Project MCP Server。
3. 读取 `self.graph.context["request_variables"]`。
4. 执行现有普通 headers 合并逻辑。
5. 如果是内部 Project MCP Server，只有当前请求变量存在真实值时，才写入对应系统关键 header。
6. 如果是内部 Project MCP Server，但当前请求变量没有真实值，则移除同名关键 header。
7. 得到最终 headers 后，判断是否涉及 `USER-WALLET-ID` / `TASK-ID`。
8. 只要最终 headers 涉及这两个请求级上下文，强制禁用整包 MCP cache，并通过 `disable_session_cache=True` 绕过 `MCPSessionManager` persistent session cache，避免复用旧 headers/config/session。
9. 保持 `update_tools(..., request_variables=request_variables)` 调用。
10. 确认 Streamable HTTP / SSE headers 正确传递。
11. 确认 stdio + mcp-proxy headers 正确注入。
12. 增加同一 Agent 工作流连续两次不同用户/任务请求的测试，验证不会串 headers。
13. 增加“第一次请求带 header、第二次请求不带 header”的测试，验证第二次不会复用第一次的 headers/config/session。
14. 在 trace 模型和 Alembic 中增加 `task_id` 字段与索引。
15. 在 tracing service / native tracer 中把 `graph.context.request_variables["TASK-ID"]` 写入 trace。
16. 在 traces API、前端类型和 Traces 表格中暴露 `Task ID` 列，并放在 `Trace ID` 后。

## 验收标准

### 后端到 Agent

- 后端调用 `/api/v2/workflows` 时传入两个 header。
- Agent 工作流上下文能读取到 `USER-WALLET-ID` 和 `TASK-ID`。

### Agent 到工具工作流

- 后端传入真实 `USER-WALLET-ID` / `TASK-ID` 时，MCP Tools 连接内部 Project MCP Server 会自动带上对应 header。
- 工具工作流上下文能读取到同一组真实 `USER-WALLET-ID` 和 `TASK-ID`。
- UI 手动执行 Simple Agent 且没有真实请求变量时，不会向内部 Project MCP Server 发送这两个 header。
- 用户无需在 MCP Tools UI 中手动配置这两个 header。

### 安全性

- MCP Tools 连接外部 MCP Server 时不会自动透传这两个 header。
- 同名手动 header 不会覆盖内部 Project MCP Server 场景下的系统自动值。
- 内部 Project MCP Server 场景下，如果当前请求没有真实变量，同名手动 header 会被移除。

### 缓存

- 不会复用上一请求的 `USER-WALLET-ID` / `TASK-ID`。
- 多用户连续请求不会串用 headers。

### Trace 关联

- 所有工作流 trace 都支持保存 `task_id`。
- Simple Agent 工作流 trace 与被代理业务工作流 trace 可以通过同一个 `Task ID` 关联。
- Traces 页面在 `Trace ID` 后显示 `Task ID`。
- traces 列表接口支持按 `task_id` 过滤和搜索。

## 最终状态

后端继续只调用 Simple Agent 工作流。

后端继续只在 `/api/v2/workflows` 请求头中传：

```text
X-LANGFLOW-GLOBAL-VAR-USER-WALLET-ID
X-LANGFLOW-GLOBAL-VAR-TASK-ID
```

Agent 调用内部 Langflow Project MCP Server 工具时，MCP Tools 自动完成第二跳透传。

被代理工具工作流继续像旧模式一样读取同一份请求级上下文。

## 2026-05-25 实现同步补充

本节记录请求级上下文透传链路之外，后续实现中补齐的观测和模型调用行为。

### Agent 模型调用也必须携带请求级上下文

对应文件：

```text
src/lfx/src/lfx/components/models_and_agents/agent.py
src/lfx/src/lfx/base/models/unified_models.py
```

实现要求：

```text
1. `AgentComponent._get_llm()` 从 `self.graph.context.request_variables` 读取 `USER-WALLET-ID` 和 `TASK-ID`。
2. 读取到真实值后传给 `get_llm(user_wallet_id=..., task_id=...)`。
3. `Direct Tool Summary` 模式下第一次工具选择模型调用、第二次总结模型调用都复用 `_get_llm()`，因此都携带这两个值。
4. `Agent Loop` 模式保留原循环，但底层 LLM 同样来自 `_get_llm()`，因此模型调用也携带这两个值。
5. 这部分只影响 Langflow 内部 Agent 组件发起的模型请求，不影响外部第三方 Agent 自带 MCP Client 的行为。
```

### Trace 写入容错

对应文件：

```text
src/backend/base/langflow/services/tracing/service.py
src/backend/base/langflow/services/tracing/native.py
src/backend/base/langflow/services/database/models/traces/model.py
src/backend/base/langflow/alembic/versions/b7c9d2e4f6a8_add_task_id_to_trace.py
```

实现要求：

```text
1. `trace.task_id` 来自 `graph.context.request_variables["TASK-ID"]`。
2. native tracer 必须优先初始化，保证本地 trace 能落库。
3. LangSmith、Langwatch、Langfuse、Arize Phoenix、Opik、Traceloop、OpenLayer 等外部 tracer 初始化失败时，只记录 debug 日志。
4. 单个外部 tracer 失败不能中断 native tracer，也不能导致 Simple Agent trace 不落库。
```

### Task ID 端到端文件映射

对应文件：

```text
src/lfx/src/lfx/graph/graph/base.py
src/lfx/src/lfx/services/tracing/base.py
src/lfx/src/lfx/services/tracing/service.py
src/backend/base/langflow/services/tracing/service.py
src/backend/base/langflow/services/tracing/native.py
src/backend/base/langflow/services/tracing/repository.py
src/backend/base/langflow/services/database/models/traces/model.py
src/backend/base/langflow/alembic/versions/b7c9d2e4f6a8_add_task_id_to_trace.py
src/backend/base/langflow/api/v1/traces.py
src/frontend/src/controllers/API/queries/traces/helpers.ts
src/frontend/src/controllers/API/queries/traces/types.ts
src/frontend/src/pages/FlowPage/components/TraceComponent/config/flowTraceColumns.tsx
src/frontend/src/pages/FlowPage/components/TraceComponent/types.ts
```

实现要求：

```text
1. Graph 初始化 run 时从 request_variables 中读取 `TASK-ID` 或 `TASK_ID`。
2. lfx tracing service 接口接收 `task_id`，并向 Langflow tracing service 传递。
3. Native trace 创建时写入 `task_id`。
4. repository / API schema / traces API 均暴露 `task_id`，列表搜索和过滤支持该字段。
5. 前端 traces 查询类型、数据转换和表格列都包含 `task_id`。
6. Traces 页面必须把 `Task ID` 放在 `Trace ID` 后面，便于关联同一次 Simple Agent trace 和业务工作流 trace。
```

### stdio + mcp-proxy 请求头限制

当前 stdio 模式通过 `mcp-proxy --headers <key> <value>` 转发 headers。`TASK-ID` / `USER-WALLET-ID` 最好使用不含空格的稳定 ID，例如：

```text
20260525113112
task_20260525_113112
uuid
```

如果 header 值包含空格，Windows shell 参数解析可能把 header 值切成多个 token，导致 `mcp-proxy` 启动参数异常。该限制只影响 stdio + mcp-proxy 形态；Streamable HTTP / SSE 直接传 headers 时没有这个命令行转义风险。
