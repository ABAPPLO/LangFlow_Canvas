# Langflow MCP 多输入参数执行方案

## 目标

让 Langflow Project MCP Server 暴露出的每一个工作流工具，都具备稳定、可读、可执行的 `inputSchema`。

Agent 调用 MCP Tool 时，传入语义化参数；Langflow MCP 执行层按组件 ID 精确注入到目标工作流的多个输入节点。

固定链路：

```text
工作流输入节点
-> Edit Tools 参数配置
-> MCP list_tools inputSchema
-> Agent 工具调用 arguments
-> MCP call_tool 参数映射
-> tweaks[component_id]["input_value"]
-> 目标工作流执行
```

## 相关文件

- 总链路方案：`langflow-agent-backend-e2e-plan.md`
- 请求头透传方案：`langflow-agent-header-forwarding-plan.md`
- Agent 完整返回方案：`langflow-agent-tool-full-response-plan.md`
- 后端调用模拟脚本：`temp_run_agent_workflow.py`

## 当前代码验证结论

当前代码尚未支持本方案描述的多输入能力，缺口集中在两个位置：

```text
src/backend/base/langflow/helpers/flow.py
  json_schema_from_flow()
  当前直接扫描输入节点 template，把可见字段名写入 MCP inputSchema。
  多个 TextInput / ChatInput 会暴露出重复的 input_value，Agent 无法区分业务含义。

src/backend/base/langflow/api/v1/mcp_utils.py
  handle_call_tool()
  当前虽然接收 arguments，但实际只读取 arguments["input_value"]。
  没有把多个语义参数转换成 tweaks[component_id]["input_value"]。
```

可复用的现有能力：

```text
src/backend/base/langflow/api/v1/schemas/__init__.py
  SimplifiedAPIRequest 已支持 tweaks。

src/backend/base/langflow/api/v1/endpoints.py
  simple_run_flow() 已支持 input_request.tweaks，并会调用 process_tweaks()。
```

因此本方案不需要重写工作流执行器，核心是补齐：

```text
工具参数配置存储
-> inputSchema 生成
-> call_tool 参数校验
-> arguments 到 tweaks 的映射
```

## 代码落地方案

### 1. 持久化 MCP 工具输入参数配置

修改文件：

```text
src/backend/base/langflow/services/database/models/flow/model.py
src/backend/base/langflow/alembic/versions/<new_revision>.py
src/backend/base/langflow/api/v1/schemas/__init__.py
src/backend/base/langflow/api/v1/mcp_projects.py
src/frontend/src/types/mcp/index.ts
src/frontend/src/pages/MainPage/pages/homePage/hooks/useMcpServer.ts
src/frontend/src/pages/MainPage/pages/homePage/utils/mcpServerUtils.tsx
src/frontend/src/pages/MainPage/pages/homePage/components/McpFlowsSection.tsx
src/frontend/src/modals/toolsModal/components/toolsTable/index.tsx
```

在 `Flow` 模型增加 JSON 字段：

```python
mcp_input_parameters: list[dict] | None = Field(default=None, sa_column=Column(JSON, nullable=True))
```

数据结构固定为：

```json
[
  {
    "parameter_name": "hot_video_info",
    "parameter_description": "爆款视频链接或解析信息",
    "parameter_type": "string",
    "required": true,
    "component_id": "TextInput-abc",
    "component_display_name": "爆款信息",
    "field": "input_value"
  }
]
```

`MCPSettings`、`MCPProjectUpdateRequest`、`update_project_mcp_settings()` 必须同步支持这个字段，让 Edit Tools 可以读写。

### 2. 生成默认参数配置

修改文件：

```text
src/backend/base/langflow/helpers/flow.py
```

新增一个独立函数：

```python
def generate_default_mcp_input_parameters(flow: Flow) -> list[dict]:
    ...
```

规则：

```text
1. Graph.from_payload(flow.data)。
2. 遍历 vertex.is_input 的节点。
3. 只接受 ChatInput / TextInput。
4. 每个输入节点生成一个参数配置。
5. component_id 使用节点真实 ID。
6. field 固定为 input_value。
7. component_display_name 使用工作流输入节点当前 display_name，仅用于 UI 展示，并随工作流节点名更新。
8. parameter_name 默认使用英文占位名 default_parameter；多个参数自动追加 _2、_3，保持稳定唯一。
9. parameter_description 默认使用固定中文引导文案，不再从工作流节点 description 自动继承。
```

默认配置只用于首次初始化或旧工作流兼容；一旦用户在 Edit Tools 保存过参数配置，后续参数名、参数描述和 required 必须以保存值为准。
工作流输入节点改名时，只同步 `component_display_name`；不要覆盖用户在 Edit Tools 中维护的 `parameter_name` / `parameter_description`。

重要约束：

```text
默认配置只是减少首次配置成本，不是生产语义来源。
生产环境中，MCP inputSchema 必须来自 Edit Tools 保存的 mcp_input_parameters。
不能继续依赖输入节点 template 自动推断业务参数含义。
```

### 3. 改造 MCP inputSchema 生成

修改文件：

```text
src/backend/base/langflow/helpers/flow.py
src/backend/base/langflow/api/v1/mcp_utils.py
```

`json_schema_from_flow(flow)` 改为：

```text
1. 优先读取 flow.mcp_input_parameters。
2. 如果为空，调用 generate_default_mcp_input_parameters(flow) 生成兼容配置。
3. 用参数配置生成 inputSchema.properties。
4. required 只包含 required=true 的参数。
5. 固定 additionalProperties=false。
```

禁止继续把多个输入节点的模板字段直接平铺成 `input_value`。

如果 `flow.mcp_input_parameters` 为空，允许生成临时兼容 schema，但必须在 schema description 或日志中标识这是自动生成配置，提醒用户到 Edit Tools 中保存明确参数配置。

### 4. 改造 MCP call_tool 执行

修改文件：

```text
src/backend/base/langflow/api/v1/mcp_utils.py
```

在 `handle_call_tool()` 内新增转换逻辑：

```python
def build_tweaks_from_mcp_arguments(flow: Flow, arguments: dict) -> dict:
    parameters = flow.mcp_input_parameters or generate_default_mcp_input_parameters(flow)
    validate_required_parameters(parameters, arguments)
    validate_unknown_parameters(parameters, arguments)

    tweaks = {}
    for item in parameters:
        name = item["parameter_name"]
        if name not in arguments:
            continue
        component_id = item["component_id"]
        field = item.get("field") or "input_value"
        tweaks.setdefault(component_id, {})[field] = arguments[name]
    return tweaks
```

然后把当前代码：

```python
input_request = SimplifiedAPIRequest(
    input_value=processed_inputs.get("input_value", ""),
    session_id=conversation_id,
)
```

替换为：

```python
input_request = SimplifiedAPIRequest(
    input_value=None,
    tweaks=tweaks,
    output_type="any",
    session_id=conversation_id,
)
```

这样两个 `TextInput` 或多个 `ChatInput` 都可以按组件 ID 分别注入。

## 唯一执行原则

所有 MCP 工作流工具统一使用“参数配置映射”机制。

不再依赖单一 `input_value` 作为长期执行模型。

兼容旧单输入工作流时，也通过自动生成一条参数配置来执行：

```text
parameter_name -> component_id.input_value
```

这样单输入、多输入、多个同类型输入节点使用同一套机制。

## 支持范围

必须支持以下输入节点：

```text
ChatInput
TextInput
```

必须支持以下工作流形态：

```text
单输入节点
多个输入节点
多个 TextInput
多个 ChatInput
TextInput 和 ChatInput 混合
```

注入方式固定为：

```text
按组件 ID 注入到该节点的 input_value 字段
```

## UI 配置位置

在 `MCP Server Tools -> Edit Tools` 右侧面板中新增：

```text
Input parameters
```

位置固定在：

```text
Tool name
Tool description
Input parameters
```

每个被启用为 MCP Tool 的工作流，都维护自己的 `Input parameters` 配置。

每个参数卡片固定包含三个主要展示/编辑区域：

```text
1. 顶部标题：工作流输入节点 display_name，只做展示，来自目标工作流，并随工作流节点名更新。
2. 参数名输入框：暴露给 Agent 的 MCP schema 参数名，默认 default_parameter / default_parameter_2，不从工作流节点名自动生成。
3. 参数描述输入框：暴露给 Agent 的 MCP schema 参数说明，默认使用引导文案，不从工作流节点 description 自动生成。
```

`Required` 复选框有实际语义：勾选后该参数会进入 MCP `inputSchema.required`，并在 `call_tool` 时参与缺失参数校验；取消勾选则该参数可不传。

注意：`parameter_name` 是 schema key，必须保持非空。前端归一化逻辑会过滤掉缺少 `parameter_name` 或 `component_id` 的参数项，因此把第二个输入框清空会导致该参数项在保存时被移除。

前端实际落点固定为：

```text
src/frontend/src/pages/MainPage/pages/homePage/components/McpFlowsSection.tsx
  MCP Server Tools 入口，继续承载 Edit Tools 按钮。

src/frontend/src/modals/toolsModal/components/toolsTable/index.tsx
  Edit Tools 右侧面板，当前 Tool name / Tool description 就在这里渲染；
  Input parameters 配置区也必须加在这里。

src/frontend/src/pages/MainPage/pages/homePage/hooks/useMcpServer.ts
  保存 MCP settings 时携带 mcp_input_parameters。

src/frontend/src/pages/MainPage/pages/homePage/utils/mcpServerUtils.tsx
  前端 flow/settings 数据映射时携带 mcp_input_parameters。

src/frontend/src/types/mcp/index.ts
  MCPSettingsType 增加 mcp_input_parameters 类型。
```

## 参数配置数据结构

每个参数配置项固定包含：

```json
{
  "parameter_name": "city",
  "parameter_description": "需要查询天气的城市名称",
  "parameter_type": "string",
  "required": true,
  "component_id": "TextInput-a1B2C",
  "component_display_name": "城市",
  "field": "input_value"
}
```

字段说明：

```text
parameter_name
  暴露给 Agent 的参数名，写入 MCP inputSchema.properties。

parameter_description
  暴露给 Agent 的参数说明，写入 MCP inputSchema.properties.<name>.description。

parameter_type
  第一版固定为 string。
  复杂对象先由上游序列化为 JSON 字符串传入。

required
  是否为必填参数，写入 MCP inputSchema.required。

component_id
  目标工作流输入节点 ID。

component_display_name
  仅用于 UI 展示和人工识别。

field
  第一版固定为 input_value。
```

## 默认参数生成规则

第一次打开 `Edit Tools` 或工具参数配置不存在时，后端扫描目标工作流中的输入节点，自动生成参数配置。

扫描节点类型：

```text
ChatInput
TextInput
```

默认生成规则：

```text
parameter_name
  默认使用 default_parameter。
  多个参数自动使用 default_parameter_2、default_parameter_3。
  不再从工作流输入节点 display_name 或 description 推断业务参数名。

parameter_description
  默认使用固定引导文案：
  请填入参数描述，例如说明该参数应传入什么内容、格式要求，以及复杂对象是否需要先序列化为 JSON 字符串。
  不再从工作流输入节点 description 或 display_name 推断业务参数说明。

parameter_type
  string

required
  true

component_id
  输入节点真实组件 ID

component_display_name
  输入节点当前 display_name。
  该字段只用于 UI 展示和人工识别，允许随工作流节点名实时同步。

field
  input_value
```

字段名冲突时，自动追加序号：

```text
city
city_2
city_3
```

## 参数名和参数说明来源

Agent 最终看到的参数名和参数说明只来自 `Edit Tools` 保存后的参数配置。

首次自动生成时使用保守默认值，要求用户在 Edit Tools 中显式维护业务语义。

固定规则：

```text
parameter_name
  默认 default_parameter / default_parameter_2。
  用户保存后以保存值为准。
  只有当保存值仍是 default_parameter 这类默认占位名时，新增节点或重建配置才允许继续补齐默认编号。

parameter_description
  默认固定引导文案。
  用户保存后以保存值为准。
  只有当保存值仍是默认引导文案、空值或旧的工作流派生默认值时，才允许被新的默认引导文案替换。

component_display_name
  始终来自工作流输入节点 display_name。
  只用于 UI 展示，不写入 MCP inputSchema.properties。
```

不通过修改 `TextInput` / `ChatInput` 组件源码来表达业务字段含义。

## MCP inputSchema 生成规则

MCP `list_tools` 返回时，必须使用保存后的 `Input parameters` 生成 `inputSchema`。

示例：

```json
{
  "name": "weather",
  "description": "查询天气工具",
  "inputSchema": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "需要查询天气的城市名称"
      },
      "date": {
        "type": "string",
        "description": "需要查询天气的日期，例如今天、明天、本周末"
      }
    },
    "required": ["city", "date"],
    "additionalProperties": false
  }
}
```

要求：

- `properties` 只包含参数配置中启用的参数。
- `required` 只包含 `required=true` 的参数名。
- `additionalProperties=false`，防止 Agent 传入未定义参数。
- Tool description 使用 `Edit Tools` 中保存的工具描述。

## MCP call_tool 执行规则

Agent 调用工具时传入：

```json
{
  "city": "厦门",
  "date": "今天"
}
```

MCP 执行层按参数配置转换为：

```python
tweaks = {
    "TextInput-a1B2C": {
        "input_value": "厦门"
    },
    "TextInput-d3E4F": {
        "input_value": "今天"
    }
}
```

然后执行目标工作流。

转换规则：

```text
1. 读取当前 Tool 的 Input parameters 配置。
2. 校验 required 参数是否全部存在。
3. 拒绝 inputSchema 中不存在的参数。
4. 将 arguments[parameter_name] 写入对应 component_id 的 input_value。
5. 生成 tweaks 后调用目标工作流。
```

## 完整示例

目标工作流有两个输入节点：

```text
TextInput-a1B2C
display_name: 城市
description: 需要查询天气的城市名称

TextInput-d3E4F
display_name: 日期
description: 需要查询天气的日期，例如今天、明天、本周末
```

Edit Tools 保存的参数配置：

```json
[
  {
    "parameter_name": "city",
    "parameter_description": "需要查询天气的城市名称",
    "parameter_type": "string",
    "required": true,
    "component_id": "TextInput-a1B2C",
    "component_display_name": "城市",
    "field": "input_value"
  },
  {
    "parameter_name": "date",
    "parameter_description": "需要查询天气的日期，例如今天、明天、本周末",
    "parameter_type": "string",
    "required": true,
    "component_id": "TextInput-d3E4F",
    "component_display_name": "日期",
    "field": "input_value"
  }
]
```

MCP `list_tools` 返回：

```json
{
  "name": "weather",
  "description": "查询天气工具",
  "inputSchema": {
    "type": "object",
    "properties": {
      "city": {
        "type": "string",
        "description": "需要查询天气的城市名称"
      },
      "date": {
        "type": "string",
        "description": "需要查询天气的日期，例如今天、明天、本周末"
      }
    },
    "required": ["city", "date"],
    "additionalProperties": false
  }
}
```

用户向 Agent 提问：

```text
查一下厦门今天的天气
```

Agent 调用工具：

```json
{
  "city": "厦门",
  "date": "今天"
}
```

Langflow 内部注入：

```python
tweaks = {
    "TextInput-a1B2C": {
        "input_value": "厦门"
    },
    "TextInput-d3E4F": {
        "input_value": "今天"
    }
}
```

## 错误处理

### 缺少必填参数

如果 Agent 未传入必填参数：

```json
{
  "city": "厦门"
}
```

而 `date` 是必填，则 MCP Tool 返回结构化错误：

```json
{
  "success": false,
  "error": {
    "code": "missing_required_parameter",
    "message": "Missing required parameter: date"
  }
}
```

### 未定义参数

如果 Agent 传入 inputSchema 中不存在的字段：

```json
{
  "city": "厦门",
  "date": "今天",
  "extra": "unknown"
}
```

MCP Tool 返回结构化错误：

```json
{
  "success": false,
  "error": {
    "code": "unknown_parameter",
    "message": "Unknown parameter: extra"
  }
}
```

## 实施步骤

1. 在 `Flow` 模型增加 `mcp_input_parameters` 字段，并创建 Alembic 迁移。
2. 在 `MCPSettings` / `MCPProjectUpdateRequest` 中增加 `mcp_input_parameters`。
3. 在 `update_project_mcp_settings()` 中保存每个工具的参数配置。
4. `Edit Tools` 右侧面板新增 `Input parameters` 配置区。
5. 首次打开时扫描 `ChatInput` / `TextInput` 自动生成默认配置。
6. `json_schema_from_flow()` 使用保存后的参数配置生成 inputSchema。
7. `handle_call_tool()` 使用参数配置把 arguments 转成 tweaks。
8. 增加 required 校验和 unknown parameter 校验。
9. 增加多输入节点、同类型多输入节点、混合输入节点测试。

## 验收标准

- Agent 能看到业务语义明确的 inputSchema。
- 两个 `TextInput` 可以分别收到不同参数。
- `TextInput` 与 `ChatInput` 混合时可以分别注入。
- 参数描述来自 Edit Tools 保存值。
- 默认参数名使用 `default_parameter` / `default_parameter_2`，不从工作流节点名自动生成。
- 默认参数描述使用固定引导文案，不从工作流节点 description 自动生成。
- 工作流输入节点 display_name 变化后，Edit Tools 顶部展示名同步更新。
- 用户保存后的参数名、参数描述和 required 不会被工作流节点信息覆盖。
- 缺少必填参数时不执行目标工作流。
- 未定义参数不会静默进入目标工作流。
- 清空 parameter_name 后该参数项会被前端归一化过滤，不能作为有效配置保存。
- 单输入工作流也走同一套参数配置映射机制。

## 2026-05-25 实现同步补充

本节记录方案落地后新增的硬约束，确保文档与代码一比一对应。

### inputSchema 必填字符串约束

对应文件：

```text
src/backend/base/langflow/alembic/versions/8f3c2d4e9a10_add_mcp_input_parameters_to_flow.py
src/backend/base/langflow/services/database/models/flow/model.py
src/backend/base/langflow/api/v1/schemas/__init__.py
src/backend/base/langflow/api/v1/mcp_projects.py
src/backend/base/langflow/helpers/flow.py
src/backend/base/langflow/api/v1/mcp_utils.py
```

实现要求：

```text
1. `json_schema_from_flow()` 基于 Edit Tools 保存的 `mcp_input_parameters` 生成 schema。
2. 对 `required=true` 的 string 参数，`inputSchema.properties.<name>` 必须包含 `minLength: 1`。
3. `inputSchema.required` 只包含 `required=true` 的参数名。
4. `additionalProperties=false` 保持开启，拒绝 schema 外参数。
```

### MCP Server 服务端二次校验

仅依赖 `inputSchema` 不够，因为不同 Agent / MCP Client 对 JSON Schema 的执行严格度不同。Project MCP Server 必须在 `build_tweaks_from_mcp_arguments()` 中做最终硬拦截：

```text
1. required 参数缺失或值为 null -> `missing_required_parameter`。
2. required string 参数为空字符串或纯空白字符串 -> `empty_required_parameter`。
3. arguments 中出现 Edit Tools 未保存的参数名 -> `unknown_parameter`。
4. 参数值不是 string -> `invalid_parameter_type`，调用方必须先把复杂对象序列化为 JSON 字符串。
5. 只有通过校验的参数才会写入 `tweaks[component_id]["input_value"]`。
```

这层校验对所有调用方生效：Simple Agent、外部 MCP Client、脚本直接调用 Project MCP Server 都不能绕过。

### Edit Tools 默认值与保存值边界

对应文件：

```text
src/backend/base/langflow/helpers/flow.py
src/frontend/src/types/mcp/index.ts
src/frontend/src/pages/MainPage/pages/homePage/hooks/useMcpServer.ts
src/frontend/src/pages/MainPage/pages/homePage/utils/mcpServerUtils.tsx
src/frontend/src/pages/MainPage/pages/homePage/components/McpFlowsSection.tsx
src/frontend/src/modals/toolsModal/components/toolsTable/index.tsx
```

实现要求：

```text
1. 输入节点展示名只作为参数卡片顶部标题，用于识别节点。
2. 参数名默认使用 `default_parameter`、`default_parameter_2` 等英文占位名。
3. 参数描述默认使用中文引导文案，提醒用户说明内容、格式和 JSON 字符串序列化要求。
4. 用户保存后的 `parameter_name`、`parameter_description`、`required` 以保存值为准。
5. 工作流节点改名后只同步 `component_display_name`，不覆盖用户维护的 schema 参数名和描述。
```
