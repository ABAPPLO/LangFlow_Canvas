# Langflow 工作流锁定最小改动方案

## 目标

当前只做最小范围的 `Lock Flow` 行为增强：

```text
在 Langflow UI 的普通工作流编辑场景中，locked=true 后不允许继续保存画布和节点参数改动。
```

本方案不是完整的后端强制只读方案，也不做多人协同、乐观锁、版本冲突合并。

## 适用范围

本方案只限制 UI 端常规编辑工作流时触发的保存路径：

```text
前端保存工作流 -> PATCH /api/v1/flows/{flow_id}
```

主要覆盖这些操作：

- 修改节点参数后触发保存
- 新增、删除、移动节点后触发保存
- 修改连线后触发保存
- 修改工作流名称、描述等普通编辑信息后触发保存
- 多个页面打开同一个工作流，其中一个页面锁定后，其他旧页面继续保存

## 行为规则

### 未锁定时

`locked=false` 时保持现有行为：

- 可以正常编辑工作流
- 可以自动保存
- 可以更新工作流名称、描述、画布数据等信息

### 已锁定时

`locked=true` 时：

- 允许：只更新 `locked=false`，也就是解锁。
- 拒绝：通过 UI 常规保存路径更新工作流其他字段。

后端返回：

```text
423 Locked
当前工作流已锁定，请先解锁后再编辑。
```

## 已知不覆盖范围

下面两个路径仍然可能更新工作流，但它们不属于本阶段的限制范围。

### 1. PUT upsert 更新路径

接口：

```text
PUT /api/v1/flows/{flow_id}
```

代码位置：

```text
src/backend/base/langflow/api/v1/flows.py
upsert_flow()
_update_existing_flow()
```

触发条件：

- 外部脚本、Postman、curl、Python 程序直接调用 API。
- 后续如果做“测试环境同步到正式环境”的同步平台，可能会使用这个接口保留原工作流 ID。
- 其他后端服务以 API 方式同步或覆盖工作流。

当前前端 UI 触发情况：

```text
当前 Langflow 普通 UI 编辑保存不会触发这个 PUT 接口。
```

影响范围：

- 它可以在 `flow_id` 已存在时更新该工作流。
- 理论上可以更新 `name`、`description`、`data`、`folder_id`、`endpoint_name`、`locked`、`fs_path`、MCP 相关字段等。
- 因为本阶段只要求 UI 端限制，所以暂不对该接口增加锁定校验。

处理结论：

```text
本阶段不限制 PUT upsert。
```

原因：

- 它不是普通 UI 编辑保存路径。
- 它主要服务于 API 同步、导入、跨环境迁移等场景。
- 当前需求只要求 UI 端常规编辑被锁定拦截。

### 2. MCP 工具配置更新路径

接口场景：

```text
项目 MCP Server -> Edit Tools -> 更新工作流作为 MCP 工具时的配置
```

代码位置：

```text
src/backend/base/langflow/api/v1/mcp_projects.py
```

可能更新的字段：

- `mcp_enabled`
- `action_name`
- `action_description`
- `mcp_input_parameters`

触发条件：

- 用户在项目的 MCP Server 页面中修改工具启用状态。
- 用户修改 MCP tool 名称、描述、输入参数配置。
- 外部 API 调用 MCP 项目配置更新接口。

影响范围：

- 不会直接修改画布节点、连线或节点参数。
- 会修改该 workflow 作为 MCP tool 暴露出去时的额外配置。
- 这些字段属于 flow 的扩展配置，而不是普通画布编辑内容。

处理结论：

```text
本阶段不限制 MCP 配置更新。
```

原因：

- MCP 属于 flow 的额外扩展能力。
- 当前需求只要求限制 UI 中普通工作流编辑保存。
- MCP 工具配置不属于本阶段的工作流画布编辑锁定范围。

## 后端改动

文件：

```text
src/backend/base/langflow/api/v1/flows.py
```

位置：

```text
update_flow()
```

逻辑：

```python
update_data = flow.model_dump(exclude_unset=True, exclude_none=True)

if db_flow.locked:
    update_keys = set(update_data)
    is_unlock_only = update_keys == {"locked"} and update_data.get("locked") is False
    if not is_unlock_only:
        raise HTTPException(
            status_code=423,
            detail="当前工作流已锁定，请先解锁后再编辑。",
        )
```

说明：

- 只拦截 `PATCH /api/v1/flows/{flow_id}`。
- 必须允许只传 `locked=false`，否则锁定后无法解锁。
- `423 Locked` 语义最接近当前场景。

## 前端改动

### 1. 解锁时只传 locked 字段

文件：

```text
src/frontend/src/hooks/flows/use-save-flow.ts
```

原因：

锁定状态下，后端只允许解锁请求。如果前端解锁时仍然携带 `name`、`data`、`description` 等字段，会被后端判断为“非解锁更新”并拒绝。

目标行为：

```text
当前保存状态是 locked=true，用户关闭 Lock Flow 时，只发送 { id, locked:false }。
```

### 2. 处理 423 错误提示

文件：

```text
src/frontend/src/hooks/flows/use-save-flow.ts
```

当后端返回 `423` 时，前端提示：

```text
当前工作流已锁定，请先解锁后再编辑。
```

### 3. 423 不重试

文件：

```text
src/frontend/src/controllers/API/queries/flows/use-patch-update-flow.ts
```

原因：

`423` 是明确的业务拒绝，不是网络波动。继续重试没有意义，还会造成重复请求和错误弹窗干扰。

## 验证用例

### 用例 1：未锁定时正常编辑

1. 打开工作流。
2. 确认 `Lock Flow` 关闭。
3. 修改节点参数。
4. 触发保存。
5. 刷新页面。

预期：

```text
修改被保存。
```

### 用例 2：锁定后禁止 UI 保存

1. 打开工作流。
2. 开启 `Lock Flow` 并保存成功。
3. 修改节点参数。
4. 触发保存。
5. 刷新页面。

预期：

```text
保存请求返回 423。
页面提示：当前工作流已锁定，请先解锁后再编辑。
刷新后不保留锁定期间的修改。
```

### 用例 3：锁定后允许解锁

1. 打开已锁定工作流。
2. 关闭 `Lock Flow`。
3. 保存。
4. 刷新页面。

预期：

```text
解锁成功，后续可以正常编辑。
```

### 用例 4：旧页面保存被拒绝

1. A、B 同时打开同一个工作流。
2. A 开启 `Lock Flow` 并保存成功。
3. B 页面不刷新，继续修改节点参数并触发保存。

预期：

```text
B 保存失败，不会覆盖 A 锁定后的工作流。
```

## 结论

本阶段采用最小方案：

```text
只限制 Langflow UI 普通工作流编辑保存路径。
锁定后，PATCH /api/v1/flows/{flow_id} 只允许解锁，不允许保存其他字段。
```

明确不做：

- 不限制 `PUT /api/v1/flows/{flow_id}` upsert。
- 不限制 MCP 工具配置更新。
- 不做乐观锁。
- 不做前端版本冲突提示。
- 不做多人实时协同。

