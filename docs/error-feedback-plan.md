# Langflow 错误反馈优化计划

## 目标
让工作流执行失败时，用户能在前端（节点 tooltip、底部状态栏、toast 通知）看到准确、可操作的错误信息。

## 问题统计
- **93 个组件文件** 中存在 **186 个吞掉异常** 的 except 块（catch 后不 raise）
- **29+ 个组件** 使用通用错误消息（"An error occurred"、"An unexpected error occurred"）
- 前端 tooltip 只展示 `errorMessage`，`stackTrace` 已传到前端但未显示
- 底部状态栏用 `truncate-doubleline` 截断了多行错误

---

## P0：修复吞掉异常的组件（核心问题）

### 原则
将 `except ... return Data(data={"error": ...})` 改为 `except ... raise`。
让异常自然传播到 `Component.build_results()` → `send_error()` → SSE → 前端。

### 按模块分组修复（共 93 个文件）

#### 第 1 批：JigsawStack 组件（11 个文件，统一模式）
统一模式：`except JigsawStackError` → `self.status = ...` → `return Data(data={"error": ...})`
改为：`raise RuntimeError(f"JigsawStack {operation} failed: {e}") from e`

| 文件 | 吞掉异常数 |
|------|-----------|
| jigsawstack/ai_scrape.py | 1 |
| jigsawstack/ai_web_search.py | 2 |
| jigsawstack/file_read.py | 2 |
| jigsawstack/file_upload.py | 1 |
| jigsawstack/image_generation.py | 1 |
| jigsawstack/nsfw.py | 1 |
| jigsawstack/object_detection.py | 1 |
| jigsawstack/sentiment.py | 2 |
| jigsawstack/text_to_sql.py | 1 |
| jigsawstack/text_translate.py | 1 |
| jigsawstack/vocr.py | 1 |

#### 第 2 批：Notion 组件（5 个文件，13 个吞掉块）
统一模式：多层 catch（JSON → Request → Exception），全部 return error_string

| 文件 | 吞掉异常数 |
|------|-----------|
| Notion/add_content_to_page.py | 2 |
| Notion/create_page.py | 2 |
| Notion/list_database_properties.py | 3 |
| Notion/list_pages.py | 4 |
| Notion/page_content_viewer.py | 2 |
| Notion/update_page_property.py | 4 |

#### 第 3 批：AssemblyAI 组件（5 个文件）

| 文件 | 吞掉异常数 |
|------|-----------|
| assemblyai/assemblyai_get_subtitles.py | 2 |
| assemblyai/assemblyai_lemur.py | 7 |
| assemblyai/assemblyai_list_transcripts.py | 1 |
| assemblyai/assemblyai_poll_transcript.py | 1 |
| assemblyai/assemblyai_start_transcript.py | 1 |

#### 第 4 批：数据源组件（7 个文件）

| 文件 | 吞掉异常数 |
|------|-----------|
| data_source/news_search.py | 2 |
| data_source/rss.py | 1 |
| data_source/web_search.py | 6 |
| data_source/sql_executor.py | 3 |
| data_source/csv_to_data.py | 1 |
| data_source/json_to_data.py | 1 |
| data_source/mock_data.py | 10 |

#### 第 5 批：YouTube 组件（5 个文件）

| 文件 | 吞掉异常数 |
|------|-----------|
| youtube/channel.py | 2 |
| youtube/comments.py | 1 |
| youtube/search.py | 1 |
| youtube/trending.py | 2 |
| youtube/video_details.py | 2 |
| youtube/youtube_transcripts.py | 3 |

#### 第 6 批：工具与搜索组件

| 文件 | 吞掉异常数 |
|------|-----------|
| tools/calculator.py | 3 |
| tools/serp_api.py | 1 |
| tools/searxng.py | 1 |
| tools/tavily_search_tool.py | 3 |
| duckduckgo/duck_duck_go_search_run.py | 1 |
| google/google_search_api_core.py | 3 |
| google/google_serper_api_core.py | 1 |
| arxiv/arxiv.py | 1 |

#### 第 7 批：Volcengine / TwelveLabs / VLMRun 组件

| 文件 | 吞掉异常数 |
|------|-----------|
| volcengine/seedance.py | 2 |
| volcengine/seedream.py | 2 |
| twelvelabs/twelvelabs_pegasus.py | 5 |
| twelvelabs/video_file.py | 3 |
| vlmrun/vlmrun_transcription.py | 3 |

#### 第 8 批：其余散落组件

| 文件 | 吞掉异常数 |
|------|-----------|
| files_and_knowledge/file.py | 7 |
| files_and_knowledge/retrieval.py | 2 |
| git/gitextractor.py | 5 |
| git/git.py | 4 |
| processing/regex.py | 2 |
| processing/data_operations.py | 2 |
| processing/dynamic_create_data.py | 1 |
| processing/converter.py | 1 |
| processing/create_data.py | 1 |
| processing/update_data.py | 1 |
| processing/text_operations.py | 1 |
| processing/message_to_data.py | 1 |
| flow_controls/conditional_router.py | 2 |
| flow_controls/data_conditional_router.py | 1 |
| utilities/calculator_core.py | 2 |
| utilities/python_repl_core.py | 3 |
| utilities/current_date.py | 1 |
| prototypes/python_function.py | 1 |
| embeddings/text_embedder.py | 1 |
| llm_operations/structured_output.py | 1 |
| llm_operations/batch_run.py | 1 |
| llm_operations/llm_selector.py | 2 |
| models_and_agents/multimodal_model.py | 1 |
| models_and_agents/agent.py | 5 |
| models_and_agents/mcp_component.py | 2 |
| deepseek/deepseek.py | 1 |
| groq/groq.py | 1 |
| cometapi/cometapi.py | 2 |
| novita/novita.py | 1 |
| xai/xai.py | 2 |
| lmstudio/lmstudiomodel.py | 2 |
| ollama/ollama.py | 1 |
| ollama/ollama_embeddings.py | 1 |
| openai/openai_chat_model.py | 1 |
| openrouter/openrouter.py | 1 |
| vllm/vllm.py | 1 |
| litellm/litellm_proxy.py | 1 |
| tavily/tavily_extract.py | 3 |
| tavily/tavily_search.py | 4 |
| langwatch/langwatch.py | 3 |
| homeassistant/home_assistant_control.py | 1 |
| homeassistant/list_home_assistant_states.py | 3 |
| elastic/opensearch.py | 2 |
| elastic/opensearch_multimodal.py | 4 |
| elastic/elasticsearch.py | 1 |
| video/video_concat.py | 2 |
| langchain_utilities/ibm_granite_handler.py | 1 |
| nvidia/system_assist.py | 1 |
| olivya/olivya.py | 1 |
| datastax/astradb_vectorstore.py | 1 |

---

## P1：前端 tooltip 展示 stackTrace + 规范化错误消息

### 前端改动（3 个文件）
1. **`use-validation-status-string.ts`** — 提取 `stackTrace` 并追加到展示字符串
2. **`build-status-display.tsx`** — 添加可折叠区域展示 stackTrace
3. **`flowBuildingComponent/index.tsx`** — 移除 `truncate-doubleline`，添加展开/收起按钮

### 后端改动（~29 个文件）
将 "An error occurred"、"An unexpected error occurred" 替换为包含上下文的具体消息。

---

## P2：输入校验前置 + 异常类型语义化

### 组件改动
- 在执行耗时操作前校验 API key、URL、文件路径等必填参数
- 使用 `ValueError`（配置错误）、`ConnectionError`（网络错误）、`RuntimeError`（运行时错误）等语义化异常
- 所有 `raise ... from e` 保留异常链

---

## P3：前端错误分类展示（可选）

### 前端改动
- 根据 SSE 中的异常类型字段区分错误类别
- tooltip 中用不同颜色/图标标识：配置错误（黄）、连接错误（红+重试）、运行时错误（红）
- 底部状态栏增加"查看详情"展开完整 stackTrace
