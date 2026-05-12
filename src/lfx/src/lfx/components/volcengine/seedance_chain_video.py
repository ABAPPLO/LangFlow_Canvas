# ruff: noqa: BLE001, EM101, EM102, PERF401, PLR2004, RUF001, RUF003, S110, SIM105, TRY003, TRY004

import json
import time
from urllib.parse import urlparse

import httpx

from lfx.custom import Component
from lfx.inputs import (
    BoolInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    MultilineInput,
    SecretStrInput,
)
from lfx.inputs.inputs import StrInput
from lfx.io import Output
from lfx.schema.data import Data
from lfx.schema.message import Message

BASE_URL = "http://localhost:3000/v1/videos"
DEFAULT_API_KEY = ""

MAX_REF_IMAGES = 9
MAX_SEGMENTS = 20
DEFAULT_REQUEST_TIMEOUT = 300
MAX_CREATE_TASK_RETRIES = 3
MAX_POLL_TRANSIENT_ERRORS = 5
MAX_RETRY_DELAY = 30

# 任务终态：含 cancelled，避免主动取消的任务被轮询到超时
TERMINAL_FAIL_STATUSES = ("failed", "error", "expired", "cancelled")


class TaskError(Exception):
    pass


class TaskTimeoutError(Exception):
    pass


class SeedanceChainVideoComponent(Component):
    display_name = "Seedance 连续视频"
    description = "使用火山引擎 Seedance 2.0 API 生成多个连续视频，通过多张主体参考图和参考视频保持主体一致性。"
    icon = "Video"
    name = "SeedanceChainVideo"

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="火山引擎 API Key。",
            required=True,
            value=DEFAULT_API_KEY,
        ),
        StrInput(
            name="user_wallet_id",
            display_name="User Wallet ID",
            info="Passed as User-Wallet-Id header to NewAPI",
            show=True,
            advanced=True,
        ),
        StrInput(
            name="task_id",
            display_name="Task ID",
            info="Passed as Task-Id header to NewAPI",
            show=True,
            advanced=True,
        ),
        MessageTextInput(
            name="base_url",
            display_name="Base URL",
            info="API 基础地址，留空使用默认地址。",
            value=BASE_URL,
            advanced=True,
        ),
        MessageTextInput(
            name="model",
            display_name="模型",
            info="模型ID，如 doubao-seedance-2-0-260128。",
            value="doubao-seedance-2-0-260128",
        ),
        MultilineInput(
            name="prompts",
            display_name="提示词列表",
            info='JSON 数组格式，每个提示词生成一个视频段落。如 ["提示词1", "提示词2"]。',
            required=True,
        ),
        MultilineInput(
            name="subject_reference_urls",
            display_name="主体参考图",
            info='可选。主体参考图片URL，JSON 数组格式，支持同一主体多角度参考，最多9张。如 ["https://xxx/front.jpg", "https://xxx/side.jpg"]。不提供则使用纯文生视频。',
            value="",
        ),
        DropdownInput(
            name="resolution",
            display_name="分辨率",
            info="输出视频分辨率。",
            options=["720p", "480p", "1080p"],
            value="720p",
            advanced=True,
        ),
        DropdownInput(
            name="ratio",
            display_name="宽高比",
            info="输出视频宽高比。",
            options=["adaptive", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
            value="adaptive",
            advanced=True,
        ),
        BoolInput(
            name="generate_audio",
            display_name="生成音频",
            info="为视频生成同步音频。",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="watermark",
            display_name="水印",
            info="是否添加水印。",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="trust_env",
            display_name="使用系统代理",
            info="开启后使用环境变量中的 HTTP/HTTPS 代理设置。需要走公司代理出网时打开。",
            value=False,
            advanced=True,
        ),
        IntInput(
            name="poll_interval",
            display_name="轮询间隔 (秒)",
            info="状态查询间隔秒数。",
            value=5,
            advanced=True,
        ),
        IntInput(
            name="max_wait_time",
            display_name="最大等待时间 (秒)",
            info="单个任务最大等待秒数。",
            value=1800,
            advanced=True,
        ),
        IntInput(
            name="request_timeout",
            display_name="请求读取超时 (秒)",
            info="单次 HTTP 请求等待响应的最大秒数。创建任务和查询任务都会使用它。",
            value=DEFAULT_REQUEST_TIMEOUT,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="所有视频URL",
            name="video_urls",
            method="get_video_urls",
            type_=Message,
            group_outputs=True,
        ),
        Output(
            display_name="任务信息",
            name="task_info",
            method="get_task_info",
            type_=Data,
            group_outputs=True,
        ),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._results: list[dict] = []
        self._generation_done: bool = False
        self._generation_error: Exception | None = None
        self._failure_info: dict | None = None

    def _debug_log(self, message: str, name: str = "Seedance") -> None:
        timestamp = time.strftime("%H:%M:%S")
        self.log(f"[{timestamp}] {message}", name=name)

    def _short_text(self, value: str, limit: int = 240) -> str:
        text = str(value or "")
        return text[:limit] + ("..." if len(text) > limit else "")

    def _stage_name(self, stage: str) -> str:
        names = {
            "parse_prompts": "解析提示词",
            "parse_reference_images": "解析主体参考图",
            "create_task": "提交生成任务",
            "poll_task": "查询任务状态",
            "extract_video_url": "解析视频 URL",
            "client": "客户端请求",
        }
        return names.get(stage, stage)

    def _set_failure(
        self,
        *,
        stage: str,
        error: Exception,
        segment: int | None = None,
        total_segments: int | None = None,
        task_id: str = "",
        prompt: str = "",
        ref_video_url: str = "",
        extra: dict | None = None,
    ) -> dict:
        details = {
            "stage": stage,
            "stage_name": self._stage_name(stage),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        if segment is not None:
            details["segment"] = segment
        if total_segments is not None:
            details["total_segments"] = total_segments
        if task_id:
            details["task_id"] = task_id
        if prompt:
            details["prompt_preview"] = self._short_text(prompt)
            details["prompt_chars"] = len(prompt)
        if ref_video_url:
            parsed_ref_video = urlparse(ref_video_url)
            details["ref_video_domain"] = parsed_ref_video.netloc
            details["ref_video_url_chars"] = len(ref_video_url)
        if extra:
            details.update(extra)

        self._failure_info = details
        segment_label = f"第 {segment}/{total_segments} 段" if segment and total_segments else "全局"
        task_label = f", task_id={task_id[:12]}..." if task_id else ""
        self.log(
            f"失败位置: {segment_label} / {self._stage_name(stage)}{task_label}; "
            f"错误类型: {type(error).__name__}; 原因: {error}",
            "ERROR",
        )
        self._debug_log(
            f"Failure details: stage={stage}, segment={segment or 'none'}, total={total_segments or 'none'}, "
            f"task_id={task_id or 'none'}, error={type(error).__name__}: {error}",
            name="ERROR",
        )
        return details

    # ---------- 输入解析 ----------

    def _parse_prompts(self) -> list[str]:
        raw = self.prompts or ""
        if isinstance(raw, Message):
            raw = raw.get_text()
        parsed = json.loads(str(raw).strip())
        if not isinstance(parsed, list):
            raise ValueError('提示词输入必须是 JSON 数组格式，如 ["提示词1", "提示词2"]')
        result = [str(item).strip() for item in parsed if str(item).strip()]
        if len(result) > MAX_SEGMENTS:
            raise ValueError(f"提示词数量不能超过 {MAX_SEGMENTS} 个，当前: {len(result)}")
        return result

    def _parse_ref_urls(self) -> list[str]:
        raw = self.subject_reference_urls or ""
        if isinstance(raw, Message):
            raw = raw.get_text()
        raw = str(raw).strip()
        if not raw:
            return []
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError('参考图输入必须是 JSON 数组格式，如 ["url1", "url2"]')
        result = [str(u).strip() for u in parsed if str(u).strip()]
        if len(result) > MAX_REF_IMAGES:
            raise ValueError(f"参考图数量不能超过 {MAX_REF_IMAGES} 张，当前: {len(result)}")
        return result

    def _get_request_variable(self, request_vars, *keys: str) -> str:
        if not request_vars or not hasattr(request_vars, "get"):
            return ""
        for key in keys:
            value = request_vars.get(key)
            if value:
                return str(value).strip()
        return ""

    def _resolve_tracking_headers(self) -> tuple[str, str]:
        wallet_id = str(getattr(self, "user_wallet_id", "") or "").strip()
        tracking_task_id = str(getattr(self, "task_id", "") or "").strip()

        request_vars = None
        if hasattr(self, "graph") and self.graph and hasattr(self.graph, "context"):
            request_vars = self.graph.context.get("request_variables")

        if not wallet_id:
            wallet_id = self._get_request_variable(
                request_vars,
                "USER-WALLET-ID",
                "USER_WALLET_ID",
                "User-Wallet-Id",
                "user_wallet_id",
            )
        if not tracking_task_id:
            tracking_task_id = self._get_request_variable(
                request_vars,
                "TASK-ID",
                "TASK_ID",
                "Task-Id",
                "task_id",
            )

        self.log(f"user_wallet_id={wallet_id}, task_id={tracking_task_id}")
        return wallet_id, tracking_task_id

    # ---------- 提示词构造 ----------

    def _build_ref_prefix(self, segment_num: int) -> str:
        # 第 1 段无参考视频，无需前缀
        if segment_num == 1:
            return ""
        # 第 N 段的 reference_video 实际是上一段视频，因此用"上一个视频"而非写死的"视频1"
        return "向后延长上一个视频。"

    def _build_content(
        self,
        prompt: str,
        ref_image_urls: list[str],
        ref_video_url: str = "",
    ) -> list[dict]:
        content: list[dict] = [{"type": "text", "text": prompt}]

        for url in ref_image_urls:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                    "role": "reference_image",
                }
            )

        if ref_video_url:
            content.append(
                {
                    "type": "video_url",
                    "video_url": {"url": ref_video_url},
                    "role": "reference_video",
                }
            )

        return content

    def _build_request_body(
        self,
        prompt: str,
        ref_image_urls: list[str],
        ref_video_url: str = "",
    ) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "metadata": {
                "content": self._build_content(prompt, ref_image_urls, ref_video_url),
                "resolution": self.resolution,
                "ratio": self.ratio,
                "duration": -1,
                "generate_audio": self.generate_audio,
                "watermark": self.watermark,
            },
        }

    # ---------- HTTP 调用 ----------

    def _http_timeout(self) -> httpx.Timeout:
        read_timeout = int(getattr(self, "request_timeout", DEFAULT_REQUEST_TIMEOUT) or DEFAULT_REQUEST_TIMEOUT)
        read_timeout = max(read_timeout, 30)
        return httpx.Timeout(connect=10, read=read_timeout, write=60, pool=10)

    def _is_retryable_status_code(self, code: int) -> bool:
        return code == 429 or 500 <= code < 600

    def _retry_delay(self, retry_count: int) -> int:
        return min(2**retry_count, MAX_RETRY_DELAY)

    def _create_task(
        self,
        client: httpx.Client,
        prompt: str,
        ref_image_urls: list[str],
        ref_video_url: str = "",
    ) -> str:
        payload = self._build_request_body(prompt, ref_image_urls, ref_video_url)
        max_attempts = MAX_CREATE_TASK_RETRIES + 1

        for attempt in range(1, max_attempts + 1):
            start = time.perf_counter()
            self._debug_log(
                f"Create task request: model={self.model}, prompt_chars={len(prompt)}, "
                f"ref_images={len(ref_image_urls)}, has_ref_video={bool(ref_video_url)}, "
                f"attempt={attempt}/{max_attempts}"
            )
            try:
                resp = client.post(self.base_url, json=payload)
                elapsed = time.perf_counter() - start
                self._debug_log(f"Create task response: status_code={resp.status_code}, elapsed={elapsed:.1f}s")
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                elapsed = time.perf_counter() - start
                code = e.response.status_code
                detail = ""
                try:
                    detail = e.response.text
                except Exception:
                    pass
                if self._is_retryable_status_code(code) and attempt < max_attempts:
                    delay = self._retry_delay(attempt)
                    self._debug_log(
                        f"Create task transient HTTP error: status_code={code}, elapsed={elapsed:.1f}s, "
                        f"retry={attempt}/{MAX_CREATE_TASK_RETRIES}, sleep={delay}s",
                        name="WARNING",
                    )
                    self.log(f"Create task HTTP {code}, retrying in {delay}s...", "WARNING")
                    time.sleep(delay)
                    continue

                error_msg = f"API error {code}: {detail}"
                self._debug_log(f"Create task HTTP error: status_code={code}, elapsed={elapsed:.1f}s", name="ERROR")
                self.log(error_msg, "ERROR")
                raise TaskError(error_msg) from e
            except (httpx.TimeoutException, httpx.TransportError) as e:
                elapsed = time.perf_counter() - start
                if attempt < max_attempts:
                    delay = self._retry_delay(attempt)
                    self._debug_log(
                        f"Create task transport error: error={type(e).__name__}: {e}, elapsed={elapsed:.1f}s, "
                        f"retry={attempt}/{MAX_CREATE_TASK_RETRIES}, sleep={delay}s, "
                        f"timeout={getattr(self, 'request_timeout', DEFAULT_REQUEST_TIMEOUT)}s",
                        name="WARNING",
                    )
                    self.log(f"Create task {type(e).__name__}, retrying in {delay}s...", "WARNING")
                    time.sleep(delay)
                    continue

                error_msg = (
                    "Create task failed after retries while waiting for response. "
                    f"Last error: {type(e).__name__}: {e}. "
                    "The server may still be processing one of the attempts."
                )
                self._debug_log(
                    f"Create task failed after retries: error={type(e).__name__}: {e}, elapsed={elapsed:.1f}s",
                    name="ERROR",
                )
                self.log(error_msg, "ERROR")
                raise TaskError(error_msg) from e
            else:
                data = resp.json()
                task_id = data.get("id", "")
                if not task_id:
                    raise TaskError(f"No task ID in response: {data}")
                self._debug_log(f"Create task success: task_id={task_id}")
                return task_id

        raise TaskError("Create task failed without response")

    def _poll_task(self, client: httpx.Client, task_id: str) -> dict:
        url = f"{self.base_url}/{task_id}"
        deadline = time.time() + self.max_wait_time
        consecutive_errors = 0
        max_consecutive_errors = MAX_POLL_TRANSIENT_ERRORS
        attempt = 0

        while time.time() < deadline:
            attempt += 1
            try:
                poll_start = time.perf_counter()
                self.log(f"Polling task {task_id[:12]}... (attempt {attempt})")
                self._debug_log(f"Poll task request: task_id={task_id[:12]}, attempt={attempt}")
                resp = client.get(url)
                elapsed = time.perf_counter() - poll_start
                self._debug_log(f"Poll task response: status_code={resp.status_code}, elapsed={elapsed:.1f}s")
                resp.raise_for_status()
                data = resp.json()
                consecutive_errors = 0

                status = data.get("status", "unknown")
                self.status = f"段落任务 {task_id[:12]}... 状态: {status}"
                self.log(f"Task {task_id[:12]}... status: {status}")
                self._debug_log(f"Poll task status: task_id={task_id[:12]}, status={status}")

                if status in ("succeeded", "completed"):
                    return data
                if status in TERMINAL_FAIL_STATUSES:
                    error_msg = data.get("error", {})
                    raise TaskError(f"Task {task_id[:12]}... ended with status={status}, error={error_msg}")

                time.sleep(self.poll_interval)

            except TaskError:
                raise
            except httpx.ReadTimeout as e:
                elapsed = time.perf_counter() - poll_start
                consecutive_errors += 1
                self._debug_log(
                    f"Poll read timeout: task_id={task_id[:12]}, attempt={attempt}, elapsed={elapsed:.1f}s, "
                    f"consecutive_errors={consecutive_errors}/{max_consecutive_errors}, "
                    f"timeout={getattr(self, 'request_timeout', DEFAULT_REQUEST_TIMEOUT)}s",
                    name="WARNING",
                )
                self.log(
                    f"Polling read timeout, retrying... ({consecutive_errors}/{max_consecutive_errors})",
                    "WARNING",
                )
                if consecutive_errors >= max_consecutive_errors:
                    raise TaskError(f"Too many read timeouts polling task {task_id[:12]}...") from e
                time.sleep(self._retry_delay(consecutive_errors))
            except httpx.HTTPStatusError as e:
                elapsed = time.perf_counter() - poll_start
                # 400/401 等参数或鉴权错误不会自愈；429 与 5xx 按瞬时错误重试
                code = e.response.status_code
                if not self._is_retryable_status_code(code):
                    detail = ""
                    try:
                        detail = e.response.text
                    except Exception:
                        pass
                    self._debug_log(
                        f"Poll task fatal HTTP error: task_id={task_id[:12]}, status_code={code}, "
                        f"elapsed={elapsed:.1f}s",
                        name="ERROR",
                    )
                    raise TaskError(f"Polling task {task_id[:12]}... HTTP {code}: {detail}") from e
                consecutive_errors += 1
                self._debug_log(
                    f"Poll task transient HTTP error: task_id={task_id[:12]}, status_code={code}, "
                    f"elapsed={elapsed:.1f}s, consecutive_errors={consecutive_errors}/{max_consecutive_errors}",
                    name="WARNING",
                )
                self.log(f"HTTP {code} (transient), retry...", "WARNING")
                if consecutive_errors >= max_consecutive_errors:
                    raise TaskError(f"Too many transient HTTP errors polling task {task_id[:12]}...") from e
                time.sleep(self._retry_delay(consecutive_errors))
            except (httpx.HTTPError, ValueError, KeyError) as e:
                elapsed = time.perf_counter() - poll_start
                consecutive_errors += 1
                self._debug_log(
                    f"Poll task error: task_id={task_id[:12]}, error={type(e).__name__}: {e}, "
                    f"elapsed={elapsed:.1f}s, consecutive_errors={consecutive_errors}/{max_consecutive_errors}",
                    name="WARNING",
                )
                self.log(f"Polling error: {e}", "WARNING")
                if consecutive_errors >= max_consecutive_errors:
                    raise TaskError(f"Too many errors polling task {task_id[:12]}...") from e
                time.sleep(self._retry_delay(consecutive_errors))

        raise TaskTimeoutError(f"Task {task_id[:12]}... timed out after {self.max_wait_time}s")

    def _extract_video_url(self, data: dict) -> str:
        metadata = data.get("metadata")
        if isinstance(metadata, dict):
            url = metadata.get("url") or metadata.get("video_url")
            if url:
                return url
        url = data.get("url") or data.get("video_url")
        if url:
            return url
        wrapped = data.get("data")
        if isinstance(wrapped, dict):
            url = self._extract_video_url(wrapped)
            if url:
                return url
        content = data.get("content")
        if isinstance(content, dict):
            return content.get("video_url", "")
        # 文档若返回 list，取第一个 video_url 字段
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("video_url"):
                    return item["video_url"]
        return ""

    # ---------- 主流程 ----------

    def _ensure_generated(self, *, raise_on_error: bool = True):
        # 已成功完成过 -> 直接复用
        if self._generation_done and self._generation_error is None:
            return
        # 上次失败 -> 抛出原错误，由调用方决定是否重试（重建实例或显式重置）
        if self._generation_error is not None:
            if raise_on_error:
                raise self._generation_error
            return

        try:
            prompts = self._parse_prompts()
        except (json.JSONDecodeError, ValueError) as e:
            self._generation_error = e
            self._generation_done = True
            self._set_failure(stage="parse_prompts", error=e)
            if raise_on_error:
                raise
            return

        if not prompts:
            self._results = []
            self._generation_done = True
            return

        try:
            ref_image_urls = self._parse_ref_urls()
        except (json.JSONDecodeError, ValueError) as e:
            self._generation_error = e
            self._generation_done = True
            self._set_failure(stage="parse_reference_images", error=e, total_segments=len(prompts))
            if raise_on_error:
                raise
            return

        self._debug_log(
            f"Start chain generation: segments={len(prompts)}, ref_images={len(ref_image_urls)}, "
            f"request_timeout={getattr(self, 'request_timeout', DEFAULT_REQUEST_TIMEOUT)}, "
            f"max_wait_time={self.max_wait_time}, poll_interval={self.poll_interval}"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        wallet_id, tracking_task_id = self._resolve_tracking_headers()
        if wallet_id:
            headers["User-Wallet-Id"] = wallet_id
        if tracking_task_id:
            headers["Task-Id"] = tracking_task_id

        # 重置结果，避免重入时残留
        self._results = []
        self._failure_info = None

        try:
            with httpx.Client(
                headers=headers,
                timeout=self._http_timeout(),
                trust_env=bool(self.trust_env),
            ) as client:
                prev_video_url = ""

                for i, prompt in enumerate(prompts):
                    segment_num = i + 1
                    self.status = f"正在生成第 {segment_num}/{len(prompts)} 个视频..."
                    self._debug_log(f"Segment start: segment={segment_num}/{len(prompts)}")

                    prefix = self._build_ref_prefix(segment_num)
                    final_prompt = prefix + prompt

                    self.log(f"=== 段落 {segment_num}/{len(prompts)} ===")
                    self.log(f"原始提示词: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                    self.log(f"最终提示词: {final_prompt[:150]}{'...' if len(final_prompt) > 150 else ''}")
                    if ref_image_urls:
                        self.log(f"参考图({len(ref_image_urls)}张)")
                    if prev_video_url:
                        self.log(f"参考视频(上一段): {prev_video_url[:80]}...")
                        self._debug_log(
                            f"Segment uses previous video: segment={segment_num}, "
                            f"prev_video_domain={urlparse(prev_video_url).netloc}, url_chars={len(prev_video_url)}"
                        )

                    task_id = ""
                    stage = "create_task"
                    try:
                        self.status = f"第 {segment_num}/{len(prompts)} 段：正在提交任务"
                        stage = "create_task"
                        task_id = self._create_task(client, final_prompt, ref_image_urls, prev_video_url)
                        self.log(f"任务已提交: {task_id}")

                        self.status = f"第 {segment_num}/{len(prompts)} 段：正在轮询任务"
                        stage = "poll_task"
                        result = self._poll_task(client, task_id)
                        stage = "extract_video_url"
                        video_url = self._extract_video_url(result)

                        if not video_url:
                            raise TaskError(f"Task {task_id[:12]}... succeeded but no video_url in response")

                        self._results.append(
                            {
                                "segment": segment_num,
                                "task_id": task_id,
                                "prompt": prompt,
                                "video_url": video_url,
                                "status": result.get("status", ""),
                                "usage": result.get("usage"),
                            }
                        )

                        prev_video_url = video_url
                        self.log(f"视频: {video_url[:80]}...")
                        parsed_video_url = urlparse(video_url)
                        self._debug_log(
                            f"Segment success: segment={segment_num}, task_id={task_id}, "
                            f"video_domain={parsed_video_url.netloc}, url_chars={len(video_url)}"
                        )

                    except (TaskError, TaskTimeoutError) as e:
                        failure_details = self._set_failure(
                            stage=stage,
                            error=e,
                            segment=segment_num,
                            total_segments=len(prompts),
                            task_id=task_id,
                            prompt=final_prompt,
                            ref_video_url=prev_video_url,
                            extra={
                                "model": self.model,
                                "ref_images": len(ref_image_urls),
                                "has_ref_video": bool(prev_video_url),
                                "request_timeout": getattr(self, "request_timeout", DEFAULT_REQUEST_TIMEOUT),
                                "max_wait_time": self.max_wait_time,
                                "poll_interval": self.poll_interval,
                            },
                        )
                        self.log(f"段落 {segment_num} 失败: {e}", "ERROR")
                        self._debug_log(
                            f"Segment failed: segment={segment_num}, task_id={task_id or 'none'}, "
                            f"stage={stage}, error={type(e).__name__}: {e}",
                            name="ERROR",
                        )
                        self._results.append(
                            {
                                "segment": segment_num,
                                "task_id": task_id,
                                "prompt": prompt,
                                "video_url": "",
                                "status": "failed",
                                "failed_stage": stage,
                                "failed_stage_name": self._stage_name(stage),
                                "error_type": type(e).__name__,
                                "error": str(e),
                                "failure": failure_details,
                            }
                        )
                        # 链断了：后续段没法依赖前段，停下并把错误记下
                        self._generation_error = e
                        break

            self.status = f"完成 {len(self._results)}/{len(prompts)} 个视频"

        except (httpx.HTTPError, ValueError, KeyError) as e:
            # 客户端层面的失败（如代理、DNS、解析错误）：直接抛给 langflow 显示节点失败
            self.log(f"客户端错误: {e}", "ERROR")
            self._generation_error = e
            self._generation_done = True
            self._set_failure(stage="client", error=e)
            if raise_on_error:
                raise
            return

        self._generation_done = True

        # 如果链中途失败，把错误抛给下游而不是静默返回半截结果
        if self._generation_error is not None and raise_on_error:
            raise self._generation_error

    # ---------- 输出 ----------

    def get_video_urls(self) -> Message:
        self._ensure_generated()
        urls = [r.get("video_url", "") for r in self._results]
        return Message(text=json.dumps(urls, ensure_ascii=False))

    def get_task_info(self) -> Data:
        self._ensure_generated(raise_on_error=False)
        return Data(
            data={
                "segments": self._results,
                "failed": self._generation_error is not None,
                "failure": self._failure_info,
                "completed_segments": len([r for r in self._results if r.get("video_url")]),
                "total_segments": self._failure_info.get("total_segments") if self._failure_info else None,
            }
        )
