"""Volcengine web search component."""

from __future__ import annotations

import copy
import json
from typing import Any

import httpx

from lfx.custom import Component
from lfx.inputs import BoolInput, DropdownInput, IntInput, MessageTextInput, SecretStrInput
from lfx.io import Output
from lfx.schema.data import Data
from lfx.schema.message import Message

BASE_URL = "https://open.feedcoopapi.com/search_api/web_search"

SEARCH_WEB = "web"
SEARCH_WEB_SUMMARY = "web_summary"
SEARCH_IMAGE = "image"
SEARCH_TYPES = [SEARCH_WEB, SEARCH_WEB_SUMMARY, SEARCH_IMAGE]

WEB_FIELD_NAMES = {
    "need_content",
    "need_url",
    "need_summary",
    "time_range",
    "sites",
    "block_hosts",
    "auth_info_level",
    "content_formats",
    "industry",
}
IMAGE_FIELD_NAMES = {
    "image_width_min",
    "image_height_min",
    "image_width_max",
    "image_height_max",
    "image_shapes",
}

ERROR_HINTS = {
    "10400": "参数错误，请检查 Query、SearchType 或 Filter 字段。",
    "10401": "无效的鉴权 Token，请检查 API Key。",
    "10402": "搜索类型非法或未开通对应搜索服务。",
    "10403": "权限错误，请确认账号已开通对应服务。",
    "10406": "免费额度已用尽，请在火山联网搜索控制台开通付费调用。",
    "10500": "服务端内部错误，可稍后重试。",
    "700429": "QPS 超限，请降低并发或申请扩容。",
}


class VolcengineWebSearchComponent(Component):
    display_name = "火山联网搜索"
    description = "调用火山联网搜索 API，支持网页搜索、网页总结和图片搜索，输出完整 Data 与可直接给大模型使用的 Message。"
    icon = "search"
    name = "VolcengineWebSearch"

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="必填。火山联网搜索控制台创建的 API Key，组件会放到 Authorization: Bearer 中；建议绑定 Langflow 全局变量。",
            required=True,
        ),
        MessageTextInput(
            name="base_url",
            display_name="Base URL",
            info=f"联网搜索接口地址。默认使用 APIKey 接入地址：{BASE_URL}；如走网关或代理，可绑定全局变量覆盖。",
            value=BASE_URL,
            advanced=True,
        ),
        MessageTextInput(
            name="query",
            display_name="Query（搜索词）",
            info="必填。用户要搜索的问题或关键词，官方限制 1-100 个字符；建议一次只放一个明确搜索意图。",
            required=True,
            tool_mode=True,
        ),
        DropdownInput(
            name="search_type",
            display_name="Search Type（搜索模式）",
            info="选择搜索模式：web 返回网页结果；web_summary 返回网页结果并生成总结；image 返回图片结果。切换后会自动显示对应参数。",
            options=SEARCH_TYPES,
            value=SEARCH_WEB,
            real_time_refresh=True,
        ),
        IntInput(
            name="count",
            display_name="Count（返回条数）",
            info="期望返回的结果数量。web/web_summary 最多 50 条，image 最多 5 条；超过上限时组件会自动压到允许范围。",
            value=10,
        ),
        BoolInput(
            name="query_rewrite",
            display_name="Query Rewrite（改写搜索词）",
            info="是否让搜索服务先优化/改写 Query。适合口语化、模糊问题；会增加耗时，精确关键词搜索可关闭。",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="need_content",
            display_name="Need Content（要求有正文）",
            info="开启后只返回带 Content 正文的网页结果。适合后续让大模型做深度分析；可能减少可返回结果数量。",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="need_url",
            display_name="Need URL（要求有链接）",
            info="开启后只返回带 Url 原文链接的网页结果。需要展示来源、引用链接或跳转原文时建议开启。",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="need_summary",
            display_name="Need Summary（精准摘要）",
            info="是否返回约 500 字的 Summary 精准摘要。相比 Snippet 更适合给大模型使用；web_summary 模式会自动强制开启。",
            value=True,
            advanced=True,
        ),
        MessageTextInput(
            name="time_range",
            display_name="Time Range（发布时间）",
            info="按发文时间过滤，不填表示不限时间。可填 OneDay、OneWeek、OneMonth、OneYear，或日期区间：YYYY-MM-DD..YYYY-MM-DD。",
            advanced=True,
        ),
        MessageTextInput(
            name="sites",
            display_name="Sites（指定站点）",
            info="只在指定域名内搜索，多个完整域名用 | 分隔，最多 20 个。示例：gov.cn|people.com.cn。",
            advanced=True,
        ),
        MessageTextInput(
            name="block_hosts",
            display_name="Block Hosts（屏蔽站点）",
            info="排除不想要的来源域名，多个完整域名用 | 分隔，最多 5 个。示例：example.com|spam.com。",
            advanced=True,
        ),
        IntInput(
            name="auth_info_level",
            display_name="Auth Info Level（权威度）",
            info="权威度过滤。0 表示不限制；1 表示仅搜索非常权威内容，适合政策、事实核验、严肃新闻场景。",
            value=0,
            advanced=True,
        ),
        DropdownInput(
            name="content_formats",
            display_name="Content Format（正文格式）",
            info="Content 正文返回格式。text 为纯文本；markdown 会尽量保留标题、列表等结构，适合给下游模型阅读。",
            options=["text", "markdown"],
            value="text",
            advanced=True,
        ),
        DropdownInput(
            name="industry",
            display_name="Industry（行业）",
            info="行业垂直搜索。不选表示通用搜索；finance 偏金融内容；game 偏电子游戏内容。",
            options=["", "finance", "game"],
            value="",
            advanced=True,
        ),
        IntInput(
            name="image_width_min",
            display_name="Image Width Min（最小宽度）",
            info="图片搜索过滤条件。只返回宽度大于等于该值的图片；填 0 表示不限制，仅 image 模式生效。",
            value=0,
            advanced=True,
            show=False,
        ),
        IntInput(
            name="image_height_min",
            display_name="Image Height Min（最小高度）",
            info="图片搜索过滤条件。只返回高度大于等于该值的图片；填 0 表示不限制，仅 image 模式生效。",
            value=0,
            advanced=True,
            show=False,
        ),
        IntInput(
            name="image_width_max",
            display_name="Image Width Max（最大宽度）",
            info="图片搜索过滤条件。只返回宽度小于等于该值的图片；填 0 表示不限制，仅 image 模式生效。",
            value=0,
            advanced=True,
            show=False,
        ),
        IntInput(
            name="image_height_max",
            display_name="Image Height Max（最大高度）",
            info="图片搜索过滤条件。只返回高度小于等于该值的图片；填 0 表示不限制，仅 image 模式生效。",
            value=0,
            advanced=True,
            show=False,
        ),
        MessageTextInput(
            name="image_shapes",
            display_name="Image Shapes（图片形状）",
            info="图片形状过滤，多个用 | 分隔。可填：横长方形、竖长方形、方形。示例：横长方形|方形。",
            advanced=True,
            show=False,
        ),
    ]

    outputs = [
        Output(display_name="Data（完整结构）", name="data", method="search_data", types=["Data"]),
        Output(display_name="Message（模型上下文）", name="message", method="search_message", type_=Message),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._raw_response: dict[str, Any] | None = None
        self._normalized_response: dict[str, Any] | None = None
        self._message_text: str | None = None

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name != "search_type":
            return build_config

        search_type = field_value or SEARCH_WEB
        is_image = search_type == SEARCH_IMAGE

        for name in WEB_FIELD_NAMES:
            if name in build_config:
                build_config[name]["show"] = not is_image

        for name in IMAGE_FIELD_NAMES:
            if name in build_config:
                build_config[name]["show"] = is_image

        return build_config

    def search_data(self) -> Data:
        normalized = self._get_normalized_response()
        self.status = self._build_status(normalized)
        return Data(data=normalized)

    def search_message(self) -> Message:
        normalized = self._get_normalized_response()
        self.status = self._build_status(normalized)
        return Message(text=self._message_text or "")

    def _get_normalized_response(self) -> dict[str, Any]:
        if self._normalized_response is not None:
            return self._normalized_response

        raw_response = self._request_search()
        normalized = self._normalize_response(raw_response)
        message_text = self._format_message(normalized)

        normalized["message_text"] = message_text
        self._normalized_response = normalized
        self._message_text = message_text
        return normalized

    def _request_search(self) -> dict[str, Any]:
        if self._raw_response is not None:
            return self._raw_response

        payload = self._build_payload()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        base_url = self._clean_str(getattr(self, "base_url", "")) or BASE_URL

        try:
            with httpx.Client(headers=headers, timeout=60.0, trust_env=False) as client:
                response = client.post(base_url, json=payload)

            if response.status_code >= 400:
                message = self._format_http_error(response)
                self.status = message
                raise ValueError(message)

            raw_response = self._parse_response(response)
            self._raise_if_api_error(raw_response)
        except httpx.TimeoutException as exc:
            message = "火山联网搜索请求超时，请稍后重试或减少返回条数。"
            self.status = message
            raise ValueError(message) from exc
        except httpx.HTTPError as exc:
            message = f"火山联网搜索请求失败：{exc}"
            self.status = message
            raise ValueError(message) from exc

        self._raw_response = raw_response
        return raw_response

    def _build_payload(self) -> dict[str, Any]:
        query = str(self.query or "").strip()
        if not query:
            msg = "Query 不能为空。"
            raise ValueError(msg)

        search_type = str(self.search_type or SEARCH_WEB).strip()
        if search_type not in SEARCH_TYPES:
            msg = f"不支持的 Search Type：{search_type}"
            raise ValueError(msg)

        max_count = 5 if search_type == SEARCH_IMAGE else 50
        default_count = 5 if search_type == SEARCH_IMAGE else 10
        count = int(getattr(self, "count", default_count) or default_count)
        count = min(max(count, 1), max_count)

        payload: dict[str, Any] = {
            "Query": query,
            "SearchType": search_type,
            "Count": count,
        }

        if getattr(self, "query_rewrite", False):
            payload["QueryControl"] = {"QueryRewrite": True}

        if search_type == SEARCH_IMAGE:
            image_filter = self._build_image_filter()
            if image_filter:
                payload["Filter"] = image_filter
            return payload

        web_filter = self._build_web_filter()
        if web_filter:
            payload["Filter"] = web_filter

        payload["NeedSummary"] = True if search_type == SEARCH_WEB_SUMMARY else bool(self.need_summary)

        time_range = self._clean_str(getattr(self, "time_range", ""))
        if time_range:
            payload["TimeRange"] = time_range

        content_formats = self._clean_str(getattr(self, "content_formats", ""))
        if content_formats:
            payload["ContentFormats"] = content_formats

        industry = self._clean_str(getattr(self, "industry", ""))
        if industry:
            payload["Industry"] = industry

        return payload

    def _build_web_filter(self) -> dict[str, Any]:
        web_filter: dict[str, Any] = {}

        if getattr(self, "need_content", False):
            web_filter["NeedContent"] = True
        if getattr(self, "need_url", False):
            web_filter["NeedUrl"] = True

        sites = self._clean_str(getattr(self, "sites", ""))
        if sites:
            web_filter["Sites"] = sites

        block_hosts = self._clean_str(getattr(self, "block_hosts", ""))
        if block_hosts:
            web_filter["BlockHosts"] = block_hosts

        auth_info_level = int(getattr(self, "auth_info_level", 0) or 0)
        if auth_info_level:
            web_filter["AuthInfoLevel"] = auth_info_level

        return web_filter

    def _build_image_filter(self) -> dict[str, Any]:
        image_filter: dict[str, Any] = {}
        int_fields = {
            "ImageWidthMin": "image_width_min",
            "ImageHeightMin": "image_height_min",
            "ImageWidthMax": "image_width_max",
            "ImageHeightMax": "image_height_max",
        }

        for api_key, field_name in int_fields.items():
            value = int(getattr(self, field_name, 0) or 0)
            if value > 0:
                image_filter[api_key] = value

        shapes = self._split_pipe_values(getattr(self, "image_shapes", ""))
        if shapes:
            image_filter["ImageShapes"] = shapes

        return image_filter

    @staticmethod
    def _clean_str(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _split_pipe_values(value: Any) -> list[str]:
        return [item.strip() for item in str(value or "").split("|") if item.strip()]

    def _parse_response(self, response: httpx.Response) -> dict[str, Any]:
        try:
            parsed = response.json()
            if isinstance(parsed, dict):
                return parsed
        except ValueError:
            pass

        frames = self._parse_sse_frames(response.text)
        if frames:
            return self._merge_sse_frames(frames)

        msg = "火山联网搜索响应不是有效 JSON。"
        raise ValueError(msg)

    @staticmethod
    def _parse_sse_frames(text: str) -> list[dict[str, Any]]:
        frames: list[dict[str, Any]] = []

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("data:"):
                line = line[5:].strip()
            if line in {"[DONE]", "[DONE", "]"}:
                continue
            if not line.startswith("{"):
                continue

            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(frame, dict):
                frames.append(frame)

        return frames

    @staticmethod
    def _merge_sse_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
        if len(frames) == 1:
            return frames[0]

        base = next(
            (
                frame
                for frame in frames
                if (frame.get("Result") or {}).get("WebResults") or (frame.get("Result") or {}).get("ImageResults")
            ),
            frames[0],
        )
        merged = copy.deepcopy(base)
        result = merged.setdefault("Result", {})

        summary_parts: list[str] = []
        usage: dict[str, Any] | None = None

        for frame in frames:
            frame_result = frame.get("Result") or {}
            usage = frame_result.get("Usage") or usage
            for choice in frame_result.get("Choices") or []:
                message = choice.get("Message") or {}
                delta = choice.get("Delta") or {}
                content = (
                    message.get("Content")
                    or message.get("content")
                    or delta.get("Content")
                    or delta.get("content")
                    or ""
                )
                if content:
                    summary_parts.append(str(content))

        if summary_parts:
            result["Choices"] = [
                {
                    "Message": {"Role": "assistant", "Content": "".join(summary_parts)},
                    "Delta": None,
                    "FinishReason": "stop",
                    "Index": 0,
                }
            ]
        if usage:
            result["Usage"] = usage
        result["StreamFrameCount"] = len(frames)

        return merged

    def _raise_if_api_error(self, raw_response: dict[str, Any]) -> None:
        metadata = raw_response.get("ResponseMetadata") or {}
        error = metadata.get("Error")
        if not error:
            return

        code = str(error.get("Code") or error.get("CodeN") or "")
        message = error.get("Message") or "火山联网搜索接口返回错误。"
        hint = ERROR_HINTS.get(code)
        if hint:
            message = f"{message}（{hint}）"
        if code:
            message = f"[{code}] {message}"
        self.status = message
        raise ValueError(message)

    def _format_http_error(self, response: httpx.Response) -> str:
        try:
            raw = response.json()
            metadata = raw.get("ResponseMetadata") or {}
            error = metadata.get("Error") or {}
            code = str(error.get("Code") or error.get("CodeN") or response.status_code)
            detail = error.get("Message") or response.text
        except ValueError:
            code = str(response.status_code)
            detail = response.text

        hint = ERROR_HINTS.get(code)
        if hint:
            return f"火山联网搜索请求失败 [{code}]：{detail}（{hint}）"
        return f"火山联网搜索请求失败 [{code}]：{detail}"

    def _normalize_response(self, raw_response: dict[str, Any]) -> dict[str, Any]:
        result = raw_response.get("Result") or raw_response
        search_context = result.get("SearchContext") or {}
        search_type = search_context.get("SearchType") or self.search_type
        web_results = [self._normalize_web_item(item) for item in result.get("WebResults") or []]
        image_results = [self._normalize_image_item(item) for item in result.get("ImageResults") or []]
        summary_text = self._extract_choice_text(result.get("Choices") or [])

        return {
            "query": search_context.get("OriginQuery") or self.query,
            "search_type": search_type,
            "result_count": result.get("ResultCount", len(web_results) + len(image_results)),
            "web_results": web_results,
            "image_results": image_results,
            "summary_text": summary_text,
            "usage": result.get("Usage") or {},
            "time_cost": result.get("TimeCost"),
            "log_id": result.get("LogId"),
            "card_results": result.get("CardResults"),
            "raw": raw_response,
        }

    @classmethod
    def _normalize_web_item(cls, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": cls._pick(item, "Id", "id"),
            "sort_id": cls._pick(item, "SortId", "sort_id"),
            "title": cls._pick(item, "Title", "title"),
            "site_name": cls._pick(item, "SiteName", "site_name"),
            "url": cls._pick(item, "Url", "url"),
            "snippet": cls._pick(item, "Snippet", "snippet"),
            "summary": cls._pick(item, "Summary", "summary"),
            "content": cls._pick(item, "Content", "content"),
            "publish_time": cls._pick(item, "PublishTime", "publish_time"),
            "logo_url": cls._pick(item, "LogoUrl", "logo_url"),
            "rank_score": cls._pick(item, "RankScore", "rank_score"),
            "auth_info_des": cls._pick(item, "AuthInfoDes", "auth_info_des"),
            "auth_info_level": cls._pick(item, "AuthInfoLevel", "auth_info_level"),
            "content_formats": cls._pick(item, "ContentFormats", "content_formats"),
            "ruyi_info": cls._pick(item, "RuyiInfo", "ruyi_info"),
        }

    @classmethod
    def _normalize_image_item(cls, item: dict[str, Any]) -> dict[str, Any]:
        image = cls._pick(item, "Image", "image") or {}
        return {
            "id": cls._pick(item, "Id", "id"),
            "sort_id": cls._pick(item, "SortId", "sort_id"),
            "title": cls._pick(item, "Title", "title"),
            "site_name": cls._pick(item, "SiteName", "site_name"),
            "url": cls._pick(item, "Url", "url"),
            "publish_time": cls._pick(item, "PublishTime", "publish_time"),
            "rank_score": cls._pick(item, "RankScore", "rank_score"),
            "image": {
                "url": cls._pick(image, "Url", "url"),
                "width": cls._pick(image, "Width", "width"),
                "height": cls._pick(image, "Height", "height"),
                "shape": cls._pick(image, "Shape", "shape"),
                "blur_des": cls._pick(image, "BlurDes", "blur_des"),
                "category": cls._pick(image, "Category", "category"),
                "watermark": cls._pick(image, "Watermark", "watermark"),
            },
        }

    @staticmethod
    def _pick(data: dict[str, Any], *keys: str) -> Any:
        for key in keys:
            if key in data:
                return data[key]
        return None

    @staticmethod
    def _extract_choice_text(choices: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for choice in choices:
            message = choice.get("Message") or {}
            delta = choice.get("Delta") or {}
            content = (
                message.get("Content")
                or message.get("content")
                or delta.get("Content")
                or delta.get("content")
                or ""
            )
            if content:
                parts.append(str(content))
        return "".join(parts).strip()

    def _format_message(self, normalized: dict[str, Any]) -> str:
        search_type = normalized.get("search_type")
        if search_type == SEARCH_IMAGE:
            return self._format_image_message(normalized.get("image_results") or [])
        return self._format_web_message(normalized)

    def _format_web_message(self, normalized: dict[str, Any]) -> str:
        lines: list[str] = []
        summary_text = normalized.get("summary_text")
        if summary_text:
            lines.append("搜索总结：")
            lines.append(str(summary_text))
            lines.append("")
            lines.append("参考来源：")

        web_results = normalized.get("web_results") or []
        if not web_results:
            return "\n".join(lines).strip() or "未搜索到网页结果。"

        for index, item in enumerate(web_results, start=1):
            title = item.get("title") or "无标题"
            site_name = item.get("site_name") or "未知站点"
            url = item.get("url") or ""
            text = item.get("summary") or item.get("snippet") or item.get("content") or ""
            publish_time = item.get("publish_time") or ""
            auth_info = item.get("auth_info_des") or ""

            lines.append(f"{index}. {title}")
            lines.append(f"站点：{site_name}")
            if url:
                lines.append(f"链接：{url}")
            if publish_time:
                lines.append(f"发布时间：{publish_time}")
            if auth_info:
                lines.append(f"权威度：{auth_info}")
            if text:
                lines.append(f"摘要：{self._clip(text, 1000)}")
            lines.append("")

        return "\n".join(lines).strip()

    def _format_image_message(self, image_results: list[dict[str, Any]]) -> str:
        if not image_results:
            return "未搜索到图片结果。"

        lines: list[str] = []
        for index, item in enumerate(image_results, start=1):
            image = item.get("image") or {}
            title = item.get("title") or "无标题"
            site_name = item.get("site_name") or "未知站点"
            image_url = image.get("url") or ""
            width = image.get("width") or ""
            height = image.get("height") or ""
            shape = image.get("shape") or ""

            lines.append(f"{index}. {title}")
            lines.append(f"站点：{site_name}")
            if item.get("url"):
                lines.append(f"落地页：{item['url']}")
            if image_url:
                lines.append(f"图片：{image_url}")
            if width or height:
                lines.append(f"尺寸：{width}x{height}")
            if shape:
                lines.append(f"形状：{shape}")
            lines.append("")

        return "\n".join(lines).strip()

    @staticmethod
    def _clip(text: Any, max_chars: int) -> str:
        value = str(text or "").strip()
        if len(value) <= max_chars:
            return value
        return f"{value[:max_chars]}..."

    @staticmethod
    def _build_status(normalized: dict[str, Any]) -> str:
        return (
            f"{normalized.get('search_type')} 搜索完成，"
            f"结果数：{normalized.get('result_count', 0)}，"
            f"LogId：{normalized.get('log_id') or '-'}"
        )
