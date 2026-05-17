from lfx.custom import Component
from lfx.io import (
    BoolInput,
    DropdownInput,
    IntInput,
    MessageTextInput,
    Output,
    SecretStrInput,
    TabInput,
)
from lfx.schema.data import Data

BASE_URL = "https://open.feedcoopapi.com/search_api/web_search"

SEARCH_TYPE_WEB = "web"
SEARCH_TYPE_WEB_SUMMARY = "web_summary"
SEARCH_TYPE_IMAGE = "image"

TIME_RANGE_OPTIONS = [
    "不限",
    "OneDay",
    "OneWeek",
    "OneMonth",
    "OneYear",
]

CONTENT_FORMAT_OPTIONS = ["text", "markdown"]

TAB_WEB = "网页搜索"
TAB_SUMMARY = "总结搜索"
TAB_IMAGE = "图片搜索"

TAB_TO_SEARCH_TYPE = {
    TAB_WEB: SEARCH_TYPE_WEB,
    TAB_SUMMARY: SEARCH_TYPE_WEB_SUMMARY,
    TAB_IMAGE: SEARCH_TYPE_IMAGE,
}


class VolcengineWebSearchComponent(Component):
    display_name = "Volcengine Web Search"
    description = "使用火山引擎联网搜索 API 搜索网页、图片或获取 AI 总结结果。"
    icon = "Search"
    name = "VolcengineWebSearch"

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="火山引擎联网搜索 API Key。",
            required=True,
        ),
        TabInput(
            name="search_mode",
            display_name="Search Mode",
            info="搜索模式：网页搜索 / 总结搜索 / 图片搜索。",
            options=[TAB_WEB, TAB_SUMMARY, TAB_IMAGE],
            value=TAB_WEB,
            tool_mode=True,
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="query",
            display_name="Query",
            info="搜索关键词，1~100 个字符。",
            required=True,
            tool_mode=True,
        ),
        IntInput(
            name="count",
            display_name="Count",
            info="返回结果条数。网页/总结搜索最多 50 条，图片搜索最多 5 条。",
            value=10,
            advanced=True,
        ),
        DropdownInput(
            name="time_range",
            display_name="Time Range",
            info="搜索结果的时间范围过滤。",
            options=TIME_RANGE_OPTIONS,
            value="不限",
            advanced=True,
        ),
        DropdownInput(
            name="content_format",
            display_name="Content Format",
            info="返回正文的格式。",
            options=CONTENT_FORMAT_OPTIONS,
            value="text",
            advanced=True,
        ),
        BoolInput(
            name="need_summary",
            display_name="Need Summary",
            info="是否返回精准摘要。使用总结搜索时自动开启。",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="need_content",
            display_name="Need Content",
            info="是否仅返回有正文的结果。",
            value=False,
            advanced=True,
        ),
        MessageTextInput(
            name="sites",
            display_name="Sites",
            info="限定搜索站点，多个站点用 | 分隔，如 zhihu.com|baidu.com。",
            advanced=True,
        ),
        MessageTextInput(
            name="block_hosts",
            display_name="Block Hosts",
            info="屏蔽的站点，多个域名用 | 分隔。",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Results", name="results", method="search"),
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "search_mode":
            is_image = field_value == TAB_IMAGE
            build_config["count"]["info"] = (
                "返回结果条数，最多 5 条。" if is_image else "返回结果条数，最多 50 条，默认 10 条。"
            )
        return build_config

    def _build_payload(self) -> dict:
        search_type = TAB_TO_SEARCH_TYPE.get(self.search_mode, SEARCH_TYPE_WEB)
        query = self.query
        if hasattr(query, "get_text"):
            query = query.get_text()
        query = str(query).strip()

        payload: dict = {
            "Query": query,
            "SearchType": search_type,
        }

        if search_type != SEARCH_TYPE_IMAGE:
            if self.count:
                payload["Count"] = min(self.count, 50)
            if self.need_summary or search_type == SEARCH_TYPE_WEB_SUMMARY:
                payload["NeedSummary"] = True
            elif self.need_summary:
                payload["NeedSummary"] = True

            time_range = self.time_range
            if time_range and time_range != "不限":
                payload["TimeRange"] = time_range

            if self.content_format:
                payload["ContentFormats"] = self.content_format

            filter_obj: dict = {}
            if self.need_content:
                filter_obj["NeedContent"] = True
            filter_obj["NeedUrl"] = True

            sites = self.sites
            if hasattr(sites, "get_text"):
                sites = sites.get_text()
            sites = str(sites).strip() if sites else ""
            if sites:
                filter_obj["Sites"] = sites

            block_hosts = self.block_hosts
            if hasattr(block_hosts, "get_text"):
                block_hosts = block_hosts.get_text()
            block_hosts = str(block_hosts).strip() if block_hosts else ""
            if block_hosts:
                filter_obj["BlockHosts"] = block_hosts

            if filter_obj:
                payload["Filter"] = filter_obj
        else:
            if self.count:
                payload["Count"] = min(self.count, 5)

        return payload

    def search(self) -> list[Data]:
        import httpx

        query = self.query
        if hasattr(query, "get_text"):
            query = query.get_text()
        query = str(query).strip()
        if not query:
            self.status = "Query is empty."
            return []

        payload = self._build_payload()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        try:
            with httpx.Client(timeout=30) as client:
                response = client.post(BASE_URL, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
        except httpx.HTTPStatusError as e:
            self.status = f"Search failed: {e.response.status_code} - {e.response.text}"
            return []
        except httpx.RequestError as e:
            self.status = f"Request error: {e}"
            return []

        results = self._parse_results(data)
        self.status = f"Found {len(results)} result(s)"
        return results

    def _parse_results(self, data: dict) -> list[Data]:
        search_type = TAB_TO_SEARCH_TYPE.get(self.search_mode, SEARCH_TYPE_WEB)
        results: list[Data] = []

        if search_type == SEARCH_TYPE_IMAGE:
            for item in data.get("ImageResults", []):
                results.append(
                    Data(
                        data={
                            "title": item.get("Title", ""),
                            "url": item.get("Url", ""),
                            "thumbnail": item.get("ThumbnailUrl", ""),
                            "source": item.get("Source", ""),
                            "width": item.get("Width"),
                            "height": item.get("Height"),
                        }
                    )
                )
        else:
            for item in data.get("WebResults", []):
                entry: dict = {
                    "title": item.get("Title", ""),
                    "url": item.get("Url", ""),
                    "snippet": item.get("Snippet", ""),
                    "site_name": item.get("SiteName", ""),
                    "publish_time": item.get("PublishTime", ""),
                    "auth_info": item.get("AuthInfoDes", ""),
                }
                if item.get("Summary"):
                    entry["summary"] = item["Summary"]
                if item.get("Content"):
                    entry["content"] = item["Content"]
                results.append(Data(data=entry))

            if search_type == SEARCH_TYPE_WEB_SUMMARY:
                choices = data.get("Choices", [])
                if choices:
                    summary_text = ""
                    for choice in choices:
                        delta = choice.get("Delta", {}) or choice.get("Message", {})
                        summary_text += delta.get("content", "")
                    if summary_text.strip():
                        results.append(
                            Data(
                                data={
                                    "title": "AI Summary",
                                    "content": summary_text.strip(),
                                }
                            )
                        )

        return results
