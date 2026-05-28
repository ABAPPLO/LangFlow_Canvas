import json
from collections.abc import Sequence
from typing import Any

import requests
from langchain.agents import Tool
from langchain_core.tools import StructuredTool
from pydantic.v1 import Field, create_model

from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.inputs.inputs import DropdownInput, IntInput, MessageTextInput, MultiselectInput
from lfx.io import Output
from lfx.schema.dotdict import dotdict


class SearXNGToolComponent(LCToolComponent):
    search_headers: dict = {}
    display_name = "SearXNG Search"
    description = "A component that searches for tools using SearXNG."
    name = "SearXNGTool"
    legacy: bool = True

    inputs = [
        MessageTextInput(
            name="url",
            display_name="URL",
            value="http://localhost",
            required=True,
            refresh_button=True,
        ),
        IntInput(
            name="max_results",
            display_name="Max Results",
            value=10,
            required=True,
        ),
        MultiselectInput(
            name="categories",
            display_name="Categories",
            options=[],
            value=[],
        ),
        DropdownInput(
            name="language",
            display_name="Language",
            options=[],
        ),
    ]

    outputs = [
        Output(display_name="Tool", name="result_tool", method="build_tool"),
    ]

    def update_build_config(self, build_config: dotdict, field_value: Any, field_name: str | None = None) -> dotdict:
        if field_name is None:
            return build_config

        if field_name != "url":
            return build_config

        url = f"{field_value}/config"

        try:
            response = requests.get(url=url, headers=self.search_headers.copy(), timeout=10)
        except requests.ConnectionError as e:
            msg = f"Could not connect to SearXNG instance at '{field_value}': {e}"
            raise ConnectionError(msg) from e
        except requests.Timeout as e:
            msg = f"Timeout connecting to SearXNG instance at '{field_value}': {e}"
            raise ConnectionError(msg) from e

        try:
            if response.headers.get("Content-Encoding") == "zstd":
                data = json.loads(response.content)
            else:
                data = response.json()
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Failed to parse response from SearXNG instance at '{field_value}': {e}"
            raise RuntimeError(msg) from e

        build_config["categories"]["options"] = data["categories"].copy()
        for selected_category in build_config["categories"]["value"]:
            if selected_category not in build_config["categories"]["options"]:
                build_config["categories"]["value"].remove(selected_category)
        languages = list(data["locales"])
        build_config["language"]["options"] = languages.copy()
        return build_config

    def build_tool(self) -> Tool:
        class SearxSearch:
            _url: str = ""
            _categories: list[str] = []
            _language: str = ""
            _headers: dict = {}
            _max_results: int = 10

            @staticmethod
            def search(query: str, categories: Sequence[str] = ()) -> list:
                if not SearxSearch._categories and not categories:
                    msg = "No categories provided."
                    raise ValueError(msg)
                all_categories = SearxSearch._categories + list(set(categories) - set(SearxSearch._categories))
                url = f"{SearxSearch._url}/"
                headers = SearxSearch._headers.copy()
                try:
                    http_response = requests.get(
                        url=url,
                        headers=headers,
                        params={
                            "q": query,
                            "categories": ",".join(all_categories),
                            "language": SearxSearch._language,
                            "format": "json",
                        },
                        timeout=10,
                    )
                    http_response.raise_for_status()
                    response = http_response.json()
                except requests.ConnectionError as e:
                    msg = f"Could not connect to SearXNG instance at '{SearxSearch._url}': {e}"
                    raise ConnectionError(msg) from e
                except requests.Timeout as e:
                    msg = f"Timeout searching SearXNG instance at '{SearxSearch._url}': {e}"
                    raise ConnectionError(msg) from e
                except requests.HTTPError as e:
                    msg = f"HTTP error from SearXNG instance at '{SearxSearch._url}': {e}"
                    raise ConnectionError(msg) from e
                except (json.JSONDecodeError, ValueError) as e:
                    msg = f"Failed to parse SearXNG response from '{SearxSearch._url}': {e}"
                    raise RuntimeError(msg) from e

                num_results = min(SearxSearch._max_results, len(response["results"]))
                return [response["results"][i] for i in range(num_results)]

        SearxSearch._url = self.url
        SearxSearch._categories = self.categories.copy()
        SearxSearch._language = self.language
        SearxSearch._headers = self.search_headers.copy()
        SearxSearch._max_results = self.max_results

        globals_ = globals()
        local = {}
        local["SearxSearch"] = SearxSearch
        globals_.update(local)

        schema_fields = {
            "query": (str, Field(..., description="The query to search for.")),
            "categories": (
                list[str],
                Field(default=[], description="The categories to search in."),
            ),
        }

        searx_search_schema = create_model("SearxSearchSchema", **schema_fields)

        return StructuredTool.from_function(
            func=local["SearxSearch"].search,
            args_schema=searx_search_schema,
            name="searxng_search_tool",
            description="A tool that searches for tools using SearXNG.\nThe available categories are: "
            + ", ".join(self.categories),
        )
