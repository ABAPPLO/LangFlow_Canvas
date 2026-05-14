import json
from typing import Any

import requests
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.field_typing import Tool
from lfx.inputs.inputs import MultilineInput, SecretStrInput, StrInput
from lfx.log.logger import logger
from lfx.schema.data import Data


class NotionPageUpdate(LCToolComponent):
    display_name: str = "Update Page Property "
    description: str = "Update the properties of a Notion page."
    documentation: str = "https://docs.langflow.org/bundles-notion"
    icon = "NotionDirectoryLoader"

    inputs = [
        StrInput(
            name="page_id",
            display_name="Page ID",
            info="The ID of the Notion page to update.",
        ),
        MultilineInput(
            name="properties",
            display_name="Properties",
            info="The properties to update on the page (as a JSON string or a dictionary).",
        ),
        SecretStrInput(
            name="notion_secret",
            display_name="Notion Secret",
            info="The Notion integration token.",
            required=True,
        ),
    ]

    class NotionPageUpdateSchema(BaseModel):
        page_id: str = Field(..., description="The ID of the Notion page to update.")
        properties: str | dict[str, Any] = Field(
            ..., description="The properties to update on the page (as a JSON string or a dictionary)."
        )

    def run_model(self) -> Data:
        result = self._update_notion_page(self.page_id, self.properties)
        output = "Updated page properties:\n"
        for prop_name, prop_value in result.get("properties", {}).items():
            output += f"{prop_name}: {prop_value}\n"
        return Data(text=output, data=result)

    def build_tool(self) -> Tool:
        return StructuredTool.from_function(
            name="update_notion_page",
            description="Update the properties of a Notion page. "
            "IMPORTANT: Use the tool to check the Database properties for more details before using this tool.",
            func=self._update_notion_page,
            args_schema=self.NotionPageUpdateSchema,
        )

    def _update_notion_page(self, page_id: str, properties: str | dict[str, Any]) -> dict[str, Any]:
        url = f"https://api.notion.com/v1/pages/{page_id}"
        headers = {
            "Authorization": f"Bearer {self.notion_secret}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

        # Parse properties if it's a string
        if isinstance(properties, str):
            try:
                parsed_properties = json.loads(properties)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format for properties: {e}") from e
        else:
            parsed_properties = properties

        data = {"properties": parsed_properties}

        try:
            response = requests.patch(url, headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            status_info = ""
            if e.response is not None:
                status_info = f" Status code: {e.response.status_code}, Response: {e.response.text}"
            raise ConnectionError(f"Failed to update Notion page '{page_id}': {e}{status_info}") from e
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to update Notion page '{page_id}': {e}") from e

    def __call__(self, *args, **kwargs):
        return self._update_notion_page(*args, **kwargs)
