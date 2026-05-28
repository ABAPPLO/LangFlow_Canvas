import json
from typing import Any

import requests
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field

from lfx.base.langchain_utilities.model import LCToolComponent
from lfx.field_typing import Tool
from lfx.inputs.inputs import MultilineInput, SecretStrInput, StrInput
from lfx.schema.data import Data


class NotionPageCreator(LCToolComponent):
    display_name: str = "Create Page "
    description: str = "A component for creating Notion pages."
    documentation: str = "https://docs.langflow.org/bundles-notion"
    icon = "NotionDirectoryLoader"

    inputs = [
        StrInput(
            name="database_id",
            display_name="Database ID",
            info="The ID of the Notion database.",
        ),
        SecretStrInput(
            name="notion_secret",
            display_name="Notion Secret",
            info="The Notion integration token.",
            required=True,
        ),
        MultilineInput(
            name="properties_json",
            display_name="Properties (JSON)",
            info="The properties of the new page as a JSON string.",
        ),
    ]

    class NotionPageCreatorSchema(BaseModel):
        database_id: str = Field(..., description="The ID of the Notion database.")
        properties_json: str = Field(..., description="The properties of the new page as a JSON string.")

    def run_model(self) -> Data:
        result = self._create_notion_page(self.database_id, self.properties_json)
        output = "Created page properties:\n"
        for prop_name, prop_value in result.get("properties", {}).items():
            output += f"{prop_name}: {prop_value}\n"
        return Data(text=output, data=result)

    def build_tool(self) -> Tool:
        return StructuredTool.from_function(
            name="create_notion_page",
            description="Create a new page in a Notion database. "
            "IMPORTANT: Use the tool to check the Database properties for more details before using this tool.",
            func=self._create_notion_page,
            args_schema=self.NotionPageCreatorSchema,
        )

    def _create_notion_page(self, database_id: str, properties_json: str) -> dict[str, Any]:
        if not database_id or not properties_json:
            raise ValueError("Both 'database_id' and 'properties_json' are required.")

        try:
            properties = json.loads(properties_json)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format for properties: {e}") from e

        headers = {
            "Authorization": f"Bearer {self.notion_secret}",
            "Content-Type": "application/json",
            "Notion-Version": "2022-06-28",
        }

        data = {
            "parent": {"database_id": database_id},
            "properties": properties,
        }

        try:
            response = requests.post("https://api.notion.com/v1/pages", headers=headers, json=data, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            status_info = ""
            if hasattr(e, "response") and e.response is not None:
                status_info = f" Status code: {e.response.status_code}, Response: {e.response.text}"
            raise ConnectionError(f"Failed to create Notion page in database '{database_id}': {e}{status_info}") from e

    def __call__(self, *args, **kwargs):
        return self._create_notion_page(*args, **kwargs)
