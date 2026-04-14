import httpx

from lfx.custom import Component
from lfx.inputs import (
    BoolInput,
    DropdownInput,
    MessageTextInput,
    MultilineInput,
    SecretStrInput,
)
from lfx.io import Output
from lfx.schema.data import Data
from lfx.schema.message import Message

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"


class SeedreamImageComponent(Component):
    display_name = "Seedream Image"
    description = "Generate images using Volcengine Seedream API."
    icon = "Image"
    name = "SeedreamImage"

    inputs = [
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="Volcengine API Key.",
            required=True,
        ),
        MessageTextInput(
            name="base_url",
            display_name="Base URL",
            info="API base URL. Leave empty to use the default Volcengine endpoint.",
            value=BASE_URL,
            advanced=True,
        ),
        MessageTextInput(
            name="model",
            display_name="Model",
            info="Model ID, e.g. doubao-seedream-3-0-t2i-250415.",
            value="doubao-seedream-3-0-t2i-250415",
        ),
        MultilineInput(
            name="prompt",
            display_name="Prompt",
            info="Text prompt for image generation.",
            required=True,
        ),
        MultilineInput(
            name="image",
            display_name="Reference Image URLs",
            info="Reference image URLs, one per line (max 14).",
            advanced=True,
        ),
        DropdownInput(
            name="size",
            display_name="Size",
            info="Output image size.",
            options=[
                "1024x1024",
                "1536x1536",
                "2048x2048",
                "2048x1536",
                "1536x2048",
                "3072x3072",
                "4096x4096",
                "4096x3072",
                "3072x4096",
            ],
            value="1024x1024",
            advanced=True,
        ),
        DropdownInput(
            name="output_format",
            display_name="Output Format",
            info="Image output format.",
            options=["jpeg", "png"],
            value="jpeg",
            advanced=True,
        ),
        DropdownInput(
            name="response_format",
            display_name="Response Format",
            info="Response format: url or base64 JSON.",
            options=["url", "b64_json"],
            value="url",
            advanced=True,
        ),
        BoolInput(
            name="watermark",
            display_name="Watermark",
            info="Add 'AI generated' watermark to the image.",
            value=False,
            advanced=True,
        ),
        BoolInput(
            name="web_search",
            display_name="Web Search",
            info="Enable web search for real-time content.",
            value=False,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Image URL",
            name="image_url",
            method="generate_image",
            type_=Message,
        ),
        Output(
            display_name="Generation Info",
            name="generation_info",
            method="get_generation_info",
            type_=Data,
        ),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._generation_info: dict | None = None

    @staticmethod
    def _parse_urls(text: str) -> list[str]:
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _build_request_body(self) -> dict:
        body: dict = {
            "model": self.model,
            "prompt": self.prompt,
            "size": self.size,
            "output_format": self.output_format,
            "response_format": self.response_format,
            "watermark": self.watermark,
        }

        ref_images = self._parse_urls(getattr(self, "image", ""))
        if ref_images:
            body["image"] = ref_images[0] if len(ref_images) == 1 else ref_images

        if getattr(self, "web_search", False):
            body["tools"] = [{"type": "web_search"}]

        return body

    def generate_image(self) -> Message:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(headers=headers, timeout=60, trust_env=False) as client:
                self.status = "Generating image..."
                self.log(f"Generating image with model: {self.model}")

                payload = self._build_request_body()
                resp = client.post(self.base_url, json=payload)

                if resp.status_code >= 400:
                    self.log(f"API error {resp.status_code}: {resp.text}", "ERROR")
                    resp.raise_for_status()

                data = resp.json()

                image_url = ""
                for item in data.get("data", []):
                    if "url" in item:
                        image_url = item["url"]
                        break
                    if "b64_json" in item:
                        image_url = f"data:image/{self.output_format};base64,{item['b64_json']}"
                        break

                self._generation_info = {
                    "model": self.model,
                    "prompt": self.prompt[:100],
                    "usage": data.get("usage", {}),
                    "image_url": image_url,
                }

                self.status = f"Image generated: {image_url[:50]}..."
                return Message(text=image_url)

        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.text
            except Exception:
                pass
            error_msg = f"HTTP {e.response.status_code}: {error_detail}"
            self.log(error_msg, "ERROR")
            self._generation_info = None
            return Message(text=f"Error: {error_msg}")
        except (httpx.HTTPError, ValueError, KeyError) as e:
            self.log(f"Error: {e}", "ERROR")
            self._generation_info = None
            return Message(text=f"Error: {e}")

    def get_generation_info(self) -> Data:
        if self._generation_info:
            return Data(data=self._generation_info)
        return Data(data={})
