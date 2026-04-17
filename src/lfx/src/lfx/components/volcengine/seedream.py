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
from lfx.io import Output
from lfx.schema.data import Data
from lfx.schema.message import Message

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/images/generations"

MODE_TEXT = "Text to Image"
MODE_TEXT_IMAGE = "Text + Image(s)"
MODE_OPTIONS = [MODE_TEXT, MODE_TEXT_IMAGE]


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
            info="Model ID, e.g. doubao-seedream-4-5-251128.",
            value="doubao-seedream-4-5-251128",
        ),
        MultilineInput(
            name="prompt",
            display_name="Prompt",
            info="Text prompt for image generation.",
            required=True,
        ),
        DropdownInput(
            name="generation_mode",
            display_name="Generation Mode",
            info="Text to Image: pure text prompt. Text + Image(s): text prompt with reference images.",
            options=MODE_OPTIONS,
            value=MODE_TEXT,
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="image",
            display_name="Reference Image URLs",
            info="Reference image URLs. Supports manual input or connection from other components (max 14).",
            is_list=True,
            list_add_label="Add Image URL",
            placeholder="Enter an image URL...",
            input_types=["Message", "Text"],
            dynamic=True,
            show=False,
        ),
        DropdownInput(
            name="size",
            display_name="Size",
            info="Output image size. Seedream 4.x supports simplified names (2K, 4K).",
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
                "2K",
                "4K",
            ],
            value="1024x1024",
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
        IntInput(
            name="n",
            display_name="Number of Images",
            info="Number of images to generate. Uses sequential_image_generation when > 1.",
            value=1,
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

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "generation_mode" and "image" in build_config:
            build_config["image"]["show"] = field_value == MODE_TEXT_IMAGE
        return build_config

    def _resolve_image_urls(self) -> list[str]:
        """Normalize image input to a flat list of URL strings."""
        raw = getattr(self, "image", None)
        if not raw:
            return []

        urls: list[str] = []
        if isinstance(raw, str):
            urls.extend(line.strip() for line in raw.splitlines() if line.strip())
        elif isinstance(raw, Message):
            text = raw.get_text()
            urls.extend(line.strip() for line in text.splitlines() if line.strip())
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, str) and item.strip():
                    urls.extend(line.strip() for line in item.splitlines() if line.strip())
                elif isinstance(item, Message):
                    text = item.get_text()
                    urls.extend(line.strip() for line in text.splitlines() if line.strip())
        return urls

    def _build_request_body(self) -> dict:
        body: dict = {
            "model": self.model,
            "prompt": self.prompt,
            "size": self.size,
            "response_format": self.response_format,
            "watermark": self.watermark,
        }

        # Reference images (always as array)
        if getattr(self, "generation_mode", MODE_TEXT) == MODE_TEXT_IMAGE:
            ref_images = self._resolve_image_urls()
            if ref_images:
                body["image"] = ref_images

        # Sequential image generation for multiple outputs
        n = max(1, getattr(self, "n", 1))
        if n > 1:
            body["sequential_image_generation"] = "auto"
            body["sequential_image_generation_options"] = {"max_images": n}

        if getattr(self, "web_search", False):
            body["tools"] = [{"type": "web_search"}]

        return body

    def generate_image(self) -> Message:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(headers=headers, timeout=120, trust_env=False) as client:
                self.status = "Generating image..."
                self.log(f"Generating image with model: {self.model}")

                payload = self._build_request_body()
                resp = client.post(self.base_url, json=payload)

                if resp.status_code >= 400:
                    self.log(f"API error {resp.status_code}: {resp.text}", "ERROR")
                    resp.raise_for_status()

                data = resp.json()

                image_urls: list[str] = []
                for item in data.get("data", []):
                    if "url" in item:
                        image_urls.append(item["url"])
                    elif "b64_json" in item:
                        image_urls.append(f"data:image/png;base64,{item['b64_json']}")

                if not image_urls:
                    self._generation_info = {"model": self.model, "error": "No image in response"}
                    return Message(text="Error: No image in response")

                if len(image_urls) == 1:
                    result_text = image_urls[0]
                else:
                    import json

                    result_text = json.dumps(image_urls, ensure_ascii=False)

                self._generation_info = {
                    "model": self.model,
                    "prompt": self.prompt[:100],
                    "usage": data.get("usage", {}),
                    "image_urls": image_urls,
                }

                self.status = f"Generated {len(image_urls)} image(s)"
                return Message(text=result_text)

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
