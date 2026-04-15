import httpx

from lfx.base.models.unified_models import (
    get_language_model_options,
    update_model_options_in_build_config,
)
from lfx.custom import Component
from lfx.inputs import (
    DropdownInput,
    IntInput,
    MessageTextInput,
    SecretStrInput,
)
from lfx.io import MessageInput, ModelInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message

MODE_TEXT = "Text to Image"
MODE_TEXT_IMAGE = "Text + Image(s)"

MODE_OPTIONS = [MODE_TEXT, MODE_TEXT_IMAGE]


class ImageGenerationComponent(Component):
    display_name = "Image Generation"
    description = "Generate images using OpenAI-compatible model providers (NewAPI, etc.)."
    icon = "Image"
    name = "ImageGeneration"

    inputs = [
        ModelInput(
            name="model",
            display_name="Image Model",
            info="Select an image generation model from your configured providers.",
            real_time_refresh=True,
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="API key. Falls back to Model Providers settings.",
            required=False,
            advanced=True,
        ),
        MessageInput(
            name="input_value",
            display_name="Prompt",
            info="Text prompt for image generation.",
        ),
        DropdownInput(
            name="generation_mode",
            display_name="Generation Mode",
            info="Select image generation mode.",
            options=MODE_OPTIONS,
            value=MODE_TEXT,
            real_time_refresh=True,
        ),
        MessageTextInput(
            name="image_urls",
            display_name="Reference Image URLs",
            info="Click '+' to add reference image URLs.",
            is_list=True,
            list_add_label="Add Image URL",
            placeholder="Enter an image URL...",
            input_types=[],
            dynamic=True,
            show=False,
        ),
        # --- Generation parameters ---
        DropdownInput(
            name="size",
            display_name="Size",
            info="Output image resolution.",
            options=[
                "1024x1024",
                "1536x1536",
                "2048x2048",
                "2048x1536",
                "1536x2048",
                "1280x720",
                "720x1280",
                "960x960",
            ],
            value="1024x1024",
            advanced=True,
        ),
        IntInput(
            name="n",
            display_name="Number of Images",
            info="Number of images to generate.",
            value=1,
            advanced=True,
        ),
        DropdownInput(
            name="response_format",
            display_name="Response Format",
            info="How the API returns the generated image.",
            options=["url", "b64_json"],
            value="url",
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Image URL",
            name="image_url",
            method="generate_image",
        ),
        Output(
            display_name="Generation Info",
            name="generation_info",
            method="get_generation_info",
        ),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._generation_info: dict | None = None

    def update_build_config(self, build_config, field_value, field_name=None):
        # Load model options from Model Providers
        build_config = update_model_options_in_build_config(
            component=self,
            build_config=build_config,
            cache_key_prefix="image_model_options",
            get_options_func=get_language_model_options,
            field_name=field_name,
            field_value=field_value,
        )

        if field_name == "generation_mode" and "image_urls" in build_config:
            build_config["image_urls"]["show"] = field_value == MODE_TEXT_IMAGE

        return build_config

    def _resolve_credentials(self) -> tuple[str, str, str]:
        """Resolve API key, base URL and model name from Model Providers or component inputs."""
        from lfx.base.models.unified_models import (
            get_all_variables_for_provider,
            get_api_key_for_provider,
        )

        model_data = self.model
        if not model_data or not isinstance(model_data, list) or len(model_data) == 0:
            msg = "Please select a model"
            raise ValueError(msg)

        model_info = model_data[0]
        model_name = model_info.get("name", "")
        provider = model_info.get("provider", "")

        # Resolve API key
        api_key = get_api_key_for_provider(self.user_id, provider, self.api_key)
        if not api_key:
            msg = (
                f"{provider} API key is required. "
                "Please provide it in the component or configure it in Model Providers."
            )
            raise ValueError(msg)

        # Resolve base URL
        base_url = None
        provider_vars = get_all_variables_for_provider(self.user_id, provider)
        for var_key, value in provider_vars.items():
            if "BASE_URL" in var_key and value:
                base_url = value
                break

        if not base_url:
            msg = (
                f"{provider} Base URL is required. "
                "Please configure it in Model Providers or provide it in the component."
            )
            raise ValueError(msg)

        # Ensure base_url ends with /v1/ for OpenAI-compatible APIs
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

        return api_key, base_url + "/", model_name

    def generate_image(self) -> Message:
        """Generate an image using the selected model via OpenAI-compatible API."""
        api_key, base_url, model_name = self._resolve_credentials()

        # Get prompt
        prompt = self.input_value
        if isinstance(prompt, Message):
            prompt = prompt.get_text()
        if not prompt:
            msg = "Please provide a prompt for image generation"
            raise ValueError(msg)

        # Build request payload
        payload: dict = {
            "model": model_name,
            "prompt": prompt,
            "n": max(1, self.n),
            "size": self.size,
            "response_format": self.response_format,
        }

        # Add reference images for Text + Image(s) mode
        if self.generation_mode == MODE_TEXT_IMAGE and self.image_urls:
            urls = self.image_urls
            payload["image"] = urls if len(urls) > 1 else urls[0]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = f"{base_url}images/generations"

        try:
            with httpx.Client(headers=headers, timeout=60, trust_env=False) as client:
                self.status = "Generating image..."
                self.log(f"Submitting request to {url}, model: {model_name}, mode: {self.generation_mode}")

                resp = client.post(url, json=payload)
                if not resp.is_success:
                    self.log(f"API error {resp.status_code}: {resp.text}", "ERROR")
                    resp.raise_for_status()

                data = resp.json()

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, "text") else ""
            msg = f"Image generation failed (HTTP {e.response.status_code}): {error_detail}"
            self.log(msg, "ERROR")
            raise ValueError(msg) from e
        except httpx.HTTPError as e:
            msg = f"Image generation failed: {e}"
            self.log(msg, "ERROR")
            raise ValueError(msg) from e

        # Extract image URL(s) from response
        image_items = data.get("data", [])
        if not image_items:
            msg = f"No image data in response: {data}"
            raise ValueError(msg)

        # Use the first image URL
        first_item = image_items[0]
        image_url = first_item.get("url", "")

        # Handle base64 response
        if not image_url and self.response_format == "b64_json":
            b64_data = first_item.get("b64_json", "")
            if b64_data:
                fmt = "png" if "png" in self.size else "jpeg"
                image_url = f"data:image/{fmt};base64,{b64_data}"

        if not image_url:
            msg = f"No image URL in response: {data}"
            raise ValueError(msg)

        # Store generation info
        self._generation_info = {
            "model": model_name,
            "mode": self.generation_mode,
            "prompt": prompt[:100],
            "size": self.size,
            "n": len(image_items),
            "usage": data.get("usage"),
            "image_url": image_url,
        }

        self.status = f"Image generated: {image_url[:80]}..."
        return Message(text=image_url)

    def get_generation_info(self) -> Data:
        """Return generation information as Data."""
        if self._generation_info:
            return Data(data=self._generation_info)
        return Data(data={})
