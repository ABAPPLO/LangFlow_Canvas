import base64
import json
import logging
import re

import httpx

logger = logging.getLogger(__name__)

# Default timeout for image generation requests (5 minutes)
IMAGE_GEN_TIMEOUT = 300

from lfx.base.models.unified_models import (
    get_image_model_options,
    update_model_options_in_build_config,
)
from lfx.custom import Component
from lfx.inputs import (
    DropdownInput,
    IntInput,
)
from lfx.inputs.inputs import MultilineInput, MessageTextInput
from lfx.io import MessageInput, ModelInput, Output
from lfx.schema.data import Data
from lfx.schema.image import Image
from lfx.schema.message import Message

MODE_TEXT = "Text to Image"
MODE_TEXT_IMAGE = "Text + Image(s)"
MODE_BATCH_JSON = "Batch from JSON"

MODE_OPTIONS = [MODE_TEXT, MODE_TEXT_IMAGE, MODE_BATCH_JSON]

REF_IMAGE_PREFIX = "ref_image_"

# Gemini models that generate images via chat completions endpoint
GEMINI_IMAGE_KEYWORDS = ("flash-image", "image-generation", "pro-image-preview")

# GPT Image models that use /v1/images/edits for reference image editing
GPT_IMAGE_KEYWORDS = ("gpt-image",)


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
        MessageInput(
            name="input_value",
            display_name="Prompt",
            info="Text prompt for image generation.",
        ),
        MessageTextInput(
            name="json_input",
            display_name="JSON Input",
            info="JSON array (or object) describing characters/items. Each object should have a field for the image prompt (e.g., description, prompt). Auto-detects the prompt field.",
            value="",
            show=False,
            placeholder='[{"name": "角色1", "description": "一个战士..."}]',
        ),
        DropdownInput(
            name="generation_mode",
            display_name="Generation Mode",
            info="Select image generation mode.",
            options=MODE_OPTIONS,
            value=MODE_TEXT,
            real_time_refresh=True,
        ),
        IntInput(
            name="ref_image_count",
            display_name="Reference Image Count",
            info="Number of reference image inputs. Change to add or remove entries.",
            value=1,
            real_time_refresh=True,
        ),
        # --- Generation parameters ---
        DropdownInput(
            name="resolution",
            display_name="Resolution",
            info="Output image resolution level.",
            options=["1K", "2K", "4K"],
            value="2K",
            advanced=True,
        ),
        DropdownInput(
            name="ratio",
            display_name="Aspect Ratio",
            info="Output image aspect ratio.",
            options=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"],
            value="1:1",
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
            display_name="Batch Result",
            name="batch_result",
            method="generate_batch",
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
            get_options_func=get_image_model_options,
            field_name=field_name,
            field_value=field_value,
        )

        if field_name == "generation_mode":
            is_text_image = field_value == MODE_TEXT_IMAGE
            is_batch = field_value == MODE_BATCH_JSON

            build_config["input_value"]["show"] = not is_batch
            build_config["json_input"]["show"] = is_batch
            build_config["ref_image_count"]["show"] = is_text_image

            # Remove old dynamic ref image fields when switching modes
            to_remove = [k for k in build_config if k.startswith(REF_IMAGE_PREFIX) and k[len(REF_IMAGE_PREFIX):].isdigit()]
            for k in to_remove:
                del build_config[k]

            if is_text_image:
                count = max(1, int(build_config.get("ref_image_count", {}).get("value", 1)))
                for i in range(1, count + 1):
                    f_name = f"{REF_IMAGE_PREFIX}{i}"
                    build_config[f_name] = {
                        "type": "str",
                        "input_types": ["Message", "Text"],
                        "name": f_name,
                        "display_name": f"Image {i}",
                        "value": "",
                        "show": True,
                        "advanced": False,
                        "multiline": False,
                        "placeholder": "Enter URL or connect component...",
                    }

        if field_name == "ref_image_count":
            count = max(1, int(field_value)) if field_value else 1

            # Only create dynamic fields if in Text + Image mode
            mode = build_config.get("generation_mode", {}).get("value", MODE_TEXT)
            if mode != MODE_TEXT_IMAGE:
                return build_config

            # Remove old dynamic ref image fields
            to_remove = [k for k in build_config if k.startswith(REF_IMAGE_PREFIX) and k[len(REF_IMAGE_PREFIX):].isdigit()]
            for k in to_remove:
                del build_config[k]

            for i in range(1, count + 1):
                f_name = f"{REF_IMAGE_PREFIX}{i}"
                build_config[f_name] = {
                    "type": "str",
                    "input_types": ["Message", "Text"],
                    "name": f_name,
                    "display_name": f"Image {i}",
                    "value": "",
                    "show": True,
                    "advanced": False,
                    "multiline": False,
                    "placeholder": "Enter URL or connect component...",
                }

        return build_config

    def _resolve_credentials(self) -> tuple[str, str, str]:
        """Resolve API key, base URL and model name from Model Providers or component inputs."""
        from lfx.base.models.unified_models import (
            get_all_variables_for_provider,
            get_api_key_for_provider,
            get_default_base_url,
        )

        model_data = self.model
        if not model_data or not isinstance(model_data, list) or len(model_data) == 0:
            msg = "Please select a model"
            raise ValueError(msg)

        model_info = model_data[0]
        model_name = model_info.get("name", "")
        provider = model_info.get("provider", "")

        # Resolve API key
        api_key = get_api_key_for_provider(self.user_id, provider)
        if not api_key:
            msg = (
                f"{provider} API key is required. "
                "Please configure it in Model Providers."
            )
            raise ValueError(msg)

        # Resolve base URL from provider variables
        base_url = None
        provider_vars = get_all_variables_for_provider(self.user_id, provider)
        for var_key, value in provider_vars.items():
            if "BASE_URL" in var_key and value:
                base_url = value
                break

        # Fall back to the provider's well-known default base URL
        if not base_url:
            base_url = get_default_base_url(provider)

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

    def _resolve_image_urls(self) -> list[str]:
        """Collect URLs from all dynamic reference image fields."""
        urls: list[str] = []
        i = 1
        while True:
            val = getattr(self, f"{REF_IMAGE_PREFIX}{i}", None)
            if val is None:
                break
            if isinstance(val, Message):
                text = val.get_text().strip()
            elif isinstance(val, str):
                text = val.strip()
            elif val:
                text = str(val).strip()
            else:
                text = ""
            if text:
                urls.extend(line.strip() for line in text.splitlines() if line.strip())
            i += 1
        return urls

    def _resolve_size(self) -> str:
        """Compute actual size string from resolution + aspect ratio."""
        res_base = {"1K": 1024, "2K": 2048, "4K": 4096}.get(self.resolution, 2048)
        ratio_sizes = {
            "1:1": (res_base, res_base),
            "16:9": (res_base, int(res_base * 9 / 16)),
            "9:16": (int(res_base * 9 / 16), res_base),
            "4:3": (res_base, int(res_base * 3 / 4)),
            "3:4": (int(res_base * 3 / 4), res_base),
            "21:9": (res_base, int(res_base * 9 / 21)),
        }
        w, h = ratio_sizes.get(self.ratio, (res_base, res_base))
        # Round to nearest 64 for compatibility
        w = max(512, round(w / 64) * 64)
        h = max(512, round(h / 64) * 64)
        return f"{w}x{h}"

    def _resolve_gpt_image_size(self) -> str:
        """Map aspect ratio to valid gpt-image size (1024x1024, 1536x1024, 1024x1536)."""
        landscape = {"16:9", "4:3", "21:9"}
        portrait = {"9:16", "3:4"}
        if self.ratio in landscape:
            return "1536x1024"
        if self.ratio in portrait:
            return "1024x1536"
        return "1024x1024"

    @staticmethod
    def _is_gemini_image_model(model_name: str) -> bool:
        """Check if the model is a Gemini image generation model."""
        return any(kw in model_name for kw in GEMINI_IMAGE_KEYWORDS)

    @staticmethod
    def _is_gpt_image_model(model_name: str) -> bool:
        """Check if the model is a GPT Image model (gpt-image-1, gpt-image-2, etc.)."""
        name_lower = model_name.lower()
        return any(kw in name_lower for kw in GPT_IMAGE_KEYWORDS)

    @staticmethod
    def _download_as_based64(url: str) -> tuple[str, str]:
        """Download a URL and return (base64_data, mime_type)."""
        with httpx.Client(timeout=IMAGE_GEN_TIMEOUT, trust_env=False) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            mime = resp.headers.get("content-type", "image/jpeg")
            b64 = base64.b64encode(resp.content).decode()
        return b64, mime

    def _ensure_displayable(self, image_url: str) -> str:
        """Download external image to local storage and return a local URL.

        Signed URLs (e.g. Volcengine TOS) may have CORS restrictions.
        Saves the image via the storage service so the frontend can
        access it through /files/images/{flow_id}/{file_name}.
        """
        if image_url.startswith("data:"):
            return image_url
        if not image_url.startswith("http"):
            return image_url

        try:
            # Download image bytes
            with httpx.Client(timeout=IMAGE_GEN_TIMEOUT, trust_env=False) as client:
                resp = client.get(image_url, follow_redirects=True)
                resp.raise_for_status()
                image_bytes = resp.content
                mime = resp.headers.get("content-type", "image/jpeg")

            # Determine extension from mime type
            ext_map = {
                "image/png": ".png",
                "image/jpeg": ".jpg",
                "image/jpg": ".jpg",
                "image/webp": ".webp",
                "image/gif": ".gif",
                "image/svg+xml": ".svg",
            }
            ext = ext_map.get(mime, ".png")

            # Save to local storage via storage service
            import uuid
            from lfx.services.deps import get_storage_service

            storage = get_storage_service()
            flow_id = str(self.graph.flow_id) if hasattr(self, "graph") and self.graph else str(uuid.uuid4())
            file_name = f"{uuid.uuid4().hex[:12]}{ext}"

            # Run async save_file in sync context
            import asyncio
            from lfx.utils.async_helpers import run_until_complete
            run_until_complete(storage.save_file(flow_id=flow_id, file_name=file_name, data=image_bytes))

            # Build local URL for the frontend to fetch
            from lfx.utils.util import transform_localhost_url
            settings_service = None
            try:
                from lfx.services.deps import get_settings_service
                settings_service = get_settings_service()
            except Exception:
                pass

            base_url = ""
            if settings_service:
                try:
                    base_url = settings_service.settings.base_url or ""
                except Exception:
                    pass
            if base_url:
                base_url = base_url.rstrip("/")
            else:
                base_url = "http://localhost:7880"

            return f"{base_url}/api/v1/files/images/{flow_id}/{file_name}"
        except Exception:
            # If anything fails, return original URL as fallback
            return image_url

    def _generate_via_gemini(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        prompt: str,
        ref_urls: list[str],
    ) -> Message:
        """Generate an image using Gemini models via the native generateContent endpoint."""
        # Build parts list
        parts: list[dict] = [{"text": prompt}]

        # Download reference images and embed as inlineData
        for ref_url in ref_urls:
            try:
                b64_data, mime = self._download_as_based64(ref_url)
                parts.append({"inlineData": {"mimeType": mime, "data": b64_data}})
            except httpx.HTTPError as e:
                msg = f"Failed to download reference image {ref_url}: {e}"
                self.log(msg, "ERROR")
                raise ValueError(msg) from e

        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT", "IMAGE"]},
        }

        # Derive the generateContent URL from base_url
        # base_url is like http://host:port/v1/ → use http://host:port/v1beta/models/{model}:generateContent
        raw_base = base_url.replace("/v1/", "").replace("/v1", "").rstrip("/")
        url = f"{raw_base}/v1beta/models/{model_name}:generateContent"

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(headers=headers, timeout=IMAGE_GEN_TIMEOUT, trust_env=False) as client:
                self.status = "Generating image via Gemini generateContent..."
                self.log(f"Submitting Gemini request to {url}, model: {model_name}")

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

        # Parse response — extract image from candidates[0].content.parts
        candidates = data.get("candidates", [])
        if not candidates:
            msg = f"No candidates in response: {data}"
            raise ValueError(msg)

        response_parts = candidates[0].get("content", {}).get("parts", [])
        image_data_url = ""
        for p in response_parts:
            inline = p.get("inlineData") or p.get("inline_data")
            if inline:
                mime = inline.get("mimeType") or inline.get("mime_type", "image/png")
                b64 = inline.get("data", "")
                if b64:
                    image_data_url = f"data:{mime};base64,{b64}"
                    break

        if not image_data_url:
            msg = f"No image found in Gemini response: {data}"
            raise ValueError(msg)

        # Store generation info
        self._generation_info = {
            "model": model_name,
            "mode": self.generation_mode,
            "prompt": prompt[:100],
            "image_url": image_data_url,
        }

        self.status = f"Image generated: {image_data_url[:80]}..."
        return Message(text=image_data_url, files=[Image(url=image_data_url)])

    def _generate_via_gpt_image_edits(
        self,
        api_key: str,
        base_url: str,
        model_name: str,
        prompt: str,
        ref_urls: list[str],
    ) -> Message:
        """Edit an image using gpt-image models via /v1/images/edits (multipart/form-data)."""
        # Download reference images
        files = []
        for ref_url in ref_urls:
            try:
                b64_data, mime = self._download_as_based64(ref_url)
                image_bytes = base64.b64decode(b64_data)
                ext = "png" if "png" in mime else "jpg"
                files.append(("image[]", (f"image.{ext}", image_bytes, mime)))
            except httpx.HTTPError as e:
                msg = f"Failed to download reference image {ref_url}: {e}"
                self.log(msg, "ERROR")
                raise ValueError(msg) from e

        size = self._resolve_gpt_image_size()
        data = {
            "prompt": prompt,
            "model": model_name,
            "n": str(max(1, self.n)),
            "size": size,
        }

        url = f"{base_url}images/edits"
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            with httpx.Client(headers=headers, timeout=IMAGE_GEN_TIMEOUT, trust_env=False) as client:
                self.status = "Editing image via GPT Image..."
                self.log(f"Submitting edit request to {url}, model: {model_name}")

                resp = client.post(url, files=files, data=data)
                if not resp.is_success:
                    self.log(f"API error {resp.status_code}: {resp.text}", "ERROR")
                    resp.raise_for_status()

                result = resp.json()

        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, "text") else ""
            msg = f"Image edit failed (HTTP {e.response.status_code}): {error_detail}"
            self.log(msg, "ERROR")
            raise ValueError(msg) from e
        except httpx.HTTPError as e:
            msg = f"Image edit failed: {e}"
            self.log(msg, "ERROR")
            raise ValueError(msg) from e

        # Extract image from response — gpt-image models always return b64_json
        image_items = result.get("data", [])
        if not image_items:
            msg = f"No image data in response: {result}"
            raise ValueError(msg)

        first_item = image_items[0]
        b64_data = first_item.get("b64_json", "")
        image_url = ""
        if b64_data:
            image_url = f"data:image/png;base64,{b64_data}"
        else:
            image_url = first_item.get("url", "")

        if not image_url:
            msg = f"No image data in response: {result}"
            raise ValueError(msg)

        self._generation_info = {
            "model": model_name,
            "mode": self.generation_mode,
            "prompt": prompt[:100],
            "size": size,
            "n": len(image_items),
            "image_url": image_url,
        }

        self.status = f"Image edited: {image_url[:80]}..."
        return Message(text=image_url, files=[Image(url=image_url)])

    def generate_image(self, *args) -> Message:
        """Generate an image using the selected model via OpenAI-compatible API."""
        # Delegate to batch mode if applicable
        if self.generation_mode == MODE_BATCH_JSON:
            return self.generate_batch(*args)

        api_key, base_url, model_name = self._resolve_credentials()

        # Get prompt
        prompt = self.input_value
        if isinstance(prompt, Message):
            prompt = prompt.get_text()
        if not prompt:
            msg = "Please provide a prompt for image generation"
            raise ValueError(msg)

        # Collect reference images if in Text + Image mode
        ref_urls: list[str] = []
        if self.generation_mode == MODE_TEXT_IMAGE:
            ref_urls = self._resolve_image_urls()

        # Route Gemini image models to native generateContent endpoint
        if self._is_gemini_image_model(model_name):
            return self._generate_via_gemini(api_key, base_url, model_name, prompt, ref_urls)

        # Route GPT Image models with reference images to /v1/images/edits endpoint
        if self._is_gpt_image_model(model_name) and ref_urls:
            return self._generate_via_gpt_image_edits(api_key, base_url, model_name, prompt, ref_urls)

        # --- Standard OpenAI images/generations flow ---
        is_gpt_image = self._is_gpt_image_model(model_name)

        if is_gpt_image:
            size = self._resolve_gpt_image_size()
        else:
            size = self._resolve_size()

        # Build request payload
        payload: dict = {
            "model": model_name,
            "prompt": prompt,
            "n": max(1, self.n),
            "size": size,
        }

        # gpt-image models do not support response_format; others use user setting
        if not is_gpt_image:
            payload["response_format"] = self.response_format

        # Add reference images for Text + Image(s) mode (non-GPT-Image models only)
        if ref_urls and not is_gpt_image:
            payload["image"] = ref_urls if len(ref_urls) > 1 else ref_urls[0]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = f"{base_url}images/generations"

        try:
            with httpx.Client(headers=headers, timeout=IMAGE_GEN_TIMEOUT, trust_env=False) as client:
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

        # Parse response — gpt-image models always return b64_json
        first_item = image_items[0]
        image_url = ""

        if is_gpt_image:
            b64_data = first_item.get("b64_json", "")
            if b64_data:
                image_url = f"data:image/png;base64,{b64_data}"
        else:
            image_url = first_item.get("url", "")
            if not image_url and self.response_format == "b64_json":
                b64_data = first_item.get("b64_json", "")
                if b64_data:
                    fmt = "png" if "png" in size else "jpeg"
                    image_url = f"data:image/{fmt};base64,{b64_data}"

        if not image_url:
            msg = f"No image URL in response: {data}"
            raise ValueError(msg)

        # Convert to base64 for display if external URL
        display_url = self._ensure_displayable(image_url)

        # Store generation info
        self._generation_info = {
            "model": model_name,
            "mode": self.generation_mode,
            "prompt": prompt[:100],
            "size": size,
            "n": len(image_items),
            "usage": data.get("usage"),
            "image_url": image_url,
        }

        self.status = f"Image generated: {image_url[:80]}..."
        return Message(text=display_url, files=[Image(url=display_url)])

    # ------------------------------------------------------------------
    # Batch JSON mode helpers
    # ------------------------------------------------------------------

    def _parse_batch_json(self, raw_text: str) -> list[dict]:
        """Parse JSON array or object text with fallback to regex extraction."""
        text = raw_text.strip()
        if not text:
            return []

        def _extract_list(parsed) -> list[dict]:
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
            if isinstance(parsed, dict):
                return [parsed]
            return []

        # Direct parse
        try:
            return _extract_list(json.loads(text))
        except (json.JSONDecodeError, TypeError):
            pass

        # JSON in code block
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            try:
                return _extract_list(json.loads(match.group(1)))
            except (json.JSONDecodeError, TypeError):
                pass

        # First JSON array or object in text
        match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
        if match:
            try:
                return _extract_list(json.loads(match.group(0)))
            except (json.JSONDecodeError, TypeError):
                pass

        return []

    @staticmethod
    def _detect_prompt_field(item: dict) -> tuple[str, str]:
        """Auto-detect which field to use as the image generation prompt.

        Returns (field_name, field_value). Falls back to the first
        string-valued field if no known prompt field is found.
        """
        prompt_candidates = [
            "prompt", "image_prompt", "description", "content",
            "text", "desc", "描述", "内容",
        ]
        for key in prompt_candidates:
            val = item.get(key)
            if isinstance(val, str) and val.strip():
                return key, val.strip()

        # Fall back to first string field
        for key, val in item.items():
            if isinstance(val, str) and val.strip() and key not in ("image_url", "name", "名字"):
                return key, val.strip()

        return "", ""

    def _generate_single_image(self, api_key: str, base_url: str, model_name: str, prompt: str) -> str:
        """Generate a single image and return its URL/data URI.

        Reuses the existing routing logic (Gemini / GPT-Image / standard).
        """
        # Route Gemini
        if self._is_gemini_image_model(model_name):
            msg = self._generate_via_gemini(api_key, base_url, model_name, prompt, [])
            return msg.get_text()

        # Route GPT-Image (no ref images in batch mode)
        if self._is_gpt_image_model(model_name):
            return self._generate_single_standard(api_key, base_url, model_name, prompt)

        return self._generate_single_standard(api_key, base_url, model_name, prompt)

    def _generate_single_standard(self, api_key: str, base_url: str, model_name: str, prompt: str) -> str:
        """Generate a single image via standard OpenAI images/generations endpoint."""
        is_gpt_image = self._is_gpt_image_model(model_name)

        if is_gpt_image:
            size = self._resolve_gpt_image_size()
        else:
            size = self._resolve_size()

        payload: dict = {
            "model": model_name,
            "prompt": prompt,
            "n": 1,
            "size": size,
        }

        if not is_gpt_image:
            payload["response_format"] = self.response_format

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        url = f"{base_url}images/generations"

        with httpx.Client(headers=headers, timeout=IMAGE_GEN_TIMEOUT, trust_env=False) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        image_items = data.get("data", [])
        if not image_items:
            msg = f"No image data in response: {data}"
            raise ValueError(msg)

        first_item = image_items[0]
        if is_gpt_image:
            b64_data = first_item.get("b64_json", "")
            if b64_data:
                return f"data:image/png;base64,{b64_data}"
        else:
            image_url = first_item.get("url", "")
            if image_url:
                return self._ensure_displayable(image_url)
            if self.response_format == "b64_json":
                b64_data = first_item.get("b64_json", "")
                if b64_data:
                    fmt = "png" if "png" in size else "jpeg"
                    return f"data:image/{fmt};base64,{b64_data}"

        msg = f"No image URL in response: {data}"
        raise ValueError(msg)

    def generate_batch(self, *args) -> Message:
        """Generate images for each item in a JSON array and return combined JSON result."""
        raw_input = (self.json_input or "").strip()

        if not raw_input:
            msg = "Please provide JSON input for batch generation"
            raise ValueError(msg)

        items = self._parse_batch_json(raw_input)
        if not items:
            msg = f"Could not parse JSON from input. First 200 chars: {raw_input[:200]}"
            raise ValueError(msg)

        api_key, base_url, model_name = self._resolve_credentials()
        results: list[dict] = []
        total = len(items)

        for i, item in enumerate(items):
            field_name, prompt = self._detect_prompt_field(item)
            if not prompt:
                item["image_url"] = ""
                item["_error"] = "No prompt field detected"
                results.append(item)
                continue

            self.status = f"Generating image {i + 1}/{total}: {prompt[:50]}..."
            self.log(f"Batch [{i + 1}/{total}] prompt (from '{field_name}'): {prompt[:80]}")

            try:
                image_url = self._generate_single_image(api_key, base_url, model_name, prompt)
                item["image_url"] = image_url
            except Exception as e:
                self.log(f"Batch [{i + 1}/{total}] failed: {e}", "ERROR")
                item["image_url"] = ""
                item["_error"] = str(e)

            results.append(item)

        self._generation_info = {
            "model": model_name,
            "mode": self.generation_mode,
            "total": total,
            "success": sum(1 for r in results if r.get("image_url")),
            "failed": sum(1 for r in results if not r.get("image_url")),
        }

        # Collect image files for frontend rendering
        image_files = [Image(url=r["image_url"]) for r in results if r.get("image_url")]

        self.status = f"Batch complete: {self._generation_info['success']}/{total} images generated"
        return Message(
            text=json.dumps(results, ensure_ascii=False, indent=2),
            files=image_files,
        )

    def get_generation_info(self, *args) -> Data:
        """Return generation information as Data."""
        if self._generation_info:
            return Data(data=self._generation_info)
        return Data(data={})
