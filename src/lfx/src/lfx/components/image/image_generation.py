import base64
import json
import logging

import httpx

from lfx.base.models.unified_models import (
    get_image_model_options,
    update_model_options_in_build_config,
)
from lfx.custom import Component
from lfx.inputs import (
    DropdownInput,
    IntInput,
)
from lfx.io import MessageInput, ModelInput, MultilineInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message

logger = logging.getLogger(__name__)

# Default timeout for image generation requests (5 minutes)
IMAGE_GEN_TIMEOUT = 300
SIZE_OPTION_PART_COUNT = 3
MAX_REFERENCE_IMAGES = 14

MODE_TEXT = "Text to Image"
MODE_TEXT_IMAGE = "Text + Image(s)"

MODE_OPTIONS = [MODE_TEXT, MODE_TEXT_IMAGE]

REF_IMAGE_PREFIX = "ref_image_"
REFERENCE_IMAGE_URLS_FIELD = "reference_image_urls"

# Gemini models that generate images via chat completions endpoint
GEMINI_IMAGE_KEYWORDS = ("flash-image", "image-generation", "pro-image-preview")
GEMINI_31_FLASH_IMAGE_KEYWORDS = ("gemini-3-1-flash-image-preview", "gemini31flashimagepreview")
GEMINI_3_PRO_IMAGE_KEYWORDS = ("gemini-3-pro-image-preview", "gemini3proimagepreview")

# GPT Image models that use /v1/images/edits for reference image editing
GPT_IMAGE_KEYWORDS = ("gpt-image", "gptimage")
GPT_IMAGE_2_MODEL_KEYWORDS = ("gpt-image-2", "gptimage2")
SEEDREAM_5_MODEL_KEYWORDS = ("seedream-5",)

DEFAULT_SEEDREAM_SIZE = "2K | 1:1 | 2048x2048"
DEFAULT_GPT_IMAGE_2_SIZE = "auto"
DEFAULT_GEMINI_IMAGE_SIZE = "1K | 1:1 | 1024x1024"
DEFAULT_GENERIC_SIZE = "1024x1024"

SEEDREAM_5_SIZE_OPTIONS = [
    "2K | 1:1 | 2048x2048",
    "2K | 4:3 | 2304x1728",
    "2K | 3:4 | 1728x2304",
    "2K | 16:9 | 2848x1600",
    "2K | 9:16 | 1600x2848",
    "2K | 3:2 | 2496x1664",
    "2K | 2:3 | 1664x2496",
    "2K | 21:9 | 3136x1344",
    "3K | 1:1 | 3072x3072",
    "3K | 4:3 | 3456x2592",
    "3K | 3:4 | 2592x3456",
    "3K | 16:9 | 4096x2304",
    "3K | 9:16 | 2304x4096",
    "3K | 2:3 | 2496x3744",
    "3K | 3:2 | 3744x2496",
    "3K | 21:9 | 4704x2016",
    "4K | 1:1 | 4096x4096",
    "4K | 3:4 | 3520x4704",
    "4K | 4:3 | 4704x3520",
    "4K | 16:9 | 5504x3040",
    "4K | 9:16 | 3040x5504",
    "4K | 2:3 | 3328x4992",
    "4K | 3:2 | 4992x3328",
    "4K | 21:9 | 6240x2656",
]

GPT_IMAGE_2_COMMON_SIZE_OPTIONS = [
    "auto",
    "1K | 1:1 | 1024x1024",
    "1K | 3:2 | 1536x1024",
    "1K | 2:3 | 1024x1536",
    "2K | 1:1 | 2048x2048",
    "2K | 16:9 | 2048x1152",
    "4K | 16:9 | 3840x2160",
    "4K | 9:16 | 2160x3840",
]

GEMINI_31_FLASH_IMAGE_SIZE_OPTIONS = [
    "512 | 1:1 | 512x512",
    "512 | 1:4 | 256x1024",
    "512 | 1:8 | 192x1536",
    "512 | 2:3 | 424x632",
    "512 | 3:2 | 632x424",
    "512 | 3:4 | 448x600",
    "512 | 4:1 | 1024x256",
    "512 | 4:3 | 600x448",
    "512 | 4:5 | 464x576",
    "512 | 5:4 | 576x464",
    "512 | 8:1 | 1536x192",
    "512 | 9:16 | 384x688",
    "512 | 16:9 | 688x384",
    "512 | 21:9 | 792x168",
    "1K | 1:1 | 1024x1024",
    "1K | 1:4 | 512x2048",
    "1K | 1:8 | 384x3072",
    "1K | 2:3 | 848x1264",
    "1K | 3:2 | 1264x848",
    "1K | 3:4 | 896x1200",
    "1K | 4:1 | 2048x512",
    "1K | 4:3 | 1200x896",
    "1K | 4:5 | 928x1152",
    "1K | 5:4 | 1152x928",
    "1K | 8:1 | 3072x384",
    "1K | 9:16 | 768x1376",
    "1K | 16:9 | 1376x768",
    "1K | 21:9 | 1584x672",
    "2K | 1:1 | 2048x2048",
    "2K | 1:4 | 1024x4096",
    "2K | 1:8 | 768x6144",
    "2K | 2:3 | 1696x2528",
    "2K | 3:2 | 2528x1696",
    "2K | 3:4 | 1792x2400",
    "2K | 4:1 | 4096x1024",
    "2K | 4:3 | 2400x1792",
    "2K | 4:5 | 1856x2304",
    "2K | 5:4 | 2304x1856",
    "2K | 8:1 | 6144x768",
    "2K | 9:16 | 1536x2752",
    "2K | 16:9 | 2752x1536",
    "2K | 21:9 | 3168x1344",
    "4K | 1:1 | 4096x4096",
    "4K | 1:4 | 2048x8192",
    "4K | 1:8 | 1536x12288",
    "4K | 2:3 | 3392x5056",
    "4K | 3:2 | 5056x3392",
    "4K | 3:4 | 3584x4800",
    "4K | 4:1 | 8192x2048",
    "4K | 4:3 | 4800x3584",
    "4K | 4:5 | 3712x4608",
    "4K | 5:4 | 4608x3712",
    "4K | 8:1 | 12288x1536",
    "4K | 9:16 | 3072x5504",
    "4K | 16:9 | 5504x3072",
    "4K | 21:9 | 6336x2688",
]

GEMINI_3_PRO_IMAGE_SIZE_OPTIONS = [
    "1K | 1:1 | 1024x1024",
    "1K | 2:3 | 848x1264",
    "1K | 3:2 | 1264x848",
    "1K | 3:4 | 896x1200",
    "1K | 4:3 | 1200x896",
    "1K | 4:5 | 928x1152",
    "1K | 5:4 | 1152x928",
    "1K | 9:16 | 768x1376",
    "1K | 16:9 | 1376x768",
    "1K | 21:9 | 1584x672",
    "2K | 1:1 | 2048x2048",
    "2K | 2:3 | 1696x2528",
    "2K | 3:2 | 2528x1696",
    "2K | 3:4 | 1792x2400",
    "2K | 4:3 | 2400x1792",
    "2K | 4:5 | 1856x2304",
    "2K | 5:4 | 2304x1856",
    "2K | 9:16 | 1536x2752",
    "2K | 16:9 | 2752x1536",
    "2K | 21:9 | 3168x1344",
    "4K | 1:1 | 4096x4096",
    "4K | 2:3 | 3392x5056",
    "4K | 3:2 | 5056x3392",
    "4K | 3:4 | 3584x4800",
    "4K | 4:3 | 4800x3584",
    "4K | 4:5 | 3712x4608",
    "4K | 5:4 | 4608x3712",
    "4K | 9:16 | 3072x5504",
    "4K | 16:9 | 5504x3072",
    "4K | 21:9 | 6336x2688",
]

DEFAULT_SIZE_OPTIONS = [
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "2048x2048",
    "2048x1152",
]


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
        DropdownInput(
            name="generation_mode",
            display_name="Generation Mode",
            info="Select image generation mode.",
            options=MODE_OPTIONS,
            value=MODE_TEXT,
            real_time_refresh=True,
        ),
        MultilineInput(
            name=REFERENCE_IMAGE_URLS_FIELD,
            display_name="Reference Image URLs",
            info='Reference image URLs in JSON array format. Supports up to 14 images, for example ["https://example.com/a.png", "https://example.com/b.png"].',
            placeholder='["https://example.com/a.png", "https://example.com/b.png"]',
            real_time_refresh=True,
            show=False,
        ),
        # --- Generation parameters ---
        DropdownInput(
            name="image_size",
            display_name="Size",
            info="Output size. Options change according to the selected image model.",
            options=DEFAULT_SIZE_OPTIONS,
            value=DEFAULT_GENERIC_SIZE,
            real_time_refresh=True,
        ),
        DropdownInput(
            name="resolution",
            display_name="Resolution",
            info="Output image resolution level.",
            options=["1K", "2K", "4K"],
            value="2K",
            advanced=True,
            show=False,
        ),
        DropdownInput(
            name="ratio",
            display_name="Aspect Ratio",
            info="Output image aspect ratio.",
            options=["1:1", "16:9", "9:16", "4:3", "3:4", "21:9"],
            value="1:1",
            advanced=True,
            show=False,
        ),
        IntInput(
            name="n",
            display_name="Number of Images",
            info="Number of images to generate.",
            value=1,
        ),
        DropdownInput(
            name="response_format",
            display_name="Response Format",
            info="How the API returns the generated image.",
            options=["url", "b64_json"],
            value="url",
            advanced=True,
            show=False,
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
            get_options_func=get_image_model_options,
            field_name=field_name,
            field_value=field_value,
        )

        if field_name in {"model", None}:
            self._update_generation_parameter_options(
                build_config,
                field_value if field_name == "model" else None,
            )

        mode_value = (
            field_value
            if field_name == "generation_mode"
            else build_config.get("generation_mode", {}).get("value", MODE_TEXT)
        )
        if REFERENCE_IMAGE_URLS_FIELD in build_config:
            build_config[REFERENCE_IMAGE_URLS_FIELD]["show"] = mode_value == MODE_TEXT_IMAGE

        if field_name == "generation_mode":
            # Remove old dynamic ref image fields when switching modes
            to_remove = [
                k for k in build_config if k.startswith(REF_IMAGE_PREFIX) and k[len(REF_IMAGE_PREFIX) :].isdigit()
            ]
            for k in to_remove:
                del build_config[k]

        return build_config

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        return model_name.lower().replace("_", "-").replace(".", "-").replace(" ", "-")

    @classmethod
    def _get_model_name_from_value(cls, model_value) -> str:
        if not isinstance(model_value, list) or not model_value:
            return ""
        first = model_value[0]
        if not isinstance(first, dict):
            return ""
        return str(first.get("name") or "")

    @classmethod
    def _is_seedream_5_model(cls, model_name: str) -> bool:
        normalized = cls._normalize_model_name(model_name)
        return any(keyword in normalized for keyword in SEEDREAM_5_MODEL_KEYWORDS)

    @classmethod
    def _is_gemini_31_flash_image_model(cls, model_name: str) -> bool:
        normalized = cls._normalize_model_name(model_name)
        compact = normalized.replace("-", "")
        return any(keyword in normalized or keyword in compact for keyword in GEMINI_31_FLASH_IMAGE_KEYWORDS)

    @classmethod
    def _is_gemini_3_pro_image_model(cls, model_name: str) -> bool:
        normalized = cls._normalize_model_name(model_name)
        compact = normalized.replace("-", "")
        return any(keyword in normalized or keyword in compact for keyword in GEMINI_3_PRO_IMAGE_KEYWORDS)

    @staticmethod
    def _is_gpt_image_2_model(model_name: str) -> bool:
        normalized = ImageGenerationComponent._normalize_model_name(model_name)
        compact = normalized.replace("-", "")
        return any(keyword in normalized or keyword in compact for keyword in GPT_IMAGE_2_MODEL_KEYWORDS)

    def _update_generation_parameter_options(self, build_config: dict, model_value=None) -> None:
        model_value = model_value if model_value is not None else build_config.get("model", {}).get("value")
        model_name = self._get_model_name_from_value(model_value)

        if self._is_seedream_5_model(model_name):
            options = SEEDREAM_5_SIZE_OPTIONS
            default = DEFAULT_SEEDREAM_SIZE
        elif self._is_gemini_31_flash_image_model(model_name):
            options = GEMINI_31_FLASH_IMAGE_SIZE_OPTIONS
            default = DEFAULT_GEMINI_IMAGE_SIZE
        elif self._is_gemini_3_pro_image_model(model_name):
            options = GEMINI_3_PRO_IMAGE_SIZE_OPTIONS
            default = DEFAULT_GEMINI_IMAGE_SIZE
        elif self._is_gpt_image_2_model(model_name):
            options = GPT_IMAGE_2_COMMON_SIZE_OPTIONS
            default = DEFAULT_GPT_IMAGE_2_SIZE
        else:
            options = DEFAULT_SIZE_OPTIONS
            default = DEFAULT_GENERIC_SIZE

        if "image_size" in build_config:
            current_value = build_config["image_size"].get("value")
            build_config["image_size"]["options"] = options
            build_config["image_size"]["value"] = current_value if current_value in options else default
            build_config["image_size"]["show"] = True
            build_config["image_size"]["advanced"] = False

        if "n" in build_config:
            build_config["n"]["show"] = True
            build_config["n"]["advanced"] = False

        if "resolution" in build_config:
            build_config["resolution"]["show"] = False

        if "ratio" in build_config:
            build_config["ratio"]["show"] = False

        if "response_format" in build_config:
            build_config["response_format"]["show"] = False
            build_config["response_format"]["advanced"] = True

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
            msg = f"{provider} API key is required. Please configure it in Model Providers."
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

    @classmethod
    def _extract_reference_url_values(cls, value) -> list[str]:
        """Normalize list, Message, Data, JSON-string, or newline text into URL strings."""
        if value is None:
            return []

        if isinstance(value, Message):
            return cls._extract_reference_url_values(value.get_text())

        if isinstance(value, Data):
            values: list[str] = []
            for key in ("urls", "url", "image_urls", "images", "text"):
                if key in value.data:
                    values.extend(cls._extract_reference_url_values(value.data[key]))
            if values:
                return values
            return cls._extract_reference_url_values(value.get_text())

        if isinstance(value, dict):
            values: list[str] = []
            for key in ("urls", "url", "image_urls", "images", "text"):
                if key in value:
                    values.extend(cls._extract_reference_url_values(value[key]))
            return values

        if isinstance(value, (list, tuple, set)):
            values: list[str] = []
            for item in value:
                values.extend(cls._extract_reference_url_values(item))
            return values

        text = str(value).strip()
        if not text:
            return []

        if text[0] in "[{":
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as e:
                msg = 'Reference Image URLs must be a valid JSON array, e.g. ["url1", "url2"]'
                raise ValueError(msg) from e
            if not isinstance(parsed, list):
                msg = 'Reference Image URLs must be a JSON array, e.g. ["url1", "url2"]'
                raise ValueError(msg)
            return cls._extract_reference_url_values(parsed)

        values = []
        for line in text.splitlines():
            for part in line.split(","):
                cleaned = part.strip().strip("\"'")
                if cleaned:
                    values.append(cleaned)
        return values

    @staticmethod
    def _dedupe_urls(urls: list[str]) -> list[str]:
        seen: set[str] = set()
        unique_urls: list[str] = []
        for url in urls:
            if url in seen:
                continue
            seen.add(url)
            unique_urls.append(url)
        return unique_urls

    def _resolve_image_urls(self) -> list[str]:
        """Collect reference image URLs from the list input and legacy dynamic fields."""
        urls: list[str] = self._extract_reference_url_values(getattr(self, REFERENCE_IMAGE_URLS_FIELD, None))

        # Backward compatibility for existing flows created before the list input.
        i = 1
        while True:
            val = getattr(self, f"{REF_IMAGE_PREFIX}{i}", None)
            if val is None:
                break
            urls.extend(self._extract_reference_url_values(val))
            i += 1

        urls = self._dedupe_urls(urls)
        if len(urls) > MAX_REFERENCE_IMAGES:
            msg = f"Reference Image URLs supports at most {MAX_REFERENCE_IMAGES} images, got {len(urls)}."
            raise ValueError(msg)
        return urls

    @staticmethod
    def _extract_api_size(size_value: str) -> str:
        value = size_value.strip()
        if value == "auto":
            return value
        if "|" in value:
            return value.rsplit("|", 1)[-1].strip()
        return value.split()[0]

    @staticmethod
    def _extract_image_size_config(size_value: str) -> tuple[str | None, str | None]:
        value = size_value.strip()
        if "|" not in value:
            return None, None
        parts = [part.strip() for part in value.split("|")]
        if len(parts) < SIZE_OPTION_PART_COUNT:
            return None, None
        image_size, aspect_ratio = parts[0], parts[1]
        return image_size, aspect_ratio

    def _resolve_size(self) -> str:
        """Compute actual size string from resolution + aspect ratio."""
        image_size = getattr(self, "image_size", None)
        if isinstance(image_size, str) and image_size.strip():
            return self._extract_api_size(image_size)

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
        image_size = getattr(self, "image_size", None)
        if isinstance(image_size, str) and image_size.strip():
            return self._extract_api_size(image_size)

        landscape = {"16:9", "4:3", "21:9"}
        portrait = {"9:16", "3:4"}
        if self.ratio in landscape:
            return "1536x1024"
        if self.ratio in portrait:
            return "1024x1536"
        return "1024x1024"

    def _resolve_gemini_image_config(self) -> tuple[str | None, str | None]:
        image_size = getattr(self, "image_size", None)
        if isinstance(image_size, str) and image_size.strip():
            return self._extract_image_size_config(image_size)
        return None, None

    @staticmethod
    def _is_gemini_image_model(model_name: str) -> bool:
        """Check if the model is a Gemini image generation model."""
        return any(kw in model_name for kw in GEMINI_IMAGE_KEYWORDS)

    @staticmethod
    def _is_gpt_image_model(model_name: str) -> bool:
        """Check if the model is a GPT Image model (gpt-image-1, gpt-image-2, etc.)."""
        name_lower = ImageGenerationComponent._normalize_model_name(model_name)
        compact = name_lower.replace("-", "")
        return any(kw in name_lower or kw in compact for kw in GPT_IMAGE_KEYWORDS)

    @staticmethod
    def _response_rejects_parameter(response: httpx.Response, parameter: str) -> bool:
        if response.status_code != 400:
            return False
        text = response.text.lower()
        parameter = parameter.lower()
        return parameter in text and ("unknown parameter" in text or "unknown_parameter" in text)

    @staticmethod
    def _download_as_based64(url: str) -> tuple[str, str]:
        """Download a URL and return (base64_data, mime_type)."""
        with httpx.Client(timeout=IMAGE_GEN_TIMEOUT, trust_env=False) as client:
            resp = client.get(url, follow_redirects=True)
            resp.raise_for_status()
            mime = resp.headers.get("content-type", "image/jpeg")
            b64 = base64.b64encode(resp.content).decode()
        return b64, mime

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
        image_size, aspect_ratio = self._resolve_gemini_image_config()
        if image_size and aspect_ratio:
            payload["generationConfig"]["responseFormat"] = {
                "image": {
                    "aspectRatio": aspect_ratio,
                    "imageSize": image_size,
                }
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
            "reference_image_count": len(ref_urls),
            "image_size": image_size,
            "aspect_ratio": aspect_ratio,
            "image_url": image_data_url,
        }

        self.status = f"Image generated: {image_data_url[:80]}..."
        return Message(text=image_data_url)

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
        # Do not set Content-Type here. httpx must add the multipart boundary
        # when files are sent, otherwise JSON parsers see the multipart prefix
        # and fail with errors such as "invalid character '-' in numeric literal".
        headers = {"Authorization": f"Bearer {api_key}"}

        try:
            with httpx.Client(headers=headers, timeout=IMAGE_GEN_TIMEOUT, trust_env=False) as client:
                self.status = "Editing image via GPT Image..."
                self.log(f"Submitting edit request to {url}, model: {model_name}")

                resp = client.post(url, files=files, data=data)
                if (
                    not resp.is_success
                    and "response_format" in data
                    and self._response_rejects_parameter(resp, "response_format")
                ):
                    self.log("Gateway rejected response_format; retrying edit request without it.")
                    retry_data = data.copy()
                    retry_data.pop("response_format", None)
                    resp = client.post(url, files=files, data=retry_data)

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
        image_url = f"data:image/png;base64,{b64_data}" if b64_data else first_item.get("url", "")

        if not image_url:
            msg = f"No image data in response: {result}"
            raise ValueError(msg)

        self._generation_info = {
            "model": model_name,
            "mode": self.generation_mode,
            "prompt": prompt[:100],
            "reference_image_count": len(ref_urls),
            "size": size,
            "n": len(image_items),
            "image_url": image_url,
        }

        self.status = f"Image edited: {image_url[:80]}..."
        return Message(text=image_url)

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

        # Collect reference images if in Text + Image mode
        ref_urls: list[str] = []
        if self.generation_mode == MODE_TEXT_IMAGE:
            ref_urls = self._resolve_image_urls()

        # Route Gemini image models to native generateContent endpoint
        if self._is_gemini_image_model(model_name):
            return self._generate_via_gemini(api_key, base_url, model_name, prompt, ref_urls)

        # Route GPT Image models with reference images to /v1/images/edits endpoint.
        if self._is_gpt_image_model(model_name) and ref_urls:
            return self._generate_via_gpt_image_edits(api_key, base_url, model_name, prompt, ref_urls)

        # --- Standard OpenAI images/generations flow ---
        is_gpt_image = self._is_gpt_image_model(model_name)
        size = self._resolve_gpt_image_size() if is_gpt_image else self._resolve_size()

        # Build request payload
        payload: dict = {
            "model": model_name,
            "prompt": prompt,
            "n": max(1, self.n),
            "size": size,
        }

        # Prefer URL responses for generic OpenAI-compatible image models.
        # GPT Image gateways may reject response_format while returning b64_json by default.
        if not is_gpt_image:
            payload["response_format"] = self.response_format

        # Add reference images for Text + Image(s) mode (non-GPT-Image models only).
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
                if (
                    not resp.is_success
                    and "response_format" in payload
                    and self._response_rejects_parameter(resp, "response_format")
                ):
                    self.log("Gateway rejected response_format; retrying generation request without it.")
                    retry_payload = payload.copy()
                    retry_payload.pop("response_format", None)
                    resp = client.post(url, json=retry_payload)
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
        else:
            image_url = first_item.get("url", "")
            if not image_url:
                b64_data = first_item.get("b64_json", "")
                if b64_data:
                    fmt = "png" if "png" in size else "jpeg"
                    image_url = f"data:image/{fmt};base64,{b64_data}"

        if not image_url:
            msg = f"No image URL in response: {data}"
            raise ValueError(msg)

        # Store generation info
        self._generation_info = {
            "model": model_name,
            "mode": self.generation_mode,
            "prompt": prompt[:100],
            "reference_image_count": len(ref_urls),
            "size": size,
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
