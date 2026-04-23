"""Multimodal Language Model component - accepts text + images/videos/audios, outputs text."""

import base64

import httpx
from langchain_core.messages import HumanMessage, SystemMessage

from lfx.base.models.model import LCModelComponent
from lfx.base.models.unified_models import (
    apply_provider_variable_config_to_build_config,
    get_language_model_options,
    get_llm,
    get_provider_for_model_name,
    update_model_options_in_build_config,
)
from lfx.base.models.watsonx_constants import IBM_WATSONX_URLS
from lfx.field_typing import LanguageModel
from lfx.field_typing.range_spec import RangeSpec
from lfx.inputs.inputs import BoolInput, DropdownInput, StrInput
from lfx.io import (
    IntInput,
    MessageInput,
    ModelInput,
    MultilineInput,
    SecretStrInput,
    SliderInput,
)

from lfx.schema.message import Message

MEDIA_FORMAT_AUTO = "Auto"
MEDIA_FORMAT_OPENAI = "OpenAI"
MEDIA_FORMAT_ANTHROPIC = "Anthropic"
MEDIA_FORMAT_GOOGLE = "Google"
MEDIA_FORMAT_OPTIONS = [
    MEDIA_FORMAT_AUTO,
    MEDIA_FORMAT_OPENAI,
    MEDIA_FORMAT_ANTHROPIC,
    MEDIA_FORMAT_GOOGLE,
]

DEFAULT_OLLAMA_URL = "http://localhost:11434"

GEMINI_MODEL_TIMEOUT = 180

IMAGE_PREFIX = "image_"
VIDEO_PREFIX = "video_"
AUDIO_PREFIX = "audio_"


class MultimodalModelComponent(LCModelComponent):
    display_name = "Multimodal Model"
    description = "Generate text from text + images/videos/audios using a multimodal language model."
    documentation: str = "https://docs.langflow.org/components-models"
    icon = "brain-circuit"
    category = "models"
    name = "MultimodalModel"

    inputs = [
        ModelInput(
            name="model",
            display_name="Multimodal Model",
            info="Select your model provider (supports vision models like GPT-4o, Claude, Gemini)",
            real_time_refresh=True,
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="Model Provider API key",
            required=False,
            show=True,
            real_time_refresh=True,
            advanced=True,
        ),
        DropdownInput(
            name="base_url_ibm_watsonx",
            display_name="watsonx API Endpoint",
            info="The base URL of the API (IBM watsonx.ai only)",
            options=IBM_WATSONX_URLS,
            value=IBM_WATSONX_URLS[0],
            show=False,
            real_time_refresh=True,
        ),
        StrInput(
            name="project_id",
            display_name="watsonx Project ID",
            info="The project ID associated with the foundation model (IBM watsonx.ai only)",
            show=False,
            required=False,
        ),
        StrInput(
            name="ollama_base_url",
            display_name="Ollama API URL",
            info=f"Endpoint of the Ollama API (Ollama only). Defaults to {DEFAULT_OLLAMA_URL}",
            value=DEFAULT_OLLAMA_URL,
            show=False,
            real_time_refresh=True,
        ),
        StrInput(
            name="openai_base_url",
            display_name="OpenAI Base URL",
            info="Custom base URL for OpenAI API (optional, for proxies or compatible endpoints)",
            show=False,
            real_time_refresh=True,
        ),
        StrInput(
            name="anthropic_base_url",
            display_name="Anthropic Base URL",
            info="Custom base URL for Anthropic API (optional, for proxies or compatible endpoints)",
            show=False,
            real_time_refresh=True,
        ),
        StrInput(
            name="google_base_url",
            display_name="Google Base URL",
            info="Custom base URL for Google Generative AI API (optional, for proxies or compatible endpoints)",
            show=False,
            real_time_refresh=True,
        ),
        StrInput(
            name="newapi_base_url",
            display_name="NewAPI Base URL",
            info="Base URL of your NewAPI gateway (NewAPI only)",
            show=False,
            real_time_refresh=True,
        ),
        MessageInput(
            name="input_value",
            display_name="Text Input",
            info="The text prompt to send to the model",
        ),
        MultilineInput(
            name="system_message",
            display_name="System Message",
            info="A system message that helps set the behavior of the assistant",
            advanced=False,
        ),
        # --- Dynamic media inputs ---
        IntInput(
            name="image_count",
            display_name="Image Count",
            info="Number of image inputs. Change to add or remove image entries.",
            value=0,
            real_time_refresh=True,
        ),
        IntInput(
            name="video_count",
            display_name="Video Count",
            info="Number of video inputs. Change to add or remove video entries.",
            value=0,
            real_time_refresh=True,
        ),
        IntInput(
            name="audio_count",
            display_name="Audio Count",
            info="Number of audio inputs. Change to add or remove audio entries.",
            value=0,
            real_time_refresh=True,
        ),
        DropdownInput(
            name="media_format",
            display_name="Media Format",
            info=(
                "Choose the content format for multimodal media. "
                "Auto: native image_url/video_url/audio_url types. "
                "OpenAI / Anthropic / Google: build content in the "
                "corresponding provider's format — useful when routing "
                "through NewAPI or similar gateways."
            ),
            options=MEDIA_FORMAT_OPTIONS,
            value=MEDIA_FORMAT_AUTO,
            advanced=True,
        ),
        # --- Generation parameters ---
        BoolInput(
            name="stream",
            display_name="Stream",
            info="Whether to stream the response",
            value=False,
            advanced=True,
        ),
        SliderInput(
            name="temperature",
            display_name="Temperature",
            value=0.1,
            info="Controls randomness in responses",
            range_spec=RangeSpec(min=0, max=1, step=0.01),
            advanced=True,
        ),
        IntInput(
            name="max_tokens",
            display_name="Max Tokens",
            info="Maximum number of tokens to generate.",
            advanced=True,
            range_spec=RangeSpec(min=1, max=128000, step=1, step_type="int"),
        ),
    ]

    def build_model(self) -> LanguageModel:
        return get_llm(
            model=self.model,
            user_id=self.user_id,
            api_key=self.api_key,
            temperature=self.temperature,
            stream=self.stream,
            max_tokens=getattr(self, "max_tokens", None),
            watsonx_url=getattr(self, "base_url_ibm_watsonx", None),
            watsonx_project_id=getattr(self, "project_id", None),
            ollama_base_url=getattr(self, "ollama_base_url", None),
            openai_base_url=getattr(self, "openai_base_url", None),
            anthropic_base_url=getattr(self, "anthropic_base_url", None),
            google_base_url=getattr(self, "google_base_url", None),
            newapi_base_url=getattr(self, "newapi_base_url", None),
        )

    def update_build_config(self, build_config: dict, field_value: str, field_name: str | None = None):
        """Dynamically update build config with model options and media input fields."""
        # Update model options
        build_config = update_model_options_in_build_config(
            component=self,
            build_config=build_config,
            cache_key_prefix="multimodal_model_options",
            get_options_func=get_language_model_options,
            field_name=field_name,
            field_value=field_value,
        )

        current_model_value = field_value if field_name == "model" else build_config.get("model", {}).get("value")
        provider = ""
        if isinstance(current_model_value, list) and current_model_value:
            selected_model = current_model_value[0]
            provider = (selected_model.get("provider") or "").strip()
            if not provider and selected_model.get("name"):
                provider = get_provider_for_model_name(str(selected_model["name"]))

        if provider:
            build_config = apply_provider_variable_config_to_build_config(build_config, provider)

        # Handle dynamic media input fields
        if field_name in ("image_count", "video_count", "audio_count"):
            if field_name == "image_count":
                prefix, label, cap = IMAGE_PREFIX, "Image", 20
            elif field_name == "video_count":
                prefix, label, cap = VIDEO_PREFIX, "Video", 10
            else:
                prefix, label, cap = AUDIO_PREFIX, "Audio", 10

            count = min(cap, max(0, int(field_value))) if field_value else 0

            # Remove old dynamic fields for this prefix
            to_remove = [k for k in build_config if k.startswith(prefix) and k[len(prefix):].isdigit()]
            for k in to_remove:
                del build_config[k]

            for i in range(1, count + 1):
                f_name = f"{prefix}{i}"
                build_config[f_name] = {
                    "type": "str",
                    "input_types": ["Message", "Text"],
                    "name": f_name,
                    "display_name": f"{label} {i}",
                    "value": "",
                    "show": True,
                    "advanced": False,
                    "multiline": False,
                    "placeholder": f"Enter {label.lower()} URL or connect component...",
                }

        return build_config

    def _collect_urls(self, prefix: str) -> list[str]:
        """Collect URLs from dynamic input fields with the given prefix."""
        urls: list[str] = []
        i = 1
        while True:
            val = getattr(self, f"{prefix}{i}", None)
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

    def _is_gemini_model(self) -> bool:
        """Check if the selected model is a Gemini model accessed through NewAPI."""
        model_data = self.model
        if not model_data or not isinstance(model_data, list) or not model_data:
            return False
        provider = (model_data[0].get("provider") or "").strip()
        model_name = (model_data[0].get("name") or "").strip()
        # NewAPI proxies Gemini models; detect by provider name or model name
        is_newapi = provider.lower() in ("newapi", "new-api", "one-api", "oneapi")
        is_gemini = model_name.startswith("gemini")
        return is_newapi and is_gemini

    def _resolve_newapi_credentials(self) -> tuple[str, str, str]:
        """Resolve NewAPI credentials: (api_key, raw_base_url, model_name)."""
        from lfx.base.models.unified_models import (
            get_all_variables_for_provider,
            get_api_key_for_provider,
        )

        model_data = self.model
        model_info = model_data[0]
        model_name = model_info.get("name", "")
        provider = model_info.get("provider", "")

        api_key = get_api_key_for_provider(self.user_id, provider)
        if not api_key:
            msg = f"{provider} API key is required. Please configure it in Model Providers."
            raise ValueError(msg)

        base_url = None
        provider_vars = get_all_variables_for_provider(self.user_id, provider)
        for var_key, value in provider_vars.items():
            if "BASE_URL" in var_key and value:
                base_url = value
                break
        if not base_url:
            msg = f"{provider} Base URL is required. Please configure it in Model Providers."
            raise ValueError(msg)

        raw_base = base_url.rstrip("/")
        return api_key, raw_base, model_name

    async def _generate_via_gemini(self, prompt: str, system_message: str | None, image_urls: list[str], video_urls: list[str], audio_urls: list[str]) -> Message:
        """Generate text using Gemini models via the native generateContent endpoint."""
        api_key, raw_base, model_name = self._resolve_newapi_credentials()

        # Build parts
        parts: list[dict] = []
        if prompt:
            parts.append({"text": prompt})

        # Download all media and embed as inlineData
        # NewAPI with parameter pass-through supports image/video/audio
        all_media = [(image_urls, "image"), (video_urls, "video"), (audio_urls, "audio")]
        with httpx.Client(timeout=120, trust_env=False) as dl_client:
            for urls, kind in all_media:
                for url in urls:
                    try:
                        resp = dl_client.get(url, follow_redirects=True)
                        resp.raise_for_status()
                        mime = resp.headers.get("content-type", "application/octet-stream")
                        b64 = base64.b64encode(resp.content).decode()
                        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
                    except httpx.HTTPError as e:
                        msg = f"Failed to download {kind} {url}: {e}"
                        raise ValueError(msg) from e

        payload: dict = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {"responseModalities": ["TEXT"]},
        }
        if system_message:
            payload["systemInstruction"] = {"parts": [{"text": system_message}]}

        url = f"{raw_base}/v1beta/models/{model_name}:generateContent"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        self.status = "Generating via Gemini..."
        self.log(f"Gemini request to {url}, model: {model_name}, parts: {len(parts)}")

        async with httpx.AsyncClient(headers=headers, timeout=GEMINI_MODEL_TIMEOUT, trust_env=False) as client:
            resp = await client.post(url, json=payload)
            if not resp.is_success:
                error_text = resp.text
                msg = f"Gemini request failed (HTTP {resp.status_code}): {error_text}"
                raise ValueError(msg)
            data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            msg = f"No candidates in Gemini response: {data}"
            raise ValueError(msg)

        response_parts = candidates[0].get("content", {}).get("parts", [])
        text_result = ""
        for p in response_parts:
            if "text" in p:
                text_result += p["text"]

        if not text_result:
            msg = f"No text in Gemini response: {data}"
            raise ValueError(msg)

        self.status = text_result
        return Message(text=text_result)

    @staticmethod
    def _build_media_content(
        media_format: str,
        image_urls: list[str],
        video_urls: list[str],
        audio_urls: list[str],
    ) -> list[dict]:
        """Build multimodal content blocks in the chosen provider format."""
        blocks: list[dict] = []

        if media_format == MEDIA_FORMAT_OPENAI:
            # OpenAI: image_url for images, video/audio as image_url passthrough
            # (NewAPI will detect actual content type and convert)
            for url in image_urls:
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            for url in video_urls:
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            for url in audio_urls:
                blocks.append({"type": "image_url", "image_url": {"url": url}})

        elif media_format == MEDIA_FORMAT_ANTHROPIC:
            # Anthropic: {"type": "image", "source": {"type": "url", "url": ...}}
            for url in image_urls:
                blocks.append(
                    {"type": "image", "source": {"type": "url", "url": url}}
                )
            # Video/Audio: use same image block as passthrough for NewAPI conversion
            for url in video_urls:
                blocks.append(
                    {"type": "image", "source": {"type": "url", "url": url}}
                )
            for url in audio_urls:
                blocks.append(
                    {"type": "image", "source": {"type": "url", "url": url}}
                )

        elif media_format == MEDIA_FORMAT_GOOGLE:
            # Google Gemini: native video_url / audio_url types supported
            for url in image_urls:
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            for url in video_urls:
                blocks.append({"type": "video_url", "video_url": {"url": url}})
            for url in audio_urls:
                blocks.append({"type": "audio_url", "audio_url": {"url": url}})

        else:
            # Auto (default): native types per media kind
            for url in image_urls:
                blocks.append({"type": "image_url", "image_url": {"url": url}})
            for url in video_urls:
                blocks.append({"type": "video_url", "video_url": {"url": url}})
            for url in audio_urls:
                blocks.append({"type": "audio_url", "audio_url": {"url": url}})

        return blocks

    async def text_response(self) -> Message:
        """Generate text from multimodal input."""
        # Resolve text prompt
        prompt = self.input_value
        if isinstance(prompt, Message):
            prompt = prompt.get_text()
        prompt = (prompt or "").strip()

        # Collect media URLs
        image_urls = self._collect_urls(IMAGE_PREFIX)
        video_urls = self._collect_urls(VIDEO_PREFIX)
        audio_urls = self._collect_urls(AUDIO_PREFIX)

        if not prompt and not image_urls and not video_urls and not audio_urls:
            msg = "Please provide a text prompt or at least one media input."
            raise ValueError(msg)

        system_message = getattr(self, "system_message", None) or None

        # Route Gemini models (via NewAPI) to native generateContent endpoint
        if self._is_gemini_model():
            return await self._generate_via_gemini(prompt, system_message, image_urls, video_urls, audio_urls)

        # --- Standard LangChain flow for other providers ---
        runnable = self.build_model()

        # Build multimodal content list
        content: list[dict] = []
        if prompt:
            content.append({"type": "text", "text": prompt})

        media_format = getattr(self, "media_format", MEDIA_FORMAT_AUTO)
        content.extend(self._build_media_content(media_format, image_urls, video_urls, audio_urls))

        result = await self._get_chat_result(
            runnable=runnable,
            stream=self.stream,
            input_value=content if len(content) > 1 else (content[0] if content else ""),
            system_message=getattr(self, "system_message", None) or None,
        )
        self.status = result
        return result

    async def _get_chat_result(
        self,
        *,
        runnable: LanguageModel,
        stream: bool,
        input_value,
        system_message: str | None = None,
    ) -> Message:
        """Invoke the model with multimodal content."""
        messages = []
        if system_message:
            messages.append(SystemMessage(content=system_message))

        # input_value can be a list of content dicts or a string
        if isinstance(input_value, list | str):
            messages.append(HumanMessage(content=input_value))
        elif isinstance(input_value, Message):
            messages.append(input_value.to_lc_message(self.name))

        runnable = runnable.with_config(
            {
                "run_name": self.display_name,
                "project_name": self.get_project_name(),
                "callbacks": self.get_langchain_callbacks(),
            }
        )

        if stream:
            lf_message, result = await self._handle_stream(runnable, messages)
        else:
            ai_message = await runnable.ainvoke(messages)
            result = ai_message.content if hasattr(ai_message, "content") else ai_message

        if stream and lf_message:
            return lf_message

        return Message(text=result)
