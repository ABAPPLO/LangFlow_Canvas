import time

import httpx

from lfx.base.models.unified_models import (
    get_language_model_options,
    update_model_options_in_build_config,
)
from lfx.custom import Component
from lfx.inputs import (
    BoolInput,
    DropdownInput,
    IntInput,
    MultilineInput,
)
from lfx.io import MessageInput, ModelInput, Output
from lfx.schema.data import Data
from lfx.schema.message import Message

MODE_TEXT = "Text to Video"
MODE_IMAGE = "Image to Video"
MODE_FIRST_LAST = "First & Last Frame"
MODE_MULTIMODAL = "Multimodal"

MODE_OPTIONS = [MODE_TEXT, MODE_IMAGE, MODE_FIRST_LAST, MODE_MULTIMODAL]

REF_IMAGE_PREFIX = "ref_image_"
REF_VIDEO_PREFIX = "ref_video_"
REF_AUDIO_PREFIX = "ref_audio_"


class TaskError(Exception):
    """Error raised when a video generation task fails."""


class TaskTimeoutError(Exception):
    """Error raised when a video generation task times out."""


class VideoGenerationComponent(Component):
    display_name = "Video Generation"
    description = "Generate videos using OpenAI-compatible model providers (NewAPI, etc.)."
    icon = "Video"
    name = "VideoGeneration"

    inputs = [
        ModelInput(
            name="model",
            display_name="Video Model",
            info="Select a video generation model from your configured providers.",
            real_time_refresh=True,
            required=True,
        ),
        MessageInput(
            name="input_value",
            display_name="Prompt",
            info="Text prompt for video generation.",
        ),
        DropdownInput(
            name="generation_mode",
            display_name="Generation Mode",
            info="Select video generation mode.",
            options=MODE_OPTIONS,
            value=MODE_TEXT,
            real_time_refresh=True,
        ),
        # --- Image to Video / First & Last Frame ---
        MultilineInput(
            name="image_url",
            display_name="First Frame Image URL",
            info="URL of the first frame image.",
            dynamic=True,
            show=False,
        ),
        MultilineInput(
            name="last_frame_url",
            display_name="Last Frame Image URL",
            info="URL of the last frame image.",
            dynamic=True,
            show=False,
        ),
        # --- Multimodal reference counts ---
        IntInput(
            name="ref_image_count",
            display_name="Ref Image Count",
            info="Number of reference image inputs (max 9).",
            value=1,
            real_time_refresh=True,
            dynamic=True,
            show=False,
        ),
        IntInput(
            name="ref_video_count",
            display_name="Ref Video Count",
            info="Number of reference video inputs (max 3).",
            value=1,
            real_time_refresh=True,
            dynamic=True,
            show=False,
        ),
        IntInput(
            name="ref_audio_count",
            display_name="Ref Audio Count",
            info="Number of reference audio inputs (max 3).",
            value=1,
            real_time_refresh=True,
            dynamic=True,
            show=False,
        ),
        # --- Generation parameters ---
        DropdownInput(
            name="resolution",
            display_name="Resolution",
            info="Output video resolution.",
            options=["720p", "480p"],
            value="720p",
            advanced=True,
        ),
        DropdownInput(
            name="ratio",
            display_name="Aspect Ratio",
            info="Output video aspect ratio.",
            options=["adaptive", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"],
            value="adaptive",
            advanced=True,
        ),
        DropdownInput(
            name="duration",
            display_name="Duration (s)",
            info="Video duration in seconds. -1 for auto.",
            options=["5", "4", "6", "8", "10", "11", "12", "15", "-1"],
            value="5",
            advanced=True,
        ),
        BoolInput(
            name="generate_audio",
            display_name="Generate Audio",
            info="Generate audio for the video.",
            value=False,
            advanced=True,
        ),
        # --- Polling settings ---
        IntInput(
            name="poll_interval",
            display_name="Poll Interval (s)",
            info="Seconds between status checks.",
            value=5,
            advanced=True,
        ),
        IntInput(
            name="max_wait_time",
            display_name="Max Wait Time (s)",
            info="Maximum seconds to wait for task completion.",
            value=1800,
            advanced=True,
        ),
    ]

    outputs = [
        Output(
            display_name="Video URL",
            name="video_url",
            method="generate_video",
        ),
        Output(
            display_name="Task Info",
            name="task_info",
            method="get_task_info",
        ),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._task_id: str | None = None
        self._task_info: dict | None = None

    def update_build_config(self, build_config, field_value, field_name=None):
        # Load model options from Model Providers
        build_config = update_model_options_in_build_config(
            component=self,
            build_config=build_config,
            cache_key_prefix="video_model_options",
            get_options_func=get_language_model_options,
            field_name=field_name,
            field_value=field_value,
        )

        # Handle mode switching
        if field_name == "generation_mode":
            mode = field_value

            # Show/hide mode-specific fields
            build_config["image_url"]["show"] = mode in (MODE_IMAGE, MODE_FIRST_LAST)
            build_config["last_frame_url"]["show"] = mode == MODE_FIRST_LAST

            is_multimodal = mode == MODE_MULTIMODAL
            build_config["ref_image_count"]["show"] = is_multimodal
            build_config["ref_video_count"]["show"] = is_multimodal
            build_config["ref_audio_count"]["show"] = is_multimodal

            # Clear all dynamic ref fields
            for prefix in (REF_IMAGE_PREFIX, REF_VIDEO_PREFIX, REF_AUDIO_PREFIX):
                to_remove = [k for k in build_config if k.startswith(prefix) and k[len(prefix):].isdigit()]
                for k in to_remove:
                    del build_config[k]

            # Re-create dynamic fields for multimodal mode
            if is_multimodal:
                self._create_dynamic_ref_fields(build_config)

        # Handle ref count changes
        if field_name in ("ref_image_count", "ref_video_count", "ref_audio_count"):
            count = max(0, int(field_value)) if field_value else 0

            if field_name == "ref_image_count":
                prefix, label, cap = REF_IMAGE_PREFIX, "Image", 9
            elif field_name == "ref_video_count":
                prefix, label, cap = REF_VIDEO_PREFIX, "Video", 3
            else:
                prefix, label, cap = REF_AUDIO_PREFIX, "Audio", 3

            count = min(count, cap)

            # Remove old fields for this prefix
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
                    "placeholder": f"Enter {label.lower()} URL or connect...",
                }

        return build_config

    def _create_dynamic_ref_fields(self, build_config) -> None:
        """Create dynamic reference fields based on current count values."""
        for prefix, count_key, label, cap in [
            (REF_IMAGE_PREFIX, "ref_image_count", "Image", 9),
            (REF_VIDEO_PREFIX, "ref_video_count", "Video", 3),
            (REF_AUDIO_PREFIX, "ref_audio_count", "Audio", 3),
        ]:
            count = min(
                max(0, int(build_config.get(count_key, {}).get("value", 1))),
                cap,
            )
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
                    "placeholder": f"Enter {label.lower()} URL or connect...",
                }

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

        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

        return api_key, base_url + "/", model_name

    def _collect_ref_urls(self, prefix: str) -> list[str]:
        """Collect URLs from dynamic reference fields with the given prefix."""
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

    def _build_content(self) -> list[dict]:
        """Build the content list based on the selected generation mode."""
        content: list[dict] = []
        mode = getattr(self, "generation_mode", MODE_TEXT)

        # Text prompt is included in all modes
        prompt = self.input_value
        if isinstance(prompt, Message):
            prompt = prompt.get_text()
        if prompt:
            content.append({"type": "text", "text": prompt})

        if mode == MODE_IMAGE:
            if self.image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self.image_url.strip()},
                    "role": "first_frame",
                })

        elif mode == MODE_FIRST_LAST:
            if self.image_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self.image_url.strip()},
                    "role": "first_frame",
                })
            if self.last_frame_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self.last_frame_url.strip()},
                    "role": "last_frame",
                })

        elif mode == MODE_MULTIMODAL:
            for url in self._collect_ref_urls(REF_IMAGE_PREFIX):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                    "role": "reference_image",
                })
            for url in self._collect_ref_urls(REF_VIDEO_PREFIX):
                content.append({
                    "type": "video_url",
                    "video_url": {"url": url},
                    "role": "reference_video",
                })
            for url in self._collect_ref_urls(REF_AUDIO_PREFIX):
                content.append({
                    "type": "audio_url",
                    "audio_url": {"url": url},
                    "role": "reference_audio",
                })

        return content

    def _build_request_body(self) -> dict:
        """Build the API request body."""
        body: dict = {
            "model": self._resolved_model,
            "content": self._build_content(),
        }

        if self.resolution:
            body["resolution"] = self.resolution
        if self.ratio:
            body["ratio"] = self.ratio
        if self.duration:
            body["duration"] = int(self.duration)
        if self.generate_audio:
            body["generate_audio"] = self.generate_audio

        return body

    def _create_task(self, client: httpx.Client, base_url: str) -> str:
        """Submit a video generation task and return the task ID."""
        payload = self._build_request_body()
        url = f"{base_url}contents/generations/tasks"

        self.log(f"Submitting task to {url}, model: {self._resolved_model}, mode: {self.generation_mode}")

        resp = client.post(url, json=payload)
        if not resp.is_success:
            self.log(f"API error {resp.status_code}: {resp.text}", "ERROR")
            resp.raise_for_status()

        data = resp.json()
        task_id = data.get("id", "")
        if not task_id:
            msg = f"No task ID in response: {data}"
            raise TaskError(msg)

        return task_id

    def _poll_task(self, client: httpx.Client, base_url: str, task_id: str) -> dict:
        """Poll task status until completion or timeout."""
        url = f"{base_url}contents/generations/tasks/{task_id}"
        max_retries = self.max_wait_time // self.poll_interval
        consecutive_errors = 0
        max_consecutive_errors = 5

        for attempt in range(max_retries):
            try:
                self.log(f"Checking task status (attempt {attempt + 1})")
                resp = client.get(url)
                resp.raise_for_status()
                data = resp.json()
                consecutive_errors = 0

                status = data.get("status", "unknown")
                self.status = f"Task status: {status}"
                self.log(f"Task status: {status}")

                if status == "succeeded":
                    self.log("Task completed successfully!")
                    return data

                if status in ("failed", "error", "expired"):
                    error_msg = data.get("error", {})
                    error_str = f"Task failed with status: {status}, error: {error_msg}"
                    self.log(error_str, "ERROR")
                    raise TaskError(error_str)

                time.sleep(self.poll_interval)

            except TaskError:
                raise
            except httpx.HTTPStatusError as e:
                consecutive_errors += 1
                self.log(f"HTTP error checking task status: {e.response.status_code}", "WARNING")
                if consecutive_errors >= max_consecutive_errors:
                    msg = "Too many consecutive errors while polling task status"
                    raise TaskError(msg) from e
                time.sleep(self.poll_interval * 2)
            except (httpx.HTTPError, ValueError, KeyError) as e:
                consecutive_errors += 1
                self.log(f"Error checking task status: {e}", "WARNING")
                if consecutive_errors >= max_consecutive_errors:
                    msg = "Too many consecutive errors while polling task status"
                    raise TaskError(msg) from e
                time.sleep(self.poll_interval * 2)

        timeout_msg = f"Timeout after {self.max_wait_time} seconds"
        self.log(timeout_msg, "ERROR")
        raise TaskTimeoutError(timeout_msg)

    def _extract_video_url(self, data: dict) -> str:
        """Extract video URL from the task response."""
        content = data.get("content")
        if isinstance(content, dict):
            url = content.get("video_url", "")
            if url:
                return url

        if isinstance(content, list):
            for item in content:
                if item.get("type") == "video_url":
                    return item.get("video_url", {}).get("url", "")
                if item.get("type") == "url":
                    return item.get("url", "")

        return ""

    def generate_video(self) -> Message:
        """Generate a video using the selected model via OpenAI-compatible API."""
        api_key, base_url, model_name = self._resolve_credentials()
        self._resolved_model = model_name

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(headers=headers, timeout=30, trust_env=False) as client:
                self.status = "Submitting video generation task..."
                task_id = self._create_task(client, base_url)
                self._task_id = task_id
                self.log(f"Task submitted: {task_id}")

                result = self._poll_task(client, base_url, task_id)
                video_url = self._extract_video_url(result)

                self._task_info = {
                    "task_id": task_id,
                    "model": model_name,
                    "mode": self.generation_mode,
                    "status": result.get("status", ""),
                    "video_url": video_url,
                }

                self.status = f"Video generated: {video_url[:80]}..."
                return Message(text=video_url)

        except (TaskError, TaskTimeoutError):
            raise
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if hasattr(e.response, "text") else ""
            error_msg = f"HTTP {e.response.status_code}: {error_detail}"
            self.log(error_msg, "ERROR")
            self._task_id = None
            self._task_info = None
            msg = f"Video generation failed: {error_msg}"
            raise ValueError(msg) from e
        except (httpx.HTTPError, ValueError, KeyError) as e:
            self.log(f"Error: {e}", "ERROR")
            self._task_id = None
            self._task_info = None
            msg = f"Video generation failed: {e}"
            raise ValueError(msg) from e

    def get_task_info(self) -> Data:
        """Return task information as Data."""
        if self._task_info:
            return Data(data=self._task_info)
        return Data(data={"task_id": self._task_id or ""})
