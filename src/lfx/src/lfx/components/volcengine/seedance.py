import time

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

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"

MODE_TEXT = "Text to Video"
MODE_FIRST_FRAME = "First Frame"
MODE_FIRST_LAST = "First & Last Frame"
MODE_MULTIMODAL = "Multimodal"

MODE_OPTIONS = [MODE_TEXT, MODE_FIRST_FRAME, MODE_FIRST_LAST, MODE_MULTIMODAL]

REF_IMAGE_PREFIX = "ref_image_"
REF_VIDEO_PREFIX = "ref_video_"
REF_AUDIO_PREFIX = "ref_audio_"


class TaskError(Exception):
    """Error raised when a task fails."""


class TaskTimeoutError(Exception):
    """Error raised when a task times out."""


class SeedanceVideoComponent(Component):
    display_name = "Seedance Video"
    description = "Generate videos using Volcengine Seedance 2.0 API."
    icon = "Video"
    name = "SeedanceVideo"

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
            info="Model ID, e.g. doubao-seedance-2-0-260128.",
            value="doubao-seedance-2-0-260128",
        ),
        DropdownInput(
            name="mode",
            display_name="Generation Mode",
            info="Select video generation mode.",
            options=MODE_OPTIONS,
            value=MODE_TEXT,
            real_time_refresh=True,
        ),
        MultilineInput(
            name="prompt",
            display_name="Prompt",
            info="Text prompt for video generation.",
            required=True,
        ),
        # --- Mode-specific: single URL fields ---
        MessageTextInput(
            name="first_frame_url",
            display_name="First Frame Image URL",
            info="URL of the first frame image.",
            dynamic=True,
            show=False,
        ),
        MessageTextInput(
            name="last_frame_url",
            display_name="Last Frame Image URL",
            info="URL of the last frame image.",
            dynamic=True,
            show=False,
        ),
        BoolInput(
            name="web_search",
            display_name="Web Search",
            info="Enable web search (Text to Video mode only).",
            value=False,
            dynamic=True,
            show=False,
        ),
        # --- Multimodal: multiline text input + dynamic handle inputs ---
        MultilineInput(
            name="ref_image_urls",
            display_name="Reference Image URLs",
            info="One URL per line (max 9 images). Or use the count below to add input handles.",
            dynamic=True,
            show=False,
        ),
        IntInput(
            name="ref_image_count",
            display_name="+ Image Inputs",
            info="Increase to add connection handles for images (max 9).",
            value=0,
            real_time_refresh=True,
        ),
        MultilineInput(
            name="ref_video_urls",
            display_name="Reference Video URLs",
            info="One URL per line (max 3 videos). Or use the count below to add input handles.",
            dynamic=True,
            show=False,
        ),
        IntInput(
            name="ref_video_count",
            display_name="+ Video Inputs",
            info="Increase to add connection handles for videos (max 3).",
            value=0,
            real_time_refresh=True,
        ),
        MultilineInput(
            name="ref_audio_urls",
            display_name="Reference Audio URLs",
            info="One URL per line (max 3 audios). Or use the count below to add input handles.",
            dynamic=True,
            show=False,
        ),
        IntInput(
            name="ref_audio_count",
            display_name="+ Audio Inputs",
            info="Increase to add connection handles for audios (max 3).",
            value=0,
            real_time_refresh=True,
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
        BoolInput(
            name="watermark",
            display_name="Watermark",
            info="Add watermark to the video.",
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
            name="result_video_url",
            method="generate_video",
            type_=Message,
        ),
        Output(
            display_name="Task Info",
            name="task_info",
            method="get_task_info",
            type_=Data,
        ),
    ]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._task_id: str | None = None
        self._task_info: dict | None = None

    def update_build_config(self, build_config, field_value, field_name=None):
        # Handle mode switching
        if field_name == "mode":
            mode = field_value

            # Show/hide mode-specific fields
            build_config["first_frame_url"]["show"] = mode in (MODE_FIRST_FRAME, MODE_FIRST_LAST)
            build_config["last_frame_url"]["show"] = mode == MODE_FIRST_LAST
            build_config["web_search"]["show"] = mode == MODE_TEXT

            is_multimodal = mode == MODE_MULTIMODAL
            for key in ("ref_image_urls", "ref_video_urls", "ref_audio_urls"):
                if key in build_config:
                    build_config[key]["show"] = is_multimodal

            # Clear all dynamic ref fields
            for prefix in (REF_IMAGE_PREFIX, REF_VIDEO_PREFIX, REF_AUDIO_PREFIX):
                to_remove = [k for k in build_config if k.startswith(prefix) and k[len(prefix):].isdigit()]
                for k in to_remove:
                    del build_config[k]

            # Re-create dynamic fields for multimodal mode
            if is_multimodal:
                self._create_dynamic_ref_fields(build_config)

        # Handle ref count changes — create/remove dynamic input handles
        if field_name in ("ref_image_count", "ref_video_count", "ref_audio_count"):
            # Only create dynamic fields in Multimodal mode
            mode = build_config.get("mode", {}).get("value", MODE_TEXT)
            if mode != MODE_MULTIMODAL:
                return build_config

            count = max(0, int(field_value)) if field_value else 0

            if field_name == "ref_image_count":
                prefix, label, cap = REF_IMAGE_PREFIX, "Image", 9
            elif field_name == "ref_video_count":
                prefix, label, cap = REF_VIDEO_PREFIX, "Video", 3
            else:
                prefix, label, cap = REF_AUDIO_PREFIX, "Audio", 3

            count = min(count, cap)

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
                max(0, int(build_config.get(count_key, {}).get("value", 0))),
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

    @staticmethod
    def _parse_multiline(text: str) -> list[str]:
        """Parse a multiline text into a list of non-empty URLs."""
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

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

    def _get_all_ref_urls(self, multiline_attr: str, prefix: str) -> list[str]:
        """Combine URLs from multiline text input + dynamic handle inputs."""
        urls: list[str] = []

        # From multiline text
        raw = getattr(self, multiline_attr, "")
        if raw:
            urls.extend(self._parse_multiline(raw))

        # From dynamic handle fields
        urls.extend(self._collect_ref_urls(prefix))

        return urls

    def _build_content(self) -> list[dict]:
        """Build the content list based on the selected mode."""
        content: list[dict] = []
        mode = getattr(self, "mode", MODE_TEXT)

        # Text prompt is included in all modes
        if self.prompt:
            content.append({"type": "text", "text": self.prompt})

        if mode == MODE_FIRST_FRAME:
            if self.first_frame_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self.first_frame_url},
                    "role": "first_frame",
                })

        elif mode == MODE_FIRST_LAST:
            if self.first_frame_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self.first_frame_url},
                    "role": "first_frame",
                })
            if self.last_frame_url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self.last_frame_url},
                    "role": "last_frame",
                })

        elif mode == MODE_MULTIMODAL:
            for url in self._get_all_ref_urls("ref_image_urls", REF_IMAGE_PREFIX):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                    "role": "reference_image",
                })
            for url in self._get_all_ref_urls("ref_video_urls", REF_VIDEO_PREFIX):
                content.append({
                    "type": "video_url",
                    "video_url": {"url": url},
                    "role": "reference_video",
                })
            for url in self._get_all_ref_urls("ref_audio_urls", REF_AUDIO_PREFIX):
                content.append({
                    "type": "audio_url",
                    "audio_url": {"url": url},
                    "role": "reference_audio",
                })

        return content

    def _build_request_body(self) -> dict:
        """Build the full API request body."""
        body: dict = {
            "model": self.model,
            "content": self._build_content(),
            "resolution": self.resolution,
            "ratio": self.ratio,
            "duration": int(self.duration),
            "generate_audio": self.generate_audio,
            "watermark": self.watermark,
        }

        # Web search tool only for text-to-video mode
        if getattr(self, "mode", MODE_TEXT) == MODE_TEXT and getattr(self, "web_search", False):
            body["tools"] = [{"type": "web_search"}]

        return body

    def _create_task(self, client: httpx.Client) -> str:
        """Submit a video generation task and return the task ID."""
        payload = self._build_request_body()

        resp = client.post(self.base_url, json=payload)
        if resp.status_code >= 400:
            self.log(f"API error {resp.status_code}: {resp.text}", "ERROR")
            resp.raise_for_status()
        data = resp.json()

        task_id = data.get("id", "")
        if not task_id:
            msg = f"No task ID in response: {data}"
            raise TaskError(msg)

        return task_id

    def _poll_task(self, client: httpx.Client, task_id: str) -> dict:
        """Poll task status until completion or timeout."""
        url = f"{self.base_url}/{task_id}"
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
        """Generate a video using Volcengine Seedance 2.0 API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(headers=headers, timeout=30, trust_env=False) as client:
                self.status = "Submitting video generation task..."
                self.log(f"Submitting task to {self.base_url}, model: {self.model}, mode: {self.mode}")

                task_id = self._create_task(client)
                self._task_id = task_id
                self.log(f"Task submitted: {task_id}")

                result = self._poll_task(client, task_id)
                video_url = self._extract_video_url(result)

                self._task_info = {
                    "task_id": task_id,
                    "model": self.model,
                    "mode": self.mode,
                    "status": result.get("status", ""),
                    "video_url": video_url,
                }

                self.status = f"Video generated: {video_url[:50]}..."
                return Message(text=video_url)

        except (TaskError, TaskTimeoutError):
            raise
        except httpx.HTTPStatusError as e:
            error_detail = ""
            try:
                error_detail = e.response.text
            except Exception:
                pass
            error_msg = f"HTTP {e.response.status_code}: {error_detail}"
            self.log(error_msg, "ERROR")
            self._task_id = None
            self._task_info = None
            return Message(text=f"Error: {error_msg}")
        except (httpx.HTTPError, ValueError, KeyError) as e:
            self.log(f"Error: {e}", "ERROR")
            self._task_id = None
            self._task_info = None
            return Message(text=f"Error: {e}")

    def get_task_info(self) -> Data:
        """Return task information as Data."""
        if self._task_info:
            return Data(data=self._task_info)
        return Data(data={"task_id": self._task_id or ""})
