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

BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/contents/generations"

MODE_TEXT = "Text to Video"
MODE_FIRST_FRAME = "First Frame"
MODE_FIRST_LAST = "First & Last Frame"
MODE_MULTIMODAL = "Multimodal"

MODE_OPTIONS = [MODE_TEXT, MODE_FIRST_FRAME, MODE_FIRST_LAST, MODE_MULTIMODAL]

# Mode-specific fields that are hidden by default
MODE_SPECIFIC_FIELDS = [
    "first_frame_url",
    "last_frame_url",
    "ref_image_urls",
    "ref_video_urls",
    "ref_audio_urls",
    "web_search",
]

# Which fields to show for each mode
MODE_FIELD_MAP: dict[str, list[str]] = {
    MODE_TEXT: ["web_search"],
    MODE_FIRST_FRAME: ["first_frame_url"],
    MODE_FIRST_LAST: ["first_frame_url", "last_frame_url"],
    MODE_MULTIMODAL: ["ref_image_urls", "ref_video_urls", "ref_audio_urls"],
}


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
        DropdownInput(
            name="model",
            display_name="Model",
            info="Seedance model version.",
            options=[
                "doubao-seedance-2-0-pro-260128",
                "doubao-seedance-2-0-260128",
            ],
            value="doubao-seedance-2-0-pro-260128",
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
        # --- Mode-specific fields (hidden by default) ---
        BoolInput(
            name="web_search",
            display_name="Web Search",
            info="Enable web search (Text to Video mode only).",
            value=False,
            dynamic=True,
            show=False,
        ),
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
        MultilineInput(
            name="ref_image_urls",
            display_name="Reference Image URLs",
            info="One URL per line (max 9 images).",
            dynamic=True,
            show=False,
        ),
        MultilineInput(
            name="ref_video_urls",
            display_name="Reference Video URLs",
            info="One URL per line (max 3 videos).",
            dynamic=True,
            show=False,
        ),
        MultilineInput(
            name="ref_audio_urls",
            display_name="Reference Audio URLs",
            info="One URL per line (max 3 audios).",
            dynamic=True,
            show=False,
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
            value=600,
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
        if field_name == "mode":
            for f in MODE_SPECIFIC_FIELDS:
                if f in build_config:
                    build_config[f]["show"] = False
            for f in MODE_FIELD_MAP.get(field_value, []):
                if f in build_config:
                    build_config[f]["show"] = True
        return build_config

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
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": url},
                    "role": "reference_image",
                }
                for url in self._parse_urls(self.ref_image_urls)
            )
            content.extend(
                {
                    "type": "video_url",
                    "video_url": {"url": url},
                    "role": "reference_video",
                }
                for url in self._parse_urls(self.ref_video_urls)
            )
            content.extend(
                {
                    "type": "audio_url",
                    "audio_url": {"url": url},
                    "role": "reference_audio",
                }
                for url in self._parse_urls(self.ref_audio_urls)
            )

        return content

    @staticmethod
    def _parse_urls(text: str) -> list[str]:
        """Parse a multiline text into a list of non-empty URLs."""
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

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

        resp = client.post(BASE_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

        task_id = data.get("id", "")
        if not task_id:
            msg = f"No task ID in response: {data}"
            raise TaskError(msg)

        return task_id

    def _poll_task(self, client: httpx.Client, task_id: str) -> dict:
        """Poll task status until completion or timeout."""
        url = f"{BASE_URL}/{task_id}"
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
        for item in data.get("content", []):
            if item.get("type") == "video_url":
                return item.get("video_url", {}).get("url", "")
            if item.get("type") == "url":
                return item.get("url", "")

        for item in data.get("data", []):
            if "url" in item:
                return item["url"]

        return ""

    def generate_video(self) -> Message:
        """Generate a video using Volcengine Seedance 2.0 API."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            with httpx.Client(headers=headers, timeout=30) as client:
                self.status = "Submitting video generation task..."
                self.log(f"Submitting task with model: {self.model}, mode: {self.mode}")

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
