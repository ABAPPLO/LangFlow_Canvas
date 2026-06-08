from __future__ import annotations

from typing import Any

import httpx

from lfx.custom import Component
from lfx.inputs import IntInput, MultilineInput
from lfx.io import MessageTextInput, Output
from lfx.schema.message import Message


class OriginalVideoURLComponent(Component):
    display_name = "Original Video URL"
    description = "Parse a shared video link and return the original video URL."
    icon = "Video"
    name = "OriginalVideoURL"

    inputs = [
        MultilineInput(
            name="link",
            display_name="Video Link",
            info="Paste the shared video text or URL to parse.",
            required=True,
            tool_mode=True,
        ),
        MessageTextInput(
            name="api_url",
            display_name="Parse API URL",
            value="http://43.139.175.114:12007/api/v1/parse_content",
            advanced=True,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            value=30,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Video URL", name="video_url", method="parse_video_url"),
    ]

    def _extract_video_url(self, payload: dict[str, Any]) -> str:
        response_data = payload.get("data")
        source = response_data if isinstance(response_data, dict) else payload

        top_level_url = source.get("url")
        if isinstance(top_level_url, str) and top_level_url.strip():
            return top_level_url.strip()

        videos = source.get("videos")
        if isinstance(videos, list):
            for video in videos:
                if not isinstance(video, dict):
                    continue

                video_url = video.get("url")
                if isinstance(video_url, str) and video_url.strip():
                    return video_url.strip()

                fullinfo = video.get("video_fullinfo")
                if isinstance(fullinfo, list):
                    for item in fullinfo:
                        if isinstance(item, dict):
                            fullinfo_url = item.get("url")
                            if isinstance(fullinfo_url, str) and fullinfo_url.strip():
                                return fullinfo_url.strip()

        return ""

    def parse_video_url(self) -> Message:
        link = (self.link or "").strip()
        if not link:
            self.status = "No video link provided."
            return Message(text="")

        api_url = (self.api_url or "").strip()
        if not api_url:
            msg = "Parse API URL is required."
            raise ValueError(msg)

        try:
            with httpx.Client(timeout=self.timeout, trust_env=False) as client:
                response = client.post(api_url, json={"link": link})
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError as exc:
            msg = f"Failed to parse video link: {exc}"
            raise ValueError(msg) from exc
        except ValueError as exc:
            msg = "Parse API returned an invalid JSON response."
            raise ValueError(msg) from exc

        if not isinstance(payload, dict):
            msg = "Parse API response must be a JSON object."
            raise TypeError(msg)

        video_url = self._extract_video_url(payload)
        if not video_url:
            msg = "No video URL found in parse API response."
            raise ValueError(msg)

        self.status = video_url
        return Message(text=video_url)
