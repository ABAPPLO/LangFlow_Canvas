from __future__ import annotations

import mimetypes
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import quote, unquote, urlparse
from uuid import uuid4

import httpx

from lfx.custom import Component
from lfx.inputs import IntInput
from lfx.io import DropdownInput, MessageTextInput, Output, SecretStrInput, StrInput
from lfx.schema.message import Message

DOWNLOAD_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"
    ),
    "Accept": "video/mp4,video/*;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


class RemoteVideoToCOSComponent(Component):
    display_name = "Remote Video to COS"
    description = "Download a remote video URL and rehost it on Tencent Cloud COS."
    icon = "CloudUpload"
    name = "RemoteVideoToCOS"

    inputs = [
        MessageTextInput(
            name="video_url",
            display_name="Video URL",
            info="Remote video URL to download and upload to COS.",
            input_types=["Message", "Text"],
            required=True,
        ),
        SecretStrInput(
            name="secret_id",
            display_name="SecretId",
            info="Tencent Cloud API SecretId.",
            required=True,
        ),
        SecretStrInput(
            name="secret_key",
            display_name="SecretKey",
            info="Tencent Cloud API SecretKey.",
            required=True,
        ),
        StrInput(
            name="region",
            display_name="Region",
            info="COS bucket region, for example ap-guangzhou.",
            value="ap-guangzhou",
        ),
        StrInput(
            name="bucket_name",
            display_name="Bucket",
            info="COS bucket name, for example my-bucket-1250000000.",
            required=True,
        ),
        StrInput(
            name="cos_prefix",
            display_name="Path Prefix",
            info="COS object key prefix.",
            value="langflow/videos",
            advanced=True,
        ),
        DropdownInput(
            name="acl",
            display_name="ACL",
            options=["private", "public-read"],
            value="public-read",
            info="Use public-read when the downstream model service must fetch the URL directly.",
            advanced=True,
        ),
        IntInput(
            name="timeout",
            display_name="Timeout",
            value=300,
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="COS Video URL", name="cos_video_url", method="upload_remote_video"),
    ]

    def _cos_client(self) -> Any:
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except ImportError as e:
            msg = "cos-python-sdk-v5 is not installed. Please install it using: uv pip install cos-python-sdk-v5"
            raise ImportError(msg) from e

        config = CosConfig(
            Region=self.region.strip(),
            SecretId=self.secret_id.strip(),
            SecretKey=self.secret_key.strip(),
        )
        return CosS3Client(config)

    def _build_key(self, source_url: str, content_type: str | None) -> str:
        prefix = (self.cos_prefix or "").strip().strip("/")
        parsed = urlparse(source_url)
        path_suffix = PurePosixPath(unquote(parsed.path)).suffix
        guessed_suffix = mimetypes.guess_extension((content_type or "").split(";", 1)[0].strip())
        suffix = path_suffix if path_suffix else guessed_suffix or ".mp4"

        filename = f"{uuid4().hex}{suffix}"
        return f"{prefix}/{filename}" if prefix else filename

    def _build_url(self, key: str) -> str:
        quoted_key = quote(key, safe="/")
        return f"https://{self.bucket_name.strip()}.cos.{self.region.strip()}.myqcloud.com/{quoted_key}"

    def _download_video(self, url: str) -> tuple[bytes, str | None]:
        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            headers=DOWNLOAD_HEADERS,
            trust_env=False,
        ) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content, response.headers.get("content-type")

    def upload_remote_video(self) -> Message:
        video_url = self.video_url
        if isinstance(video_url, Message):
            video_url = video_url.get_text()
        video_url = str(video_url or "").strip()

        if not video_url:
            self.status = "No video URL provided."
            return Message(text="")

        try:
            body, content_type = self._download_video(video_url)
        except httpx.HTTPError as exc:
            msg = f"Failed to download remote video: {exc}"
            raise ValueError(msg) from exc

        client = self._cos_client()
        key = self._build_key(video_url, content_type)
        upload_kwargs = {
            "Bucket": self.bucket_name.strip(),
            "Key": key,
            "Body": body,
            "ACL": self.acl,
        }
        if content_type:
            upload_kwargs["ContentType"] = content_type
        client.put_object(**upload_kwargs)

        cos_url = self._build_url(key)
        self.status = cos_url
        return Message(text=cos_url)
