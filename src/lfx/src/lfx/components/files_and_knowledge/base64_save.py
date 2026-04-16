"""Base64 Save component - decode base64 data and save to configured storage backend."""

from __future__ import annotations

import base64
import binascii
import re
import uuid

from lfx.custom import Component
from lfx.io import DropdownInput, HandleInput, MessageTextInput, Output
from lfx.log.logger import logger
from lfx.schema.image import IMAGE_ENDPOINT, Image
from lfx.schema.message import Message
from lfx.services.deps import get_storage_service

# Regex to match data URI: data:image/png;base64,xxxxx
_DATA_URI_RE = re.compile(r"^data:([\w/+.-]+);base64,(.+)$", re.DOTALL)

# Map MIME types to file extensions
_MIME_TO_EXT: dict[str, str] = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/svg+xml": ".svg",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/ogg": ".ogg",
    "audio/aac": ".aac",
    "audio/flac": ".flac",
    "application/pdf": ".pdf",
    "application/octet-stream": ".bin",
}

FORMAT_OPTIONS = list({ext.lstrip("."): ext for ext in _MIME_TO_EXT.values()}.keys())

_MIN_BASE64_LENGTH = 20
_MIN_EXTRACT_BASE64_LENGTH = 80


def _parse_data_uri(value: str) -> tuple[str, str]:
    """Parse a data URI into (mime_type, raw_base64).

    Returns ("", raw) if not a data URI (assumed to be raw base64).
    """
    match = _DATA_URI_RE.match(value.strip())
    if match:
        return match.group(1), match.group(2)
    return "", value.strip()


class Base64SaveComponent(Component):
    display_name = "Save Base64"
    description = "Decode base64 data and save to the configured storage backend (local, S3, COS, OSS)."
    icon = "save"
    name = "Base64Save"
    metadata = {
        "keywords": [
            "base64",
            "save",
            "image",
            "file",
            "storage",
            "download",
            "decode",
        ],
    }

    inputs = [
        HandleInput(
            name="data_input",
            display_name="Base64 Data",
            info="Base64 string or Message containing base64 data. "
            "Supports data URI (data:image/png;base64,...) or raw base64.",
            input_types=["Message", "Data", "Text"],
            required=True,
        ),
        DropdownInput(
            name="format",
            display_name="Format",
            info="File format. Ignored when input uses data URI (auto-detected from MIME type).",
            options=FORMAT_OPTIONS,
            value="png",
        ),
        MessageTextInput(
            name="file_name_prefix",
            display_name="File Name Prefix",
            info="Prefix for generated file names. Defaults to 'b64_'.",
            value="b64_",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Message", name="message", method="save_base64"),
    ]

    def _extract_base64_strings(self) -> list[str]:
        """Extract base64 strings from the input data."""
        data = self.data_input

        if isinstance(data, Message):
            texts = []
            if data.text:
                texts.append(str(data.text))
            return self._extract_from_texts(texts)

        if isinstance(data, list):
            return self._extract_from_texts([str(item) for item in data if item])

        if isinstance(data, str):
            return self._extract_from_texts([data])

        if hasattr(data, "data") and isinstance(data.data, dict):
            for key in ("base64", "image", "data", "content", "url"):
                val = data.data.get(key)
                if isinstance(val, str) and val:
                    return self._extract_from_texts([val])
            return self._extract_from_texts([str(data.data)])

        return self._extract_from_texts([str(data)])

    def _extract_from_texts(self, texts: list[str]) -> list[str]:
        """Extract data URI and standalone base64 chunks from mixed text."""
        results: list[str] = []
        for text in texts:
            # 1. Extract data URIs: data:image/png;base64,xxxxx
            data_uris = [m.group(0) for m in _DATA_URI_RE.finditer(text)]
            results.extend(data_uris)

            # 2. If no data URIs found, try the whole text as raw base64
            if not data_uris and text.strip():
                stripped = re.sub(r"\s+", "", text)
                if re.fullmatch(r"[A-Za-z0-9+/]+=*", stripped) and len(stripped) > _MIN_BASE64_LENGTH:
                    results.append(text.strip())
                else:
                    # Extract long base64-like substrings from mixed text
                    pattern = rf"[A-Za-z0-9+/]{{{_MIN_EXTRACT_BASE64_LENGTH},}}={{0,2}}"
                    results.extend(m.group(0) for m in re.finditer(pattern, text))

        return results

    def _detect_extension(self, b64_str: str) -> str:
        """Detect file extension from data URI or user config."""
        mime_type, _ = _parse_data_uri(b64_str)
        if mime_type and mime_type in _MIME_TO_EXT:
            return _MIME_TO_EXT[mime_type]

        # Fallback to user-selected format
        fmt = getattr(self, "format", "png")
        return f".{fmt}" if not fmt.startswith(".") else fmt

    async def save_base64(self) -> Message:
        """Decode base64 data and save files to storage."""
        storage_service = get_storage_service()
        if not storage_service:
            msg = "Storage service is not available."
            raise ValueError(msg)

        if not self.user_id:
            msg = "User ID is required for file saving."
            raise ValueError(msg)

        flow_id = str(self.user_id)
        prefix = getattr(self, "file_name_prefix", "b64_")

        b64_strings = self._extract_base64_strings()
        if not b64_strings:
            self.status = "No base64 data found in input."
            return Message(text="No base64 data found in input.")

        saved_paths: list[str] = []
        files: list[str | Image] = []
        urls: list[str] = []

        for b64_str in b64_strings:
            # Handle multi-line or whitespace in base64
            _mime_type, raw_b64 = _parse_data_uri(b64_str)

            # Clean whitespace from base64 data
            raw_b64 = re.sub(r"\s+", "", raw_b64)

            try:
                file_bytes = base64.b64decode(raw_b64)
            except (ValueError, binascii.Error) as e:
                logger.warning(f"Failed to decode base64 data: {e}")
                continue

            ext = self._detect_extension(b64_str)
            file_name = f"{prefix}{uuid.uuid4().hex[:12]}{ext}"

            await storage_service.save_file(
                flow_id=flow_id,
                file_name=file_name,
                data=file_bytes,
            )

            path = f"{flow_id}/{file_name}"
            url = f"{IMAGE_ENDPOINT}{path}"
            saved_paths.append(path)
            files.append(Image(path=path))
            urls.append(url)
            logger.debug(f"Saved base64 file: {path} ({len(file_bytes)} bytes)")

        if not saved_paths:
            self.status = "No valid base64 data could be decoded."
            return Message(text="No valid base64 data could be decoded.")

        # Build result text with accessible URLs
        result_text = urls[0] if len(urls) == 1 else "\n".join(urls)

        self.status = f"Saved {len(saved_paths)} file(s) to storage."

        return Message(text=result_text, files=files)
