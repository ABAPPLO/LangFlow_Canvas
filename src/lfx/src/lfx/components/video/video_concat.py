"""Video Concatenator component - concatenate multiple videos in order."""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

import httpx

from lfx.custom import Component
from lfx.inputs import DropdownInput, MessageTextInput
from lfx.schema import Data
from lfx.template import Output


class VideoConcatenatorComponent(Component):
    display_name = "Video Concatenator"
    description = "Concatenate multiple video URLs into a single video in the order they are added."
    icon = "video"
    name = "VideoConcatenator"

    inputs = [
        MessageTextInput(
            name="video_urls",
            display_name="Video URLs",
            info="Click '+' to add video URLs or local file paths. Videos are concatenated in order.",
            is_list=True,
            list_add_label="Add Video",
            placeholder="Enter video URL or local path...",
            input_types=["Message", "Text"],
        ),
        DropdownInput(
            name="output_format",
            display_name="Output Format",
            info="Output video container format.",
            options=["mp4", "mov", "avi", "mkv"],
            value="mp4",
        ),
    ]

    outputs = [
        Output(display_name="Video", name="video", method="concat_videos", types=["Data"]),
    ]

    def _parse_urls(self) -> list[str]:
        """Parse the video_urls list input into a list of URL strings."""
        raw = getattr(self, "video_urls", None)
        if not raw:
            return []
        if isinstance(raw, list):
            urls = []
            for u in raw:
                if hasattr(u, "get_text"):
                    urls.append(u.get_text().strip())
                else:
                    s = str(u).strip()
                    if s:
                        urls.append(s)
            return urls
        if isinstance(raw, str) and raw.strip():
            return [raw.strip()]
        if hasattr(raw, "get_text"):
            text = raw.get_text().strip()
            return [text] if text else []
        return []

    def _download(self, url: str, tmp_dir: Path) -> Path | None:
        """Download a remote URL to tmp_dir. Returns local path or None on failure."""
        if not url.startswith(("http://", "https://")):
            # Local path — validate existence
            p = Path(url).expanduser().resolve()
            if not p.exists():
                self.status = f"File not found: {url}"
                return None
            return p

        # Remote URL — download
        ext = Path(url.split("?")[0]).suffix or ".mp4"
        local_path = tmp_dir / f"input_{uuid.uuid4().hex[:8]}{ext}"
        try:
            with httpx.Client(timeout=300, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                local_path.write_bytes(resp.content)
        except httpx.HTTPError as e:
            self.status = f"Download failed: {url} — {e}"
            return None
        return local_path

    def _get_duration(self, video_path: str) -> float:
        """Get video duration using ffprobe."""
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        if result.returncode != 0:
            return 0.0
        try:
            return float(result.stdout.strip())
        except ValueError:
            return 0.0

    def _concat_copy(self, filelist_path: str, output_path: str) -> bool:
        """Try fast concatenation using stream copy (no re-encode)."""
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", filelist_path,
            "-c", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        return result.returncode == 0

    def _concat_reencode(self, filelist_path: str, output_path: str) -> bool:
        """Fallback: re-encode all videos to a common format."""
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", filelist_path,
            "-c:v", "libx264",
            "-c:a", "aac",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        return result.returncode == 0

    def concat_videos(self) -> Data:
        """Download/resolve videos and concatenate them with ffmpeg."""
        urls = self._parse_urls()
        if not urls:
            self.status = "No video URLs provided"
            return Data(text="", data={"error": "No video URLs provided"})

        if len(urls) == 1:
            # Single video — just return it
            url = urls[0]
            if not url.startswith(("http://", "https://")):
                p = Path(url).expanduser().resolve()
                if p.exists():
                    duration = self._get_duration(str(p))
                    self.status = f"Single video: {p}"
                    return Data(text=str(p), data={"path": str(p), "duration": duration, "count": 1})
            self.status = f"Single video: {url}"
            return Data(text=url, data={"path": url, "count": 1})

        with tempfile.TemporaryDirectory(prefix="video_concat_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            # Download / resolve all videos
            local_paths: list[Path] = []
            for url in urls:
                path = self._download(url, tmp_dir)
                if path is None:
                    self.status = f"Failed to resolve: {url}"
                    return Data(text="", data={"error": f"Failed to resolve: {url}"})
                local_paths.append(path)

            # Create ffmpeg concat file list
            filelist_path = tmp_dir / "filelist.txt"
            with filelist_path.open("w", encoding="utf-8") as f:
                for p in local_paths:
                    # Escape single quotes in path for ffmpeg concat format
                    escaped = str(p).replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")

            # Output path
            fmt = self.output_format or "mp4"
            output_dir = Path("uploads")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"concat_{uuid.uuid4().hex[:8]}.{fmt}"

            # Try fast copy first, fall back to re-encode
            if not self._concat_copy(str(filelist_path), str(output_path)):
                if not self._concat_reencode(str(filelist_path), str(output_path)):
                    self.status = "FFmpeg concatenation failed"
                    return Data(text="", data={"error": "FFmpeg concatenation failed"})

            duration = self._get_duration(str(output_path))
            self.status = f"Concatenated {len(local_paths)} videos → {output_path} ({duration:.1f}s)"

            return Data(
                text=str(output_path),
                data={
                    "path": str(output_path),
                    "duration": duration,
                    "count": len(local_paths),
                },
            )
