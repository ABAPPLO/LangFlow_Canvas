"""Video Concatenator component - concatenate multiple videos in order."""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from pathlib import Path

import httpx

from lfx.custom import Component
from lfx.inputs import DropdownInput, IntInput
from lfx.schema import Data
from lfx.schema.message import Message
from lfx.template import Output

VIDEO_FIELD_PREFIX = "video_"


class VideoConcatenatorComponent(Component):
    display_name = "Video Concatenator"
    description = "Concatenate multiple video URLs into a single video in the order they are added."
    icon = "video"
    name = "VideoConcatenator"

    inputs = [
        IntInput(
            name="input_count",
            display_name="Video Count",
            info="Number of video inputs. Change to add or remove video entries.",
            value=2,
            real_time_refresh=True,
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

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "input_count":
            count = max(1, int(field_value)) if field_value else 2

            # Remove old dynamic video fields
            to_remove = [k for k in build_config if k.startswith(VIDEO_FIELD_PREFIX) and k[len(VIDEO_FIELD_PREFIX):].isdigit()]
            for k in to_remove:
                del build_config[k]

            # Create individual input fields with connection handles
            for i in range(1, count + 1):
                field_name_i = f"{VIDEO_FIELD_PREFIX}{i}"
                build_config[field_name_i] = {
                    "type": "str",
                    "input_types": ["Message", "Text"],
                    "name": field_name_i,
                    "display_name": f"Video {i}",
                    "value": "",
                    "show": True,
                    "advanced": False,
                    "multiline": False,
                    "placeholder": "Enter URL or connect component...",
                }

        return build_config

    def _parse_urls(self) -> list[str]:
        """Collect URLs from all dynamic input fields, preserving order."""
        urls: list[str] = []
        i = 1
        while True:
            val = getattr(self, f"{VIDEO_FIELD_PREFIX}{i}", None)
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
                urls.append(text)
            i += 1
        return urls

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
            with httpx.Client(timeout=300, follow_redirects=True, trust_env=False) as client:
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

    def _has_audio(self, video_path: str) -> bool:
        """Check if a video file has an audio stream."""
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        return bool(result.stdout.strip())

    def _normalize_video(self, input_path: Path, output_path: Path) -> bool:
        """Normalize a single video to consistent format (h264 + aac, yuv420p)."""
        has_audio = self._has_audio(str(input_path))

        # All inputs first, then output options, then output path
        cmd = ["ffmpeg", "-y", "-i", str(input_path)]
        if not has_audio:
            # Add silent audio source as second input
            cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"])

        # Output options
        cmd.extend([
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",
            "-r", "30",
            "-movflags", "+faststart",
        ])
        if not has_audio:
            cmd.append("-shortest")
        cmd.append(str(output_path))

        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        if result.returncode != 0:
            from lfx.log.logger import logger
            logger.error(f"ffmpeg normalize failed: {result.stderr}")
        return result.returncode == 0

    def _concat_normalized(self, filelist_path: str, output_path: str) -> bool:
        """Concatenate pre-normalized videos using stream copy (fast, safe)."""
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", filelist_path,
            "-c", "copy",
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
            normalized_dir = tmp_dir / "normalized"
            normalized_dir.mkdir()

            # Download / resolve all videos
            local_paths: list[Path] = []
            for url in urls:
                path = self._download(url, tmp_dir)
                if path is None:
                    self.status = f"Failed to resolve: {url}"
                    return Data(text="", data={"error": f"Failed to resolve: {url}"})
                local_paths.append(path)

            # Step 1: Normalize all videos to consistent format
            normalized_paths: list[Path] = []
            for idx, path in enumerate(local_paths):
                norm_path = normalized_dir / f"norm_{idx:03d}.mp4"
                if not self._normalize_video(path, norm_path):
                    self.status = f"Failed to normalize video: {path}"
                    return Data(text="", data={"error": f"Failed to normalize video: {path}"})
                normalized_paths.append(norm_path)

            # Step 2: Create concat file list from normalized videos
            filelist_path = normalized_dir / "filelist.txt"
            with filelist_path.open("w", encoding="utf-8") as f:
                for p in normalized_paths:
                    escaped = str(p).replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")

            # Output path — use absolute path so frontend can proxy it
            fmt = self.output_format or "mp4"
            output_dir = Path("uploads").resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"concat_{uuid.uuid4().hex[:8]}.{fmt}"

            # Step 3: Concatenate (copy mode is safe now — all videos are normalized)
            if not self._concat_normalized(str(filelist_path), str(output_path)):
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
