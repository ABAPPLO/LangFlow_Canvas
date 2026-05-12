"""Video Concatenator component - concatenate multiple videos in order."""

from __future__ import annotations

import json
import subprocess
import tempfile
import threading
import uuid
from pathlib import Path

import httpx

from lfx.custom import Component
from lfx.inputs import DropdownInput, IntInput
from langflow.io import HandleInput
from lfx.schema import Data
from lfx.schema.message import Message
from lfx.template import Output

VIDEO_FIELD_PREFIX = "video_"

MODE_UPSTREAM = "连接上游"
MODE_MANUAL = "手动输入"

# ---------- 资源保护常量 ----------
MAX_DOWNLOAD_SIZE = 200 * 1024 * 1024      # 单文件下载上限 200 MB
MAX_TOTAL_DURATION = 300.0                  # 所有视频总时长上限 300 秒
MAX_RESOLUTION = 1920                       # 最长边像素上限
FFMPEG_THREADS = 2                          # ffmpeg 编码线程数
LOCK_WAIT_TIMEOUT = 600                     # 等待并发锁超时（秒）

# 全局并发锁
_concat_lock = threading.Lock()
_concat_lock_holder = ""


class VideoConcatenatorComponent(Component):
    display_name = "Video Concatenator"
    description = "Concatenate multiple video URLs into a single video in the order they are added."
    icon = "video"
    name = "VideoConcatenator"

    inputs = [
        DropdownInput(
            name="mode",
            display_name="输入模式",
            options=[MODE_UPSTREAM, MODE_MANUAL],
            value=MODE_UPSTREAM,
            info="连接上游：接收上游组件的视频列表；手动输入：逐个添加视频URL。",
            real_time_refresh=True,
        ),
        HandleInput(
            name="video_list",
            display_name="视频列表",
            info="接收上游组件传入的视频URL列表（JSON数组字符串），如 Seedance 连续视频的输出。",
            input_types=["Message"],
            required=False,
        ),
        IntInput(
            name="input_count",
            display_name="视频数量",
            info="手动模式下视频输入数量。",
            value=2,
            real_time_refresh=True,
            show=False,
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
        if field_name == "mode":
            is_upstream = field_value == MODE_UPSTREAM
            build_config["video_list"]["show"] = is_upstream
            build_config["input_count"]["show"] = not is_upstream

            if is_upstream:
                to_remove = [k for k in build_config if k.startswith(VIDEO_FIELD_PREFIX) and k[len(VIDEO_FIELD_PREFIX):].isdigit()]
                for k in to_remove:
                    del build_config[k]

            if not is_upstream:
                count = max(1, int(build_config.get("input_count", {}).get("value", 2)))
                for i in range(1, count + 1):
                    f_name = f"{VIDEO_FIELD_PREFIX}{i}"
                    if f_name not in build_config:
                        build_config[f_name] = {
                            "type": "str",
                            "input_types": ["Message", "Text"],
                            "name": f_name,
                            "display_name": f"Video {i}",
                            "value": "",
                            "show": True,
                            "advanced": False,
                            "multiline": False,
                            "placeholder": "Enter URL or connect component...",
                        }

        if field_name == "input_count":
            mode = build_config.get("mode", {}).get("value", MODE_UPSTREAM)
            if mode != MODE_MANUAL:
                return build_config

            count = max(1, int(field_value)) if field_value else 2

            to_remove = [k for k in build_config if k.startswith(VIDEO_FIELD_PREFIX) and k[len(VIDEO_FIELD_PREFIX):].isdigit()]
            for k in to_remove:
                del build_config[k]

            for i in range(1, count + 1):
                f_name = f"{VIDEO_FIELD_PREFIX}{i}"
                build_config[f_name] = {
                    "type": "str",
                    "input_types": ["Message", "Text"],
                    "name": f_name,
                    "display_name": f"Video {i}",
                    "value": "",
                    "show": True,
                    "advanced": False,
                    "multiline": False,
                    "placeholder": "Enter URL or connect component...",
                }

        return build_config

    def _parse_urls(self) -> list[str]:
        video_list = getattr(self, "video_list", None)
        if video_list:
            if isinstance(video_list, Message):
                text = video_list.get_text().strip()
                if text.startswith("["):
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            return [str(u).strip() for u in parsed if str(u).strip()]
                    except (json.JSONDecodeError, TypeError):
                        pass

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

    # ---------- ffprobe 工具 ----------

    _ffprobe_available: bool | None = None

    def _check_ffprobe(self) -> bool:
        if self._ffprobe_available is None:
            try:
                subprocess.run(["ffprobe", "-version"], capture_output=True, check=True)
                self._ffprobe_available = True
            except (FileNotFoundError, subprocess.CalledProcessError):
                self._ffprobe_available = False
        return self._ffprobe_available

    def _get_duration(self, video_path: str) -> float:
        if not self._check_ffprobe():
            return 0.0
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

    def _get_resolution(self, video_path: str) -> tuple[int, int]:
        if not self._check_ffprobe():
            return 0, 0
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        if result.returncode != 0 or not result.stdout.strip():
            return 0, 0
        try:
            parts = result.stdout.strip().split(",")
            return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            return 0, 0

    def _get_codec_info(self, video_path: str) -> dict:
        if not self._check_ffprobe():
            return {}
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_name,codec_type,width,height,pix_fmt",
            "-of", "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        if result.returncode != 0:
            return {}
        try:
            return json.loads(result.stdout)
        except (json.JSONDecodeError, ValueError):
            return {}

    def _has_audio(self, video_path: str) -> bool:
        if not self._check_ffprobe():
            return False
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        return bool(result.stdout.strip())

    def _get_file_size(self, path) -> int:
        try:
            return Path(path).stat().st_size
        except OSError:
            return 0

    # ---------- 下载 ----------

    def _download(self, url: str, tmp_dir: Path) -> Path | None:
        if not url.startswith(("http://", "https://")):
            p = Path(url).expanduser().resolve()
            if not p.exists():
                self.status = f"File not found: {url}"
                return None
            return p

        ext = Path(url.split("?")[0]).suffix or ".mp4"
        local_path = tmp_dir / f"input_{uuid.uuid4().hex[:8]}{ext}"
        try:
            with httpx.Client(timeout=300, follow_redirects=True, trust_env=False) as client:
                # 先 HEAD 检查大小
                try:
                    head = client.head(url)
                    content_length = int(head.headers.get("content-length", 0))
                    if content_length > MAX_DOWNLOAD_SIZE:
                        self.status = f"文件过大: {content_length / 1024 / 1024:.0f}MB > {MAX_DOWNLOAD_SIZE / 1024 / 1024:.0f}MB"
                        return None
                except httpx.HTTPError:
                    pass

                # 流式下载，避免大文件一次性加载到内存
                with client.stream("GET", url) as resp:
                    resp.raise_for_status()
                    total = 0
                    with local_path.open("wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=1024 * 1024):
                            total += len(chunk)
                            if total > MAX_DOWNLOAD_SIZE:
                                self.status = f"文件过大: 已下载 {total / 1024 / 1024:.0f}MB > {MAX_DOWNLOAD_SIZE / 1024 / 1024:.0f}MB"
                                local_path.unlink(missing_ok=True)
                                return None
                            f.write(chunk)
        except httpx.HTTPError as e:
            self.status = f"Download failed: {url} — {e}"
            return None
        return local_path

    # ---------- 编码一致性检测 ----------

    def _check_codecs_compatible(self, paths: list[Path]) -> bool:
        if len(paths) < 2:
            return True

        baseline = None
        for p in paths:
            info = self._get_codec_info(str(p))
            streams = info.get("streams", [])
            video_stream = None
            has_audio = False
            audio_codec = None
            for s in streams:
                if s.get("codec_type") == "video" and video_stream is None:
                    video_stream = s
                elif s.get("codec_type") == "audio":
                    has_audio = True
                    audio_codec = s.get("codec_name", "")

            if video_stream is None:
                return False

            sig = (
                video_stream.get("codec_name"),
                video_stream.get("width"),
                video_stream.get("height"),
                video_stream.get("pix_fmt"),
                has_audio,
                audio_codec if has_audio else None,
            )

            if baseline is None:
                baseline = sig
            elif baseline != sig:
                return False

        return True

    # ---------- ffmpeg 操作 ----------

    def _normalize_video(self, input_path: Path, output_path: Path) -> bool:
        if not self._check_ffprobe():
            self.log("ffprobe 不可用，跳过 normalize", "WARNING")
            return False
        has_audio = self._has_audio(str(input_path))

        cmd = ["ffmpeg", "-y", "-threads", str(FFMPEG_THREADS), "-i", str(input_path)]
        if not has_audio:
            cmd.extend(["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100"])

        cmd.extend([
            "-c:v", "libx264",
            "-threads", str(FFMPEG_THREADS),
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
        cmd = [
            "ffmpeg", "-y",
            "-threads", str(FFMPEG_THREADS),
            "-f", "concat", "-safe", "0",
            "-i", filelist_path,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        return result.returncode == 0

    def _concat_direct_copy(self, paths: list[Path], output_path: str) -> bool:
        filelist_path = paths[0].parent / "filelist.txt"
        with filelist_path.open("w", encoding="utf-8") as f:
            for p in paths:
                escaped = str(p).replace("'", "'\\''")
                f.write(f"file '{escaped}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-threads", str(FFMPEG_THREADS),
            "-f", "concat", "-safe", "0",
            "-i", str(filelist_path),
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, shell=False)
        if result.returncode != 0:
            from lfx.log.logger import logger
            logger.error(f"ffmpeg direct concat failed: {result.stderr}")
        return result.returncode == 0

    # ---------- 输入校验 ----------

    def _check_video_constraints(self, paths: list[Path]) -> str | None:
        total_duration = 0.0
        for p in paths:
            # 单文件大小
            size = self._get_file_size(p)
            if size > MAX_DOWNLOAD_SIZE:
                return f"文件过大: {p.name} ({size / 1024 / 1024:.0f}MB > {MAX_DOWNLOAD_SIZE / 1024 / 1024:.0f}MB)"

            # 时长
            duration = self._get_duration(str(p))
            total_duration += duration

            # 分辨率
            w, h = self._get_resolution(str(p))
            max_edge = max(w, h)
            if max_edge > MAX_RESOLUTION:
                return f"分辨率超限: {p.name} ({w}x{h}, 最长边 {max_edge}px > {MAX_RESOLUTION}px)"

        if total_duration > MAX_TOTAL_DURATION:
            return f"总时长超限: {total_duration:.0f}s > {MAX_TOTAL_DURATION:.0f}s"

        return None

    # ---------- 主流程 ----------

    def concat_videos(self) -> Data:
        if not self._check_ffprobe():
            self.status = "ffprobe/ffmpeg 未安装，无法执行视频拼接"
            return Data(text="", data={"error": self.status})

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

        # 并发限制：阻塞等待获取锁，10 分钟超时
        global _concat_lock_holder
        self.status = "等待 ffmpeg 锁..."
        self.log(f"等待并发锁（当前持有者: {_concat_lock_holder or '无'}）")
        acquired = _concat_lock.acquire(blocking=True, timeout=LOCK_WAIT_TIMEOUT)
        if not acquired:
            self.status = f"等待超时（{LOCK_WAIT_TIMEOUT}s），另一个任务仍在执行"
            return Data(text="", data={"error": self.status})
        _concat_lock_holder = f"job-{uuid.uuid4().hex[:8]}"
        self.log(f"获取并发锁: {_concat_lock_holder}")

        try:
            return self._do_concat(urls)
        finally:
            _concat_lock_holder = ""
            _concat_lock.release()

    def _do_concat(self, urls: list[str]) -> Data:
        with tempfile.TemporaryDirectory(prefix="video_concat_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)

            # 1. 下载
            local_paths: list[Path] = []
            for url in urls:
                path = self._download(url, tmp_dir)
                if path is None:
                    return Data(text="", data={"error": self.status or f"Failed to resolve: {url}"})
                local_paths.append(path)

            # 2. 输入校验
            validation_error = self._check_video_constraints(local_paths)
            if validation_error:
                self.status = validation_error
                return Data(text="", data={"error": validation_error})

            # 3. 检测编码一致性，尝试直接 copy 拼接
            if self._check_codecs_compatible(local_paths):
                self.log("编码一致，尝试 -c copy 直接拼接")
                output_dir = Path("uploads").resolve()
                output_dir.mkdir(parents=True, exist_ok=True)
                fmt = self.output_format or "mp4"
                output_path = output_dir / f"concat_{uuid.uuid4().hex[:8]}.{fmt}"

                if self._concat_direct_copy(local_paths, str(output_path)):
                    duration = self._get_duration(str(output_path))
                    self.status = f"Direct concat {len(local_paths)} videos → {output_path} ({duration:.1f}s)"
                    return Data(
                        text=str(output_path),
                        data={"path": str(output_path), "duration": duration, "count": len(local_paths)},
                    )
                else:
                    self.log("直接拼接失败，回退到 normalize + concat", "WARNING")

            # 4. 不一致或 copy 失败 → normalize 后再 concat
            normalized_dir = tmp_dir / "normalized"
            normalized_dir.mkdir()

            normalized_paths: list[Path] = []
            for idx, path in enumerate(local_paths):
                norm_path = normalized_dir / f"norm_{idx:03d}.mp4"
                if not self._normalize_video(path, norm_path):
                    self.status = f"Failed to normalize video: {path}"
                    return Data(text="", data={"error": self.status})
                normalized_paths.append(norm_path)

            filelist_path = normalized_dir / "filelist.txt"
            with filelist_path.open("w", encoding="utf-8") as f:
                for p in normalized_paths:
                    escaped = str(p).replace("'", "'\\''")
                    f.write(f"file '{escaped}'\n")

            output_dir = Path("uploads").resolve()
            output_dir.mkdir(parents=True, exist_ok=True)
            fmt = self.output_format or "mp4"
            output_path = output_dir / f"concat_{uuid.uuid4().hex[:8]}.{fmt}"

            if not self._concat_normalized(str(filelist_path), str(output_path)):
                self.status = "FFmpeg concatenation failed"
                return Data(text="", data={"error": self.status})

            duration = self._get_duration(str(output_path))
            self.status = f"Concatenated {len(local_paths)} videos → {output_path} ({duration:.1f}s)"

            return Data(
                text=str(output_path),
                data={"path": str(output_path), "duration": duration, "count": len(local_paths)},
            )