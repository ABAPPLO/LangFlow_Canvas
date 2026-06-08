import json
import time
from ipaddress import ip_address
from urllib.parse import quote
from uuid import uuid4

from lfx.custom import Component
from lfx.io import IntInput, MultilineInput, Output, SecretStrInput, StrInput
from lfx.schema.data import Data
from lfx.schema.message import Message


class TencentMPSURLMergerWatermarkComponent(Component):
    display_name = "腾讯云 MPS 视频拼接"
    description = "拼接 URL 视频，可选添加 URL 图片水印"
    icon = "Video"
    name = "TencentMPSURLMergerWatermark"

    TRANSCODE_DEFINITION = 101005

    inputs = [
        SecretStrInput(
            name="secret_id",
            display_name="Secret ID",
            info="腾讯云 API SecretId。",
            required=True,
            advanced=True,
        ),
        SecretStrInput(
            name="secret_key",
            display_name="Secret Key",
            info="腾讯云 API SecretKey。",
            required=True,
            advanced=True,
        ),
        StrInput(
            name="region",
            display_name="MPS Region",
            info="腾讯云 MPS 服务地域，如 ap-guangzhou、ap-shanghai。",
            value="ap-guangzhou",
            required=True,
            advanced=True,
        ),
        MultilineInput(
            name="video_urls",
            display_name="视频 URL 列表",
            info='待拼接视频 URL。支持 JSON 数组，如 ["https://example.com/a.mp4"]；也支持每行一个 URL。',
            required=True,
        ),
        StrInput(
            name="output_bucket",
            display_name="输出 COS Bucket",
            info="输出视频所在的 COS Bucket，格式通常为 bucket-appid。",
            required=True,
            advanced=True,
        ),
        StrInput(
            name="output_region",
            display_name="输出 COS Region",
            info="输出 COS Bucket 所在地域，如 ap-guangzhou、ap-shanghai。",
            value="ap-guangzhou",
            required=True,
            advanced=True,
        ),
        StrInput(
            name="output_dir",
            display_name="输出目录",
            info="输出文件目录，如 langflow/videos。组件会自动处理前后斜杠。",
            value="videos/",
            required=True,
            advanced=True,
        ),
        MultilineInput(
            name="watermarks",
            display_name="水印配置",
            required=False,
            input_types=[],
            info=(
                "可选。留空则不添加水印。"
                "填写 JSON 数组，每个对象代表一个图片水印。"
                "示例："
                '[{"url":"https://example.com/watermark.png",'
                '"origin":"TopRight","x":"3%","y":"3%",'
                '"width":"12%","height":""}]'
            ),
            value="",
        ),
        IntInput(
            name="timeout",
            display_name="超时时间(分钟)",
            info="等待腾讯云 MPS 任务完成的最长分钟数。",
            value=5,
            required=True,
            advanced=True,
        ),
    ]

    outputs = [
        Output(name="video_url", display_name="最终视频 URL", method="merge_and_watermark"),
        Output(name="task_info", display_name="任务信息", method="get_task_info"),
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._result: dict | None = None
        self._error: Exception | None = None

    @staticmethod
    def _text(value) -> str:
        if isinstance(value, Message):
            return value.get_text()
        return str(value or "")

    @staticmethod
    def _validate_public_http_url(url: str, field_name: str) -> str:
        from urllib.parse import urlparse

        value = str(url or "").strip()
        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{field_name} 必须是 http/https URL: {value}")

        hostname = parsed.hostname or ""
        if hostname.lower() in {"localhost"} or hostname.lower().endswith(".local"):
            raise ValueError(f"{field_name} 不允许使用本地地址: {value}")

        try:
            host_ip = ip_address(hostname)
        except ValueError:
            return value

        if host_ip.is_private or host_ip.is_loopback or host_ip.is_link_local or host_ip.is_reserved:
            raise ValueError(f"{field_name} 不允许使用内网或保留地址: {value}")
        return value

    def _parse_video_urls(self) -> list[str]:
        raw = self._text(self.video_urls).strip()
        if not raw:
            raise ValueError("视频 URL 列表不能为空")

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = [line.strip() for line in raw.splitlines() if line.strip()]

        if isinstance(parsed, str):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError("视频 URL 列表必须是 JSON 数组，或每行一个 URL")

        urls = [self._validate_public_http_url(item, f"视频 URL[{index}]") for index, item in enumerate(parsed)]
        if not urls:
            raise ValueError("视频 URL 列表不能为空")
        return urls

    def _parse_watermarks(self) -> list[dict]:
        raw = self._text(getattr(self, "watermarks", "")).strip()
        if not raw:
            return []

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ValueError(f"水印配置不是合法 JSON: {e}") from e

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError("水印配置必须是 JSON 数组")

        watermarks = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"水印配置[{index}] 必须是对象")
            url = self._validate_public_http_url(item.get("url", ""), f"水印 URL[{index}]")
            watermarks.append(
                {
                    "url": url,
                    "origin": str(item.get("origin") or "TopRight").strip(),
                    "x": str(item.get("x") or "3%").strip(),
                    "y": str(item.get("y") or "3%").strip(),
                    "width": str(item.get("width") or "12%").strip(),
                    "height": str(item.get("height") or "").strip(),
                }
            )
        return watermarks

    def _timeout_seconds(self) -> int:
        timeout = int(getattr(self, "timeout", 5) or 5)
        if timeout < 1 or timeout > 180:
            raise ValueError("超时时间必须在 1 到 180 分钟之间")
        return timeout * 60

    def _mps_client(self):
        try:
            from tencentcloud.common import credential
            from tencentcloud.mps.v20190612 import mps_client
        except ImportError as e:
            msg = (
                "tencentcloud-sdk-python is not installed. "
                "Please install it using: uv pip install tencentcloud-sdk-python"
            )
            raise ImportError(msg) from e

        cred = credential.Credential(self.secret_id.strip(), self.secret_key.strip())
        return mps_client.MpsClient(cred, self.region.strip())

    def _output_storage(self) -> dict:
        return {
            "Type": "COS",
            "CosOutputStorage": {
                "Bucket": self.output_bucket.strip(),
                "Region": self.output_region.strip(),
            },
        }

    def _output_dir(self) -> str:
        output_dir = self.output_dir.strip().strip("/")
        if not output_dir:
            raise ValueError("输出目录不能为空")
        return f"/{output_dir}/"

    def _public_cos_url(self, path: str) -> str:
        return (
            f"https://{self.output_bucket.strip()}.cos.{self.output_region.strip()}.myqcloud.com"
            f"{quote(path, safe='/._-')}"
        )

    def wait_task(self, client, task_id: str, label: str):
        start_time = time.time()
        timeout_seconds = self._timeout_seconds()

        while time.time() - start_time < timeout_seconds:
            time.sleep(3)

            from tencentcloud.mps.v20190612 import models

            req = models.DescribeTaskDetailRequest()
            req.from_json_string(json.dumps({"TaskId": task_id}))

            detail = json.loads(client.DescribeTaskDetail(req).to_json_string())
            status = detail["Status"]

            self.status = f"{label}任务状态：{status}"

            if status == "FINISH":
                return detail

            if status == "FAIL":
                raise RuntimeError(f"{label}任务失败\n{json.dumps(detail, ensure_ascii=False)}")

        raise TimeoutError(f"{label}任务超时，任务ID: {task_id}")

    def _run_mps_tasks(self) -> dict:
        try:
            from tencentcloud.mps.v20190612 import models

            video_urls = self._parse_video_urls()
            watermarks = self._parse_watermarks()

            output_dir = self._output_dir()
            file_id = uuid4().hex
            merged_path = f"{output_dir}merged_{file_id}"
            final_path = f"{output_dir}watermarked_{file_id}"

            client = self._mps_client()
            output_storage = self._output_storage()

            file_infos = []
            video_items = []

            for index, video_url in enumerate(video_urls):
                material_id = f"video_{index}"

                file_infos.append(
                    {
                        "Id": material_id,
                        "InputInfo": {
                            "Type": "URL",
                            "UrlInputInfo": {
                                "Url": video_url,
                            },
                        },
                    }
                )

                video_items.append(
                    {
                        "Type": "Video",
                        "Video": {
                            "SourceMedia": {
                                "FileId": material_id,
                            },
                        },
                    }
                )

            edit_req = models.EditMediaRequest()
            edit_req.from_json_string(
                json.dumps(
                    {
                        "FileInfos": file_infos,
                        "OutputStorage": output_storage,
                        "OutputObjectPath": merged_path,
                        "ComposeConfig": {
                            "TargetInfo": {
                                "Container": "mp4",
                            },
                            "Tracks": [
                                {
                                    "Type": "Video",
                                    "Items": video_items,
                                }
                            ],
                        },
                    },
                    ensure_ascii=False,
                )
            )

            edit_res = client.EditMedia(edit_req)
            edit_task_id = json.loads(edit_res.to_json_string())["TaskId"]

            self.status = f"已提交拼接任务：{edit_task_id}"
            edit_detail = self.wait_task(client, edit_task_id, "拼接")

            real_merged_path = edit_detail["EditMediaTask"]["Output"]["Path"]
            merged_url = self._public_cos_url(real_merged_path)

            if not watermarks:
                self.status = "拼接完成，未配置水印，返回拼接视频 URL"
                return {
                    "merged_url": merged_url,
                    "final_url": merged_url,
                    "edit_task_id": edit_task_id,
                    "process_task_id": "",
                    "watermark_enabled": False,
                    "output_bucket": self.output_bucket.strip(),
                    "output_region": self.output_region.strip(),
                }

            watermark_set = []

            for watermark in watermarks:
                watermark_set.append(
                    {
                        "Definition": 0,
                        "RawParameter": {
                            "Type": "image",
                            "CoordinateOrigin": watermark["origin"],
                            "XPos": watermark["x"],
                            "YPos": watermark["y"],
                            "ImageTemplate": {
                                "ImageContent": {
                                    "Type": "URL",
                                    "UrlInputInfo": {
                                        "Url": watermark["url"],
                                    },
                                },
                                "Width": watermark["width"],
                                "Height": watermark["height"],
                            },
                        },
                        "StartTimeOffset": 0,
                        "EndTimeOffset": 0,
                    }
                )

            process_req = models.ProcessMediaRequest()
            process_req.from_json_string(
                json.dumps(
                    {
                        "InputInfo": {
                            "Type": "URL",
                            "UrlInputInfo": {
                                "Url": merged_url,
                            },
                        },
                        "OutputStorage": output_storage,
                        "OutputDir": output_dir,
                        "MediaProcessTask": {
                            "TranscodeTaskSet": [
                                {
                                    "Definition": self.TRANSCODE_DEFINITION,
                                    "OutputObjectPath": final_path,
                                    "WatermarkSet": watermark_set,
                                }
                            ]
                        },
                    },
                    ensure_ascii=False,
                )
            )

            process_res = client.ProcessMedia(process_req)
            process_task_id = json.loads(process_res.to_json_string())["TaskId"]

            self.status = f"已提交水印任务：{process_task_id}"
            process_detail = self.wait_task(client, process_task_id, "水印")

            real_final_path = process_detail["WorkflowTask"]["MediaProcessResultSet"][0]["TranscodeTask"]["Output"][
                "Path"
            ]

            final_url = self._public_cos_url(real_final_path)

            return {
                "merged_url": merged_url,
                "final_url": final_url,
                "edit_task_id": edit_task_id,
                "process_task_id": process_task_id,
                "watermark_enabled": True,
                "output_bucket": self.output_bucket.strip(),
                "output_region": self.output_region.strip(),
            }

        except Exception as e:  # noqa: BLE001
            msg = f"腾讯云 MPS 视频拼接失败: {e}"
            self.status = msg
            raise ValueError(msg) from e

    def _ensure_result(self, *, raise_on_error: bool = True) -> None:
        if self._result is not None:
            return
        if self._error is not None:
            if raise_on_error:
                raise self._error
            return

        try:
            self._result = self._run_mps_tasks()
        except Exception as e:  # noqa: BLE001
            self._error = e
            if raise_on_error:
                raise

    def merge_and_watermark(self) -> Message:
        self._ensure_result()
        return Message(text=(self._result or {}).get("final_url", ""), sender="System")

    def get_task_info(self) -> Data:
        self._ensure_result(raise_on_error=False)
        if self._result is not None:
            return Data(data=self._result)
        return Data(data={"error": str(self._error) if self._error else ""})
