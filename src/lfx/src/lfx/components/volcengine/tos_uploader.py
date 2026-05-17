from pathlib import Path

from lfx.custom import Component
from lfx.io import (
    DropdownInput,
    MessageTextInput,
    MultilineInput,
    Output,
    SecretStrInput,
    StrInput,
)

MODE_BATCH = "Batch Upload"
MODE_SINGLE = "Single Upload"


class TOSUploaderComponent(Component):
    display_name = "TOS File Uploader"
    description = "上传文件到火山引擎对象存储（TOS），输出文件访问 URL。"
    icon = "Upload"
    name = "TOSUploader"

    inputs = [
        SecretStrInput(
            name="access_key",
            display_name="AccessKey",
            info="火山引擎 AccessKey ID。",
            required=True,
        ),
        SecretStrInput(
            name="secret_key",
            display_name="SecretKey",
            info="火山引擎 AccessKey Secret。",
            required=True,
        ),
        StrInput(
            name="region",
            display_name="Region",
            info="存储桶地域，如 cn-beijing、cn-guangzhou。",
            value="cn-beijing",
        ),
        StrInput(
            name="endpoint",
            display_name="Endpoint",
            info="TOS 访问域名，如 tos-cn-beijing.volces.com。留空则根据 Region 自动拼接。",
            advanced=True,
        ),
        StrInput(
            name="bucket_name",
            display_name="Bucket",
            info="存储桶名称。",
            required=True,
        ),
        StrInput(
            name="tos_prefix",
            display_name="Path Prefix",
            info="上传路径前缀，如 uploads/。留空则上传到根目录。",
            advanced=True,
        ),
        DropdownInput(
            name="upload_mode",
            display_name="Upload Mode",
            info="Batch：多文件路径手动填写；Single：单文件，支持组件连线输入。",
            options=[MODE_BATCH, MODE_SINGLE],
            value=MODE_BATCH,
            real_time_refresh=True,
        ),
        MultilineInput(
            name="file_paths",
            display_name="File Paths",
            info="本地文件路径，每行一个。支持绝对路径和相对路径。",
            dynamic=True,
            show=True,
        ),
        MessageTextInput(
            name="file_path",
            display_name="File Path",
            info="单个文件路径，支持手动输入或从其他组件连线。",
            input_types=["Message", "Text"],
            dynamic=True,
            show=False,
        ),
        DropdownInput(
            name="acl",
            display_name="ACL",
            options=["private", "public-read"],
            value="public-read",
            info="文件访问权限：private（私有读写）、public-read（公有读私有写）。",
        ),
    ]
    outputs = [
        Output(display_name="URLs", name="urls", method="upload_files"),
    ]

    def _tos_client(self):
        try:
            from tos import TosClientV2
        except ImportError as e:
            msg = (
                "tos is not installed. "
                "Please install it using: uv pip install tos"
            )
            raise ImportError(msg) from e

        region = self.region.strip()
        endpoint = (self.endpoint or "").strip()
        if not endpoint:
            endpoint = f"tos-{region}.volces.com"

        return TosClientV2(
            ak=self.access_key.strip(),
            sk=self.secret_key.strip(),
            endpoint=endpoint,
            region=region,
        )

    def _build_key(self, file_path: str) -> str:
        filename = Path(file_path).name
        prefix = (self.tos_prefix or "").strip()
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return f"{prefix}{filename}"

    def _build_url(self, key: str) -> str:
        region = self.region.strip()
        endpoint = (self.endpoint or "").strip()
        if not endpoint:
            endpoint = f"tos-{region}.volces.com"
        return f"https://{self.bucket_name.strip()}.{endpoint}/{key}"

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "upload_mode":
            is_single = field_value == MODE_SINGLE
            build_config["file_paths"]["show"] = not is_single
            build_config["file_path"]["show"] = is_single
        return build_config

    def _upload_one(self, client, file_path: str) -> str | None:
        p = Path(file_path).expanduser().resolve()
        if not p.exists():
            self.status = f"File not found: {file_path}"
            return None

        key = self._build_key(file_path)
        client.put_object_from_file(
            bucket=self.bucket_name.strip(),
            key=key,
            file_path=str(p),
            acl_header={
                "x-tos-acl": self.acl,
            },
        )
        return self._build_url(key)

    def upload_files(self) -> str:
        mode = getattr(self, "upload_mode", MODE_BATCH)

        if mode == MODE_SINGLE:
            path = self.file_path
            if not path:
                return ""
            from lfx.schema.message import Message

            if isinstance(path, Message):
                path = path.get_text()
            path = str(path).strip()

            client = self._tos_client()
            url = self._upload_one(client, path)
            if url:
                self.status = f"Uploaded: {url}"
            return url or ""

        if not self.file_paths:
            return ""

        client = self._tos_client()
        paths = [p.strip() for p in self.file_paths.splitlines() if p.strip()]
        urls = [u for fp in paths if (u := self._upload_one(client, fp))]

        self.status = f"Uploaded {len(urls)} file(s)"
        return "\n".join(urls)
