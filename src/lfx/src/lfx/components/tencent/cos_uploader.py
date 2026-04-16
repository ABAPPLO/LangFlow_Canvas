from pathlib import Path

from lfx.custom import Component
from lfx.io import DropdownInput, MultilineInput, Output, SecretStrInput, StrInput


class COSUploaderComponent(Component):
    display_name = "COS File Uploader"
    description = "上传文件到腾讯云对象存储（COS），输出文件访问 URL。"
    icon = "Upload"
    name = "COSUploader"

    inputs = [
        SecretStrInput(
            name="secret_id",
            display_name="SecretId",
            info="腾讯云 API SecretId。",
            required=True,
        ),
        SecretStrInput(
            name="secret_key",
            display_name="SecretKey",
            info="腾讯云 API SecretKey。",
            required=True,
        ),
        StrInput(
            name="region",
            display_name="Region",
            info="存储桶地域，如 ap-guangzhou、ap-beijing、ap-shanghai。",
            value="ap-guangzhou",
        ),
        StrInput(
            name="bucket_name",
            display_name="Bucket",
            info="存储桶名称，格式为 bucket-appid，如 my-bucket-1250000000。",
            required=True,
        ),
        StrInput(
            name="cos_prefix",
            display_name="Path Prefix",
            info="上传路径前缀，如 uploads/。留空则上传到根目录。",
            advanced=True,
        ),
        DropdownInput(
            name="mode",
            display_name="Mode",
            options=["Batch", "Single"],
            value="Batch",
            info="Batch：多行路径输入，多行 URL 输出；Single：单个文件路径输入，单个 URL 输出。",
        ),
        MultilineInput(
            name="file_paths",
            display_name="File Paths",
            info="本地文件路径，每行一个。支持绝对路径和相对路径。",
            show=False,
        ),
        StrInput(
            name="file_path",
            display_name="File Path",
            info="单个本地文件路径。支持绝对路径和相对路径。",
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
        Output(display_name="URL", name="url", method="upload_single"),
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "mode":
            is_single = field_value == "Single"
            build_config["file_paths"]["show"] = not is_single
            build_config["file_path"]["show"] = is_single
        return build_config

    def _cos_client(self):
        try:
            from qcloud_cos import CosConfig, CosS3Client
        except ImportError as e:
            msg = (
                "cos-python-sdk-v5 is not installed. "
                "Please install it using: uv pip install cos-python-sdk-v5"
            )
            raise ImportError(msg) from e

        config = CosConfig(
            Region=self.region.strip(),
            SecretId=self.secret_id.strip(),
            SecretKey=self.secret_key.strip(),
        )
        return CosS3Client(config)

    def _build_key(self, file_path: str) -> str:
        filename = Path(file_path).name
        prefix = (self.cos_prefix or "").strip()
        if prefix and not prefix.endswith("/"):
            prefix += "/"
        return f"{prefix}{filename}"

    def _build_url(self, key: str) -> str:
        return f"https://{self.bucket_name.strip()}.cos.{self.region.strip()}.myqcloud.com/{key}"

    def _upload_one(self, client, fp: str) -> str | None:
        p = Path(fp).expanduser().resolve()
        if not p.exists():
            self.status = f"File not found: {fp}"
            return None

        key = self._build_key(fp)
        client.upload_file(
            Bucket=self.bucket_name.strip(),
            Key=key,
            LocalFilePath=str(p),
            ACL=self.acl,
        )
        return self._build_url(key)

    def upload_files(self) -> str:
        if not self.file_paths:
            return ""

        client = self._cos_client()
        paths = [p.strip() for p in self.file_paths.splitlines() if p.strip()]
        urls = []

        for fp in paths:
            url = self._upload_one(client, fp)
            if url:
                urls.append(url)

        return "\n".join(urls)

    def upload_single(self) -> str:
        if not self.file_path:
            return ""

        client = self._cos_client()
        url = self._upload_one(client, self.file_path.strip())
        if not url:
            return ""
        self.status = url
        return url
