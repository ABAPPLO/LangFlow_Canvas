"""Tencent Cloud COS storage service implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from langflow.logging.logger import logger

from .service import StorageService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langflow.services.session.service import SessionService
    from langflow.services.settings.service import SettingsService


class COSStorageService(StorageService):
    """A service class for handling Tencent Cloud COS storage operations."""

    def __init__(self, session_service: SessionService, settings_service: SettingsService) -> None:
        super().__init__(session_service, settings_service)

        self.bucket_name = settings_service.settings.object_storage_bucket_name
        if not self.bucket_name:
            msg = "COS bucket name is required when using COS storage. Set LANGFLOW_OBJECT_STORAGE_BUCKET_NAME."
            raise ValueError(msg)

        region = settings_service.settings.cos_region
        secret_id = settings_service.settings.cos_secret_id
        secret_key = settings_service.settings.cos_secret_key

        if not all([region, secret_id, secret_key]):
            msg = (
                "COS region, secret_id, and secret_key are required. "
                "Set LANGFLOW_COS_REGION, LANGFLOW_COS_SECRET_ID, LANGFLOW_COS_SECRET_KEY."
            )
            raise ValueError(msg)

        self.prefix = settings_service.settings.object_storage_prefix or ""
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"

        self.tags = settings_service.settings.object_storage_tags or {}

        try:
            from qcloud_cos import CosConfig, CosS3Client
        except ImportError as exc:
            msg = "cos-python-sdk-v5 is required for COS storage. Install with: uv pip install cos-python-sdk-v5"
            raise ImportError(msg) from exc

        config = CosConfig(Region=region.strip(), SecretId=secret_id.strip(), SecretKey=secret_key.strip())
        self.client = CosS3Client(config)

        self.set_ready()
        logger.info(f"COS storage initialized: bucket={self.bucket_name}, prefix={self.prefix}, region={region}")

    def build_full_path(self, flow_id: str, file_name: str) -> str:
        return f"{self.prefix}{flow_id}/{file_name}"

    def parse_file_path(self, full_path: str) -> tuple[str, str]:
        path_without_prefix = full_path
        if self.prefix and full_path.startswith(self.prefix):
            path_without_prefix = full_path[len(self.prefix) :]

        if "/" not in path_without_prefix:
            return "", path_without_prefix

        flow_id, file_name = path_without_prefix.rsplit("/", 1)
        return flow_id, file_name

    def resolve_component_path(self, logical_path: str) -> str:
        return logical_path

    async def save_file(self, flow_id: str, file_name: str, data: bytes, *, append: bool = False) -> None:
        if append:
            msg = "Append mode is not supported for COS storage"
            raise NotImplementedError(msg)

        key = self.build_full_path(flow_id, file_name)

        try:
            await asyncio.to_thread(self._put_object, key, data)
        except Exception as e:  # noqa: BLE001
            error_msg = str(e)
            await logger.aerror(f"Error saving file {file_name} to COS in flow {flow_id}: {error_msg}")
            self._map_save_error(e, file_name)
        else:
            await logger.ainfo(f"File {file_name} saved to COS: {self.bucket_name}/{key}")

    async def get_file(self, flow_id: str, file_name: str) -> bytes:
        key = self.build_full_path(flow_id, file_name)

        try:
            content = await asyncio.to_thread(self._get_object, key)
        except Exception as e:
            error_msg = str(e)
            if "NoSuchKey" in error_msg or "404" in error_msg:
                await logger.awarning(f"File {file_name} not found in COS flow {flow_id}")
                msg = f"File not found: {file_name}"
                raise FileNotFoundError(msg) from e
            logger.exception(f"Error retrieving file {file_name} from COS in flow {flow_id}")
            raise
        else:
            logger.debug(f"File {file_name} retrieved from COS: {self.bucket_name}/{key}")
            return content

    async def get_file_stream(self, flow_id: str, file_name: str, chunk_size: int = 8192) -> AsyncIterator[bytes]:
        content = await self.get_file(flow_id, file_name)
        for i in range(0, len(content), chunk_size):
            yield content[i : i + chunk_size]

    async def list_files(self, flow_id: str) -> list[str]:
        if not isinstance(flow_id, str):
            flow_id = str(flow_id)

        prefix = self.build_full_path(flow_id, "")

        try:
            files = await asyncio.to_thread(self._list_objects, prefix)
        except Exception:
            logger.exception(f"Error listing files in COS flow {flow_id}")
            raise
        else:
            return files

    async def delete_file(self, flow_id: str, file_name: str) -> None:
        key = self.build_full_path(flow_id, file_name)

        try:
            await asyncio.to_thread(self._delete_object, key)
        except Exception:
            logger.exception(f"Error deleting file {file_name} from COS in flow {flow_id}")
            raise

    async def get_file_size(self, flow_id: str, file_name: str) -> int:
        key = self.build_full_path(flow_id, file_name)

        try:
            size = await asyncio.to_thread(self._head_object, key)
        except Exception as e:
            error_msg = str(e)
            if "NoSuchKey" in error_msg or "404" in error_msg:
                await logger.awarning(f"File {file_name} not found in COS flow {flow_id}")
                msg = f"File not found: {file_name}"
                raise FileNotFoundError(msg) from e
            logger.exception(f"Error getting file size for {file_name} in COS flow {flow_id}")
            raise
        else:
            return size

    async def teardown(self) -> None:
        logger.info("COS storage service teardown complete")

    # -- Synchronous helpers (called via asyncio.to_thread) --

    def _put_object(self, key: str, data: bytes) -> dict:
        params: dict = {"Bucket": self.bucket_name, "Key": key, "Body": data}
        if self.tags:
            tag_string = "&".join(f"{k}={v}" for k, v in self.tags.items())
            params["Tagging"] = tag_string
        return self.client.put_object(**params)

    def _get_object(self, key: str) -> bytes:
        resp = self.client.get_object(Bucket=self.bucket_name, Key=key)
        body = resp["Body"]
        return body.read()

    def _list_objects(self, prefix: str) -> list[str]:
        files = []
        marker = ""
        while True:
            resp = self.client.list_objects(Bucket=self.bucket_name, Prefix=prefix, Marker=marker, MaxKeys=1000)
            contents = resp.get("Contents", [])
            for obj in contents:
                file_name = obj["Key"][len(prefix) :]
                if file_name:
                    files.append(file_name)
            if resp.get("IsTruncated") == "true":
                marker = contents[-1]["Key"] if contents else ""
            else:
                break
        return files

    def _delete_object(self, key: str) -> None:
        self.client.delete_object(Bucket=self.bucket_name, Key=key)

    def _head_object(self, key: str) -> int:
        resp = self.client.head_object(Bucket=self.bucket_name, Key=key)
        return int(resp.get("Content-Length", 0))

    # -- Error handling helpers --

    @staticmethod
    def _map_save_error(e: Exception, file_name: str) -> None:
        error_msg = str(e)
        if "NoSuchBucket" in error_msg:
            msg = f"COS bucket does not exist for file '{file_name}'"
            raise FileNotFoundError(msg) from e
        if "AccessDenied" in error_msg or "SignatureDoesNotMatch" in error_msg:
            msg = "Access denied to COS. Check your SecretId/SecretKey and bucket permissions"
            raise PermissionError(msg) from e
        msg = f"Failed to save file to COS: {error_msg}"
        raise RuntimeError(msg) from e
