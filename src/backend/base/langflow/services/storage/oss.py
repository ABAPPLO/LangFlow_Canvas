"""Alibaba Cloud OSS storage service implementation."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from langflow.logging.logger import logger

from .service import StorageService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langflow.services.session.service import SessionService
    from langflow.services.settings.service import SettingsService


class OSSStorageService(StorageService):
    """A service class for handling Alibaba Cloud OSS storage operations."""

    def __init__(self, session_service: SessionService, settings_service: SettingsService) -> None:
        super().__init__(session_service, settings_service)

        self.bucket_name = settings_service.settings.object_storage_bucket_name
        if not self.bucket_name:
            msg = "OSS bucket name is required when using OSS storage. Set LANGFLOW_OBJECT_STORAGE_BUCKET_NAME."
            raise ValueError(msg)

        endpoint = settings_service.settings.oss_endpoint
        access_key_id = settings_service.settings.oss_access_key_id
        access_key_secret = settings_service.settings.oss_access_key_secret

        if not all([endpoint, access_key_id, access_key_secret]):
            msg = (
                "OSS endpoint, access_key_id, and access_key_secret are required. "
                "Set LANGFLOW_OSS_ENDPOINT, LANGFLOW_OSS_ACCESS_KEY_ID, LANGFLOW_OSS_ACCESS_KEY_SECRET."
            )
            raise ValueError(msg)

        self.prefix = settings_service.settings.object_storage_prefix or ""
        if self.prefix and not self.prefix.endswith("/"):
            self.prefix += "/"

        self.tags = settings_service.settings.object_storage_tags or {}

        try:
            import oss2
        except ImportError as exc:
            msg = "oss2 is required for OSS storage. Install with: uv pip install oss2"
            raise ImportError(msg) from exc

        auth = oss2.Auth(access_key_id.strip(), access_key_secret.strip())
        self._oss2 = oss2
        self.bucket = oss2.Bucket(auth, endpoint.strip(), self.bucket_name)

        self.set_ready()
        logger.info(f"OSS storage initialized: bucket={self.bucket_name}, prefix={self.prefix}, endpoint={endpoint}")

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
            msg = "Append mode is not supported for OSS storage"
            raise NotImplementedError(msg)

        key = self.build_full_path(flow_id, file_name)

        try:
            await asyncio.to_thread(self._put_object, key, data)
        except Exception as e:  # noqa: BLE001
            self._map_and_raise(e, "save", file_name)
        else:
            await logger.ainfo(f"File {file_name} saved to OSS: {self.bucket_name}/{key}")

    async def get_file(self, flow_id: str, file_name: str) -> bytes:
        key = self.build_full_path(flow_id, file_name)

        try:
            content = await asyncio.to_thread(self._get_object, key)
        except Exception as e:  # noqa: BLE001
            self._map_and_raise(e, "get", file_name)
        else:
            logger.debug(f"File {file_name} retrieved from OSS: {self.bucket_name}/{key}")
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
            logger.exception(f"Error listing files in OSS flow {flow_id}")
            raise
        else:
            return files

    async def delete_file(self, flow_id: str, file_name: str) -> None:
        key = self.build_full_path(flow_id, file_name)

        try:
            await asyncio.to_thread(self._delete_object, key)
        except Exception:
            logger.exception(f"Error deleting file {file_name} from OSS in flow {flow_id}")
            raise

    async def get_file_size(self, flow_id: str, file_name: str) -> int:
        key = self.build_full_path(flow_id, file_name)

        try:
            size = await asyncio.to_thread(self._head_object, key)
        except Exception as e:  # noqa: BLE001
            self._map_and_raise(e, "get_size", file_name)
        else:
            return size

    async def teardown(self) -> None:
        logger.info("OSS storage service teardown complete")

    # -- Synchronous helpers (called via asyncio.to_thread) --

    def _put_object(self, key: str, data: bytes) -> None:
        self.bucket.put_object(key, data)
        if self.tags:
            tag_set = [{"Key": k, "Value": v} for k, v in self.tags.items()]
            self.bucket.put_object_tagging(key, tag_set)

    def _get_object(self, key: str) -> bytes:
        result = self.bucket.get_object(key)
        return result.read()

    def _list_objects(self, prefix: str) -> list[str]:
        files = []
        for obj_info in self._oss2.ObjectIterator(self.bucket, prefix=prefix):
            file_name = obj_info.key[len(prefix) :]
            if file_name:
                files.append(file_name)
        return files

    def _delete_object(self, key: str) -> None:
        self.bucket.delete_object(key)

    def _head_object(self, key: str) -> int:
        info = self.bucket.head_object(key)
        return info.content_length

    # -- Error handling helpers --

    def _map_and_raise(self, e: Exception, operation: str, file_name: str) -> None:
        oss2 = self._oss2

        if isinstance(e, (oss2.exceptions.NoSuchKey, oss2.exceptions.NoSuchBucket)):
            msg = f"File not found: {file_name}"
            raise FileNotFoundError(msg) from e
        if isinstance(e, (oss2.exceptions.AccessDenied, oss2.exceptions.SignatureDoesNotMatch)):
            msg = "Access denied to OSS. Check your AccessKey and bucket permissions"
            raise PermissionError(msg) from e
        msg = f"Failed to {operation} file in OSS: {e}"
        raise RuntimeError(msg) from e
