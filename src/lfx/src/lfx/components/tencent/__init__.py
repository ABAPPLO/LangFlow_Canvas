"""Tencent module - COS and other Tencent Cloud components."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lfx.components._importing import import_mod

if TYPE_CHECKING:
    from lfx.components.tencent.cos_uploader import COSUploaderComponent
    from lfx.components.tencent.remote_video_to_cos import RemoteVideoToCOSComponent

_dynamic_imports = {
    "COSUploaderComponent": "cos_uploader",
    "RemoteVideoToCOSComponent": "remote_video_to_cos",
}

__all__ = ["COSUploaderComponent", "RemoteVideoToCOSComponent"]


def __getattr__(attr_name: str) -> Any:
    """Lazily import Tencent components on attribute access."""
    if attr_name not in _dynamic_imports:
        msg = f"module '{__name__}' has no attribute '{attr_name}'"
        raise AttributeError(msg)
    try:
        result = import_mod(attr_name, _dynamic_imports[attr_name], __spec__.parent)
    except (ModuleNotFoundError, ImportError, AttributeError) as e:
        msg = f"Could not import '{attr_name}' from '{__name__}': {e}"
        raise AttributeError(msg) from e
    globals()[attr_name] = result
    return result


def __dir__() -> list[str]:
    return list(__all__)
