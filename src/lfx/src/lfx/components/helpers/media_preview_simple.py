import base64
from pathlib import Path

from lfx.custom import Component
from lfx.io import DropdownInput, MultilineInput, Output
from lfx.utils.helpers import get_mime_type


def _file_to_data_uri(file_path: str) -> str:
    """Read a local file and convert it to a data URI."""
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return f"file_not_found:{file_path}"
    mime = get_mime_type(p)
    data = base64.b64encode(p.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


class MediaPreviewSimpleComponent(Component):
    display_name = "Media Preview Simple"
    description = "Receives media URLs, file paths, or raw base64 content and displays previews below the node."
    icon = "Play"
    name = "MediaPreviewSimple"

    inputs = [
        DropdownInput(
            name="input_type",
            display_name="Input Type",
            options=["URL", "File Path", "Raw Content (Base64)"],
            value="URL",
            info="Select the input format: URL for web links, File Path for local files, Raw Content for base64 data URIs.",
            real_time_refresh=True,
        ),
        MultilineInput(
            name="media_url",
            display_name="Media URL",
            info="Media URL to preview. Multiple URLs can be separated by newlines.",
        ),
        MultilineInput(
            name="file_path",
            display_name="File Path",
            info="Local file path(s) to preview. One path per line.",
            show=False,
        ),
        MultilineInput(
            name="raw_content",
            display_name="Raw Content (Base64)",
            info="Base64 data URI(s), e.g. data:image/png;base64,iVBOR... One per line.",
            show=False,
        ),
    ]
    outputs = [
        Output(display_name="Preview", name="preview", method="preview_media"),
    ]

    def update_build_config(self, build_config, field_value, field_name=None):
        if field_name == "input_type":
            build_config["media_url"]["show"] = field_value == "URL"
            build_config["file_path"]["show"] = field_value == "File Path"
            build_config["raw_content"]["show"] = field_value == "Raw Content (Base64)"
        return build_config

    def preview_media(self) -> str:
        if self.input_type == "File Path":
            if not self.file_path:
                return ""
            paths = [p.strip() for p in self.file_path.splitlines() if p.strip()]
            data_uris = []
            for fp in paths:
                data_uris.append(_file_to_data_uri(fp))
            return "\n".join(data_uris)
        if self.input_type == "Raw Content (Base64)":
            return self.raw_content or ""
        return self.media_url or ""
