from lfx.custom import Component
from lfx.io import DropdownInput, MultilineInput, Output


class MediaPreviewSimpleComponent(Component):
    display_name = "Media Preview Simple"
    description = "Receives media URLs or raw base64 content and displays previews below the node."
    icon = "Play"
    name = "MediaPreviewSimple"

    inputs = [
        DropdownInput(
            name="input_type",
            display_name="Input Type",
            options=["URL", "Raw Content (Base64)"],
            value="URL",
            info="Select the input format: URL for web links, Raw Content for base64 data URIs.",
            real_time_refresh=True,
        ),
        MultilineInput(
            name="media_url",
            display_name="Media URL",
            info="Media URL to preview. Multiple URLs can be separated by newlines.",
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
            build_config["raw_content"]["show"] = field_value == "Raw Content (Base64)"
        return build_config

    def preview_media(self) -> str:
        if self.input_type == "Raw Content (Base64)":
            return self.raw_content or ""
        return self.media_url or ""
