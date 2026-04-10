from lfx.custom import Component
from lfx.io import MultilineInput, Output


class MediaPreviewTestComponent(Component):
    display_name = "Media Preview"
    description = "Receives a media URL and displays it as an image, video, or audio preview in the node output."
    icon = "Play"
    name = "MediaPreview"

    inputs = [
        MultilineInput(
            name="media_url",
            display_name="Media URL",
            info="Media URL to preview. Supports image, video, and audio URLs. Multiple URLs can be separated by newlines.",
        ),
    ]
    outputs = [
        Output(display_name="Preview", name="preview", method="preview_media"),
    ]

    def preview_media(self) -> str:
        return self.media_url or ""
