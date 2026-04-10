from lfx.custom import Component
from lfx.io import Output, TableInput


class MediaPreviewTestComponent(Component):
    display_name = "Media Preview"
    description = "Receives media URLs and displays them as image, video, or audio previews below the node."
    icon = "Play"
    name = "MediaPreview"

    inputs = [
        TableInput(
            name="media_list",
            display_name="Media List",
            info="List of media items to preview. Each row should have a 'url' field. The 'type' field (image/video/audio) is auto-detected from the URL extension but can be overridden.",
        ),
    ]
    outputs = [
        Output(display_name="Preview", name="preview", method="preview_media"),
    ]

    def preview_media(self) -> list[dict]:
        if not self.media_list:
            return []
        # Normalize: ensure each item has a url field
        result = []
        for item in self.media_list:
            if isinstance(item, dict):
                url = item.get("url") or item.get("media_url") or item.get("src") or item.get("path") or ""
                if url:
                    result.append({"url": str(url), **{k: v for k, v in item.items() if k != "url"}})
            elif isinstance(item, str) and item.strip():
                result.append({"url": item.strip()})
        return result
