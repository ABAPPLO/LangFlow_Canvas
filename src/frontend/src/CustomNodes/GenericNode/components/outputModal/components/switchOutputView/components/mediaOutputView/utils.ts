export type MediaType = "image" | "video" | "audio";

export interface MediaUrl {
  url: string;
  type: MediaType;
}

const IMAGE_EXTENSIONS = [
  ".png",
  ".jpg",
  ".jpeg",
  ".gif",
  ".webp",
  ".bmp",
  ".svg",
];

const VIDEO_EXTENSIONS = [".mp4", ".webm", ".ogg", ".mov", ".avi", ".mkv"];

const AUDIO_EXTENSIONS = [
  ".mp3",
  ".wav",
  ".ogg",
  ".aac",
  ".flac",
  ".m4a",
  ".wma",
];

/**
 * Determine the media type of a URL based on its extension or data URI prefix.
 */
export function getMediaType(url: string): MediaType | null {
  const trimmed = url.trim().toLowerCase();

  // Check data URIs
  if (trimmed.startsWith("data:")) {
    if (trimmed.startsWith("data:image/")) return "image";
    if (trimmed.startsWith("data:video/")) return "video";
    if (trimmed.startsWith("data:audio/")) return "audio";
    return null;
  }

  // Extract the path portion (strip query params and fragments)
  const pathOnly = trimmed.split("?")[0]?.split("#")[0] ?? trimmed;

  for (const ext of IMAGE_EXTENSIONS) {
    if (pathOnly.endsWith(ext)) return "image";
  }
  for (const ext of VIDEO_EXTENSIONS) {
    if (pathOnly.endsWith(ext)) return "video";
  }
  for (const ext of AUDIO_EXTENSIONS) {
    if (pathOnly.endsWith(ext)) return "audio";
  }

  return null;
}

/**
 * Extract media URLs from component output data.
 * Handles strings (including newline-separated), arrays, and dicts with common URL fields.
 */
export function extractMediaUrls(value: unknown): MediaUrl[] {
  if (value == null) return [];

  if (typeof value === "string") {
    const urls = value
      .split(/[\n\r]+/)
      .map((line) => line.trim())
      .filter(Boolean);

    const result: MediaUrl[] = [];
    for (const url of urls) {
      const mediaType = getMediaType(url);
      if (mediaType) {
        result.push({ url, type: mediaType });
      }
    }
    return result;
  }

  if (Array.isArray(value)) {
    return value.flatMap((item) => extractMediaUrls(item));
  }

  if (typeof value === "object") {
    const dict = value as Record<string, unknown>;
    const urlKeys = [
      "url",
      "image_url",
      "src",
      "path",
      "file_path",
      "image",
      "video",
      "audio",
      "media_url",
    ];

    const result: MediaUrl[] = [];
    for (const key of urlKeys) {
      if (typeof dict[key] === "string") {
        const mediaType = getMediaType(dict[key] as string);
        if (mediaType) {
          result.push({ url: dict[key] as string, type: mediaType });
        }
      }
    }
    return result;
  }

  return [];
}
