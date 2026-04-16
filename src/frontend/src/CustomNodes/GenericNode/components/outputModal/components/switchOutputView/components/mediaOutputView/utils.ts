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

const ALL_MEDIA_EXTENSIONS = [
  ...IMAGE_EXTENSIONS,
  ...VIDEO_EXTENSIONS,
  ...AUDIO_EXTENSIONS,
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
 * Check if a URL looks like a local file path (absolute Unix/Windows path).
 */
function isLocalFilePath(url: string): boolean {
  return url.startsWith("/") || /^[A-Za-z]:\\/.test(url);
}

/**
 * Rewrite a local file path to use the backend proxy endpoint.
 */
function rewriteLocalPath(url: string): string {
  return `/api/v1/files/local${url}`;
}

/**
 * Normalize a URL for deduplication purposes.
 * Strips common API path prefixes so that different representations
 * of the same file resolve to the same key.
 */
function normalizeForDedup(url: string): string {
  let normalized = url;
  // Strip common backend proxy prefixes
  for (const prefix of ["/api/v1/files/local", "/files/images", "/files/"]) {
    if (normalized.startsWith(prefix)) {
      normalized = normalized.slice(prefix.length);
      break;
    }
  }
  // Lowercase for case-insensitive comparison
  return normalized.toLowerCase();
}

const MAX_RECURSION_DEPTH = 5;

/**
 * Internal recursive extraction with a shared seen-set for deduplication.
 */
function _extractMediaUrls(
  value: unknown,
  depth: number,
  seen: Set<string>,
): MediaUrl[] {
  if (value == null || depth > MAX_RECURSION_DEPTH) return [];

  if (typeof value === "string") {
    const urls = value
      .split(/[\n\r]+/)
      .map((line) => line.trim())
      .filter(Boolean);

    const result: MediaUrl[] = [];
    for (const url of urls) {
      const mediaType = getMediaType(url);
      if (mediaType) {
        const finalUrl = isLocalFilePath(url) ? rewriteLocalPath(url) : url;
        const dedupKey = normalizeForDedup(finalUrl);
        if (!seen.has(dedupKey)) {
          seen.add(dedupKey);
          result.push({ url: finalUrl, type: mediaType });
        }
      }
    }
    return result;
  }

  if (Array.isArray(value)) {
    return value.flatMap((item) => _extractMediaUrls(item, depth + 1, seen));
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

    // 1. Extract from known URL keys (backward compatible)
    for (const key of urlKeys) {
      if (typeof dict[key] === "string") {
        const rawUrl = dict[key] as string;
        const mediaType = getMediaType(rawUrl);
        if (mediaType) {
          const finalUrl = isLocalFilePath(rawUrl)
            ? rewriteLocalPath(rawUrl)
            : rawUrl;
          const dedupKey = normalizeForDedup(finalUrl);
          if (!seen.has(dedupKey)) {
            seen.add(dedupKey);
            result.push({ url: finalUrl, type: mediaType });
          }
        }
      }
    }

    // 2. Recurse into all values for nested Table/JSON structures
    for (const val of Object.values(dict)) {
      if (typeof val === "object" && val !== null) {
        result.push(..._extractMediaUrls(val, depth + 1, seen));
      }
      // String values that aren't in urlKeys but look like media URLs
      // are handled by passing the string through the same recursion
    }

    return result;
  }

  return [];
}

/**
 * Extract media URLs from component output data.
 * Handles strings (including newline-separated), arrays, and dicts.
 * Deeply recurses into nested structures to find media URLs in Table/JSON data.
 * Local file paths are rewritten to use the backend proxy endpoint.
 * Deduplication is applied automatically across all sources.
 */
export function extractMediaUrls(value: unknown): MediaUrl[] {
  return _extractMediaUrls(value, 0, new Set());
}

/**
 * Deduplicate media URLs by their final (rewritten) URL string.
 */
export function dedupeMediaUrls(urls: MediaUrl[]): MediaUrl[] {
  const seen = new Set<string>();
  return urls.filter((item) => {
    if (seen.has(item.url)) return false;
    seen.add(item.url);
    return true;
  });
}

/**
 * Extract text content from component output data.
 * Recursively extracts text from strings, arrays, and dicts
 * (looking at common text keys like "text", "content", "message").
 */
export function extractTextContent(
  value: unknown,
  depth: number = 0,
): string[] {
  if (value == null || depth > MAX_RECURSION_DEPTH) return [];

  if (typeof value === "string") {
    const trimmed = value.trim();
    // Skip if it looks like a media URL (those are rendered separately)
    if (trimmed && !getMediaType(trimmed)) {
      return [trimmed];
    }
    return [];
  }

  if (Array.isArray(value)) {
    return value.flatMap((item) => extractTextContent(item, depth + 1));
  }

  if (typeof value === "object") {
    const dict = value as Record<string, unknown>;
    const textKeys = [
      "text",
      "content",
      "message",
      "output",
      "result",
      "description",
    ];
    const result: string[] = [];

    // Extract from known text keys
    for (const key of textKeys) {
      const val = dict[key];
      if (typeof val === "string" && val.trim()) {
        result.push(val.trim());
      }
    }

    // Recurse into nested objects/arrays for deeper text
    for (const val of Object.values(dict)) {
      if (typeof val === "object" && val !== null) {
        result.push(...extractTextContent(val, depth + 1));
      }
    }

    return result;
  }

  return [];
}

const MAX_TEXT_LENGTH = 300;
const MAX_TEXT_ITEMS = 5;

/**
 * Deduplicate and truncate text content for display.
 */
export function dedupeAndTruncateText(texts: string[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const text of texts) {
    // Deduplicate by first 100 chars
    const key = text.slice(0, 100);
    if (seen.has(key)) continue;
    seen.add(key);
    result.push(
      text.length > MAX_TEXT_LENGTH
        ? text.slice(0, MAX_TEXT_LENGTH) + "..."
        : text,
    );
    if (result.length >= MAX_TEXT_ITEMS) break;
  }
  return result;
}
