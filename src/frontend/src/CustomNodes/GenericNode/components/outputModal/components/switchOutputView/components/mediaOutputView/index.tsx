import ForwardedIconComponent from "@/components/common/genericIconComponent";
import type { MediaType, MediaUrl } from "./utils";

interface MediaOutputViewProps {
  urls: MediaUrl[];
  onToggleRaw: () => void;
}

function MediaItem({ item }: { item: MediaUrl }) {
  const filename = getFilename(item.url);

  switch (item.type) {
    case "image":
      return (
        <div className="flex flex-col items-center gap-2">
          <img
            src={item.url}
            alt={filename}
            className="max-h-[400px] max-w-full rounded-lg border border-border object-contain"
          />
          {filename && (
            <span className="text-xs text-muted-foreground">{filename}</span>
          )}
        </div>
      );
    case "video":
      return (
        <div className="flex flex-col items-center gap-2">
          <video
            src={item.url}
            controls
            className="max-h-[400px] max-w-full rounded-lg border border-border"
          />
          {filename && (
            <span className="text-xs text-muted-foreground">{filename}</span>
          )}
        </div>
      );
    case "audio":
      return (
        <div className="flex w-full flex-col gap-2 rounded-lg border border-border p-4">
          <div className="flex items-center gap-2 text-sm text-muted-foreground">
            <ForwardedIconComponent name="Music" className="h-4 w-4" />
            {filename && <span>{filename}</span>}
          </div>
          <audio src={item.url} controls className="w-full" />
        </div>
      );
    default:
      return null;
  }
}

function getFilename(url: string): string {
  if (url.startsWith("data:")) return "";
  try {
    const pathname = new URL(url).pathname;
    const segments = pathname.split("/");
    return segments[segments.length - 1] ?? "";
  } catch {
    const segments = url.split("/");
    return segments[segments.length - 1] ?? "";
  }
}

export default function MediaOutputView({
  urls,
  onToggleRaw,
}: MediaOutputViewProps) {
  if (urls.length === 0) {
    return <div className="text-muted-foreground">No media to display</div>;
  }

  // Group by type for organized display
  const images = urls.filter((u) => u.type === "image");
  const videos = urls.filter((u) => u.type === "video");
  const audios = urls.filter((u) => u.type === "audio");

  return (
    <div className="flex flex-col gap-4">
      <div className="flex justify-end">
        <button
          onClick={onToggleRaw}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
        >
          <ForwardedIconComponent name="FileText" className="h-3 w-3" />
          View Raw
        </button>
      </div>

      <div className="flex flex-col gap-4">
        {images.length > 0 && (
          <MediaSection title="Images" icon="Image" items={images} />
        )}
        {videos.length > 0 && (
          <MediaSection title="Videos" icon="Video" items={videos} />
        )}
        {audios.length > 0 && (
          <MediaSection title="Audio" icon="Music" items={audios} />
        )}
      </div>
    </div>
  );
}

function MediaSection({
  title,
  icon,
  items,
}: {
  title: string;
  icon: string;
  items: MediaUrl[];
}) {
  return (
    <div className="flex flex-col gap-2">
      {items.length > 1 && (
        <div className="flex items-center gap-1.5 text-sm font-medium text-muted-foreground">
          <ForwardedIconComponent name={icon} className="h-4 w-4" />
          {title} ({items.length})
        </div>
      )}
      <div className="flex flex-col gap-3">
        {items.map((item, index) => (
          <MediaItem key={index} item={item} />
        ))}
      </div>
    </div>
  );
}

export type { MediaType, MediaUrl };
