import { memo, useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import ForwardedIconComponent from "@/components/common/genericIconComponent";
import useFlowStore from "@/stores/flowStore";
import type { NodeDataType } from "@/types/flow";
import type { MediaUrl } from "../outputModal/components/switchOutputView/components/mediaOutputView/utils";
import {
  extractMediaUrls,
  dedupeMediaUrls,
  extractTextContent,
  dedupeAndTruncateText,
} from "../outputModal/components/switchOutputView/components/mediaOutputView/utils";

const COLLAPSE_THRESHOLD = 4;

function MediaGrid({ items }: { items: MediaUrl[] }) {
  const images = items.filter((u) => u.type === "image");
  const videos = items.filter((u) => u.type === "video");
  const audios = items.filter((u) => u.type === "audio");

  return (
    <div className="flex flex-col gap-2">
      {images.length > 0 && (
        <div className="grid grid-cols-2 gap-1.5">
          {images.map((item, i) => (
            <img
              key={`img-${i}`}
              src={item.url}
              alt={`preview-${i}`}
              className="max-h-[140px] w-full rounded-md object-contain"
            />
          ))}
        </div>
      )}
      {videos.map((item, i) => (
        <video
          key={`vid-${i}`}
          src={item.url}
          controls
          className="max-h-[160px] max-w-full rounded-md"
        />
      ))}
      {audios.map((item, i) => (
        <div
          key={`aud-${i}`}
          className="flex items-center gap-2 rounded-md border p-1.5"
        >
          <ForwardedIconComponent
            name="Music"
            className="h-3.5 w-3.5 shrink-0 text-muted-foreground"
          />
          <audio src={item.url} controls className="h-7 w-full" />
        </div>
      ))}
    </div>
  );
}

function TextPreview({ texts }: { texts: string[] }) {
  const [expanded, setExpanded] = useState(false);
  const visibleTexts = expanded ? texts : texts.slice(0, 2);

  return (
    <div className="flex flex-col gap-1">
      {visibleTexts.map((text, i) => (
        <div
          key={`txt-${i}`}
          className="rounded-md border bg-muted/30 px-2 py-1 text-xs text-muted-foreground whitespace-pre-wrap break-words"
        >
          {text}
        </div>
      ))}
      {texts.length > 2 && !expanded && (
        <button
          onClick={() => setExpanded(true)}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          +{texts.length - 2} more
        </button>
      )}
    </div>
  );
}

function NodeMediaPreview({ data }: { data: NodeDataType }) {
  const flowPool = useFlowStore((state) => state.flowPool);
  const [expanded, setExpanded] = useState(false);
  const { t } = useTranslation("components");

  const { mediaUrls, textItems } = useMemo(() => {
    const flowPoolNode = (flowPool[data.id] ?? [])[
      (flowPool[data.id]?.length ?? 1) - 1
    ];
    if (!flowPoolNode?.data?.outputs) return { mediaUrls: [], textItems: [] };

    const allMessages: unknown[] = [];
    for (const output of Object.values(
      flowPoolNode.data.outputs as Record<string, { message?: unknown }>,
    )) {
      let msg = output.message;
      if (msg && typeof msg === "object" && "raw" in (msg as object)) {
        msg = (msg as { raw: unknown }).raw;
      }
      if (msg != null && msg !== "") {
        allMessages.push(msg);
      }
    }

    const urls = dedupeMediaUrls(extractMediaUrls(allMessages));
    const texts = dedupeAndTruncateText(
      extractTextContent(allMessages).filter(
        (t) => !urls.some((u) => t.includes(u.url)),
      ),
    );

    return { mediaUrls: urls, textItems: texts };
  }, [flowPool, data.id]);

  if (mediaUrls.length === 0 && textItems.length === 0) return null;

  const needsCollapse = mediaUrls.length > COLLAPSE_THRESHOLD;
  const visibleItems = expanded
    ? mediaUrls
    : mediaUrls.slice(0, COLLAPSE_THRESHOLD);

  return (
    <div className="border-t px-3 py-2">
      {textItems.length > 0 && <TextPreview texts={textItems} />}
      {mediaUrls.length > 0 && (
        <>
          <MediaGrid items={visibleItems} />
          {needsCollapse && (
            <button
              onClick={() => setExpanded(!expanded)}
              className="mt-1.5 w-full rounded-md px-2 py-1 text-xs text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              {expanded
                ? t("mediaPreview.collapse")
                : t("mediaPreview.expandAll", { count: mediaUrls.length })}
            </button>
          )}
        </>
      )}
    </div>
  );
}

export default memo(NodeMediaPreview);
