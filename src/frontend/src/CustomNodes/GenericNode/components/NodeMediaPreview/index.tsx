import { memo, useMemo } from "react";
import ForwardedIconComponent from "@/components/common/genericIconComponent";
import useFlowStore from "@/stores/flowStore";
import type { NodeDataType } from "@/types/flow";
import { extractMediaUrls } from "../outputModal/components/switchOutputView/components/mediaOutputView/utils";

function NodeMediaPreview({ data }: { data: NodeDataType }) {
  const flowPool = useFlowStore((state) => state.flowPool);

  const mediaUrls = useMemo(() => {
    const flowPoolNode = (flowPool[data.id] ?? [])[
      (flowPool[data.id]?.length ?? 1) - 1
    ];
    if (!flowPoolNode?.data?.outputs) return [];

    // Collect results from all outputs
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

    return extractMediaUrls(allMessages);
  }, [flowPool, data.id]);

  if (mediaUrls.length === 0) return null;

  return (
    <div className="border-t px-3 py-2">
      <div className="flex flex-col gap-2">
        {mediaUrls.map((item, index) => {
          if (item.type === "image") {
            return (
              <img
                key={index}
                src={item.url}
                alt={`preview-${index}`}
                className="max-h-[200px] max-w-full rounded-md object-contain"
              />
            );
          }
          if (item.type === "video") {
            return (
              <video
                key={index}
                src={item.url}
                controls
                className="max-h-[200px] max-w-full rounded-md"
              />
            );
          }
          if (item.type === "audio") {
            return (
              <div
                key={index}
                className="flex items-center gap-2 rounded-md border p-2"
              >
                <ForwardedIconComponent
                  name="Music"
                  className="h-4 w-4 text-muted-foreground"
                />
                <audio src={item.url} controls className="h-8 w-full" />
              </div>
            );
          }
          return null;
        })}
      </div>
    </div>
  );
}

export default memo(NodeMediaPreview);
