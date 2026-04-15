import { useState } from "react";
import type { ToolCall } from "@/stores/assistantChatStore";

export function ToolCallDisplay({ toolCall }: { toolCall: ToolCall }) {
  const [expanded, setExpanded] = useState(false);

  const statusIcon =
    toolCall.status === "running" ? (
      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-blue-400 border-t-transparent" />
    ) : toolCall.status === "completed" ? (
      <span className="inline-block h-3 w-3 rounded-full bg-green-400" />
    ) : (
      <span className="inline-block h-3 w-3 rounded-full bg-red-400" />
    );

  return (
    <div className="my-1 rounded border border-border bg-muted/50 text-xs">
      <button
        type="button"
        className="flex w-full items-center gap-2 px-2 py-1.5 text-left hover:bg-muted"
        onClick={() => setExpanded(!expanded)}
      >
        {statusIcon}
        <span className="font-medium text-foreground">
          {toolCall.name === "list_components"
            ? "List Components"
            : toolCall.name === "build_flow"
              ? "Build Flow"
              : toolCall.name === "run_flow"
                ? "Run Flow"
                : toolCall.name}
        </span>
        <span className="ml-auto text-muted-foreground">
          {toolCall.status === "running" ? "Running..." : ""}
        </span>
      </button>
      {expanded && (
        <div className="border-t border-border px-2 py-1.5">
          {toolCall.result && (
            <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words text-muted-foreground">
              {toolCall.result}
            </pre>
          )}
          {!toolCall.result && toolCall.status === "running" && (
            <p className="text-muted-foreground">Executing...</p>
          )}
        </div>
      )}
    </div>
  );
}
