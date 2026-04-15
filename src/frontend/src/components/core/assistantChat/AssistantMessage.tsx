import type { AssistantMessage as AssistantMessageType } from "@/stores/assistantChatStore";
import { ToolCallDisplay } from "./ToolCallDisplay";

export function AssistantMessage({
  message,
}: {
  message: AssistantMessageType;
}) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground">
          {message.content}
        </div>
      </div>
    );
  }

  return (
    <div className="flex justify-start">
      <div className="max-w-[90%] space-y-1">
        <div className="rounded-2xl rounded-bl-sm bg-muted px-3 py-2 text-sm text-foreground">
          {message.content || (message.isStreaming ? "" : "")}
          {message.isStreaming && !message.content && (
            <span className="inline-block h-4 w-1 animate-pulse bg-foreground/50" />
          )}
        </div>
        {message.toolCalls?.map((tc) => (
          <ToolCallDisplay key={tc.id} toolCall={tc} />
        ))}
      </div>
    </div>
  );
}
