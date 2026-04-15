import { useEffect, useRef } from "react";
import type { AssistantMessage as AssistantMessageType } from "@/stores/assistantChatStore";
import { AssistantMessage } from "./AssistantMessage";

export function AssistantMessageList({
  messages,
}: {
  messages: AssistantMessageType[];
}) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center p-4">
        <div className="text-center text-muted-foreground">
          <p className="text-lg font-medium">AI Assistant</p>
          <p className="mt-1 text-sm">
            Describe the workflow you want to build, and I'll create it for you.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 space-y-3 overflow-y-auto p-3">
      {messages.map((msg) => (
        <AssistantMessage key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
