import { useAssistantChat } from "@/hooks/use-assistant-chat";
import { useAssistantChatStore } from "@/stores/assistantChatStore";
import { AssistantChatInput } from "./AssistantChatInput";
import { AssistantMessageList } from "./AssistantMessageList";
import { AssistantModelSelect } from "./AssistantModelSelect";

export function AssistantChatPanel() {
  const messages = useAssistantChatStore((s) => s.messages);
  const isLoading = useAssistantChatStore((s) => s.isLoading);
  const clearMessages = useAssistantChatStore((s) => s.clearMessages);
  const { sendMessage, stopStreaming } = useAssistantChat();

  return (
    <div className="flex h-full flex-col bg-background">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <h3 className="text-sm font-semibold text-foreground">AI Assistant</h3>
        <div className="flex-1" />
        <AssistantModelSelect />
        <button
          type="button"
          onClick={clearMessages}
          className="text-xs text-muted-foreground hover:text-foreground"
        >
          Clear
        </button>
      </div>

      {/* Messages */}
      <AssistantMessageList messages={messages} />

      {/* Input */}
      <AssistantChatInput
        onSend={sendMessage}
        onStop={stopStreaming}
        isLoading={isLoading}
      />
    </div>
  );
}
