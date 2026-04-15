import { useAssistantChatStore } from "@/stores/assistantChatStore";
import { AssistantChatPanel } from "./AssistantChatPanel";

export function AssistantWidget() {
  const isOpen = useAssistantChatStore((s) => s.isOpen);
  const toggleOpen = useAssistantChatStore((s) => s.toggleOpen);
  const isExpanded = useAssistantChatStore((s) => s.isExpanded);
  const setExpanded = useAssistantChatStore((s) => s.setExpanded);

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col items-end">
      {/* Chat Panel */}
      {isOpen && (
        <div
          className={`mb-2 overflow-hidden rounded-xl border border-border bg-background shadow-2xl transition-all duration-300 ${
            isExpanded ? "h-[600px] w-[450px]" : "h-[400px] w-[360px]"
          }`}
        >
          <AssistantChatPanel />
        </div>
      )}

      {/* Control buttons */}
      <div className="flex items-center gap-2">
        {isOpen && (
          <button
            type="button"
            onClick={() => setExpanded(!isExpanded)}
            className="flex h-9 w-9 items-center justify-center rounded-full border border-border bg-background text-muted-foreground shadow-lg transition-colors hover:bg-muted hover:text-foreground"
            title={isExpanded ? "Minimize" : "Expand"}
          >
            {isExpanded ? (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
              </svg>
            ) : (
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
              </svg>
            )}
          </button>
        )}

        {/* Toggle button */}
        <button
          type="button"
          onClick={toggleOpen}
          className="flex h-12 w-12 items-center justify-center rounded-full bg-primary text-primary-foreground shadow-lg transition-transform hover:scale-105 active:scale-95"
          title={isOpen ? "Close Assistant" : "Open AI Assistant"}
        >
          {isOpen ? (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          ) : (
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2Z" />
              <path d="M12 16v-4" />
              <path d="M12 8h.01" />
            </svg>
          )}
        </button>
      </div>
    </div>
  );
}
