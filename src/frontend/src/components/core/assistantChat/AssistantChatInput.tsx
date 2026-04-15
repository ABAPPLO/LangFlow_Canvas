import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";

interface AssistantChatInputProps {
  onSend: (message: string) => void;
  onStop: () => void;
  isLoading: boolean;
}

export function AssistantChatInput({
  onSend,
  onStop,
  isLoading,
}: AssistantChatInputProps) {
  const [input, setInput] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  }, [input]);

  const handleSubmit = () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;
    onSend(trimmed);
    setInput("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t border-border p-3">
      <div className="flex items-end gap-2 rounded-xl border border-border bg-background px-3 py-2">
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your workflow..."
          className="max-h-[120px] min-h-[24px] flex-1 resize-none bg-transparent text-sm outline-none placeholder:text-muted-foreground"
          rows={1}
          disabled={isLoading}
        />
        {isLoading ? (
          <Button
            variant="ghost"
            size="sm"
            onClick={onStop}
            className="h-7 shrink-0 px-2 text-xs"
          >
            Stop
          </Button>
        ) : (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleSubmit}
            disabled={!input.trim()}
            className="h-7 shrink-0 px-2 text-xs"
          >
            Send
          </Button>
        )}
      </div>
    </div>
  );
}
