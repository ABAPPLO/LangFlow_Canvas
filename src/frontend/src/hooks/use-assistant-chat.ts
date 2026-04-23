import { useCallback, useRef } from "react";
import { useAssistantChatStore } from "@/stores/assistantChatStore";
import useFlowStore from "@/stores/flowStore";

interface SSEMessage {
  event: string;
  data: string;
}

/**
 * Parse complete SSE messages from text. Only parses blocks ending with \n\n.
 * Returns parsed messages and the remaining unparsed text.
 */
function parseSSE(text: string): { messages: SSEMessage[]; remaining: string } {
  const messages: SSEMessage[] = [];

  // Find the last complete SSE block boundary
  const lastBoundary = text.lastIndexOf("\n\n");
  if (lastBoundary === -1) {
    return { messages: [], remaining: text };
  }

  const completeText = text.slice(0, lastBoundary);
  const remaining = text.slice(lastBoundary + 2);
  const blocks = completeText.split("\n\n").filter(Boolean);

  for (const block of blocks) {
    let event = "message";
    let data = "";
    for (const line of block.split("\n")) {
      if (line.startsWith("event: ")) {
        event = line.slice(7).trim();
      } else if (line.startsWith("data: ")) {
        data = line.slice(6);
      }
    }
    if (data) {
      messages.push({ event, data });
    }
  }

  return { messages, remaining };
}

const STREAM_TIMEOUT_MS = 300_000; // 5min timeout for no data (CLI startup + LLM inference)

export function useAssistantChat() {
  const abortControllerRef = useRef<AbortController | null>(null);

  const addMessage = useAssistantChatStore((s) => s.addMessage);
  const appendToLastAssistantMessage = useAssistantChatStore(
    (s) => s.appendToLastAssistantMessage,
  );
  const updateLastAssistantMessage = useAssistantChatStore(
    (s) => s.updateLastAssistantMessage,
  );
  const addToolCallToLastMessage = useAssistantChatStore(
    (s) => s.addToolCallToLastMessage,
  );
  const updateToolCall = useAssistantChatStore((s) => s.updateToolCall);
  const setLoading = useAssistantChatStore((s) => s.setLoading);
  const messages = useAssistantChatStore((s) => s.messages);
  const modelConfig = useAssistantChatStore((s) => s.modelConfig);
  const sessionId = useAssistantChatStore((s) => s.sessionId);
  const setSessionId = useAssistantChatStore((s) => s.setSessionId);

  const sendMessage = useCallback(
    async (message: string) => {
      const flowId = useFlowStore.getState().currentFlow?.id;
      if (!flowId) return;

      if (!modelConfig || !modelConfig.name) {
        addMessage({
          id: String(Date.now() + 1),
          role: "assistant",
          content: "Please select a model first.",
        });
        return;
      }

      // Add user message
      addMessage({
        id: String(Date.now()),
        role: "user",
        content: message,
      });

      // Add empty assistant message for streaming
      addMessage({
        id: String(Date.now() + 1),
        role: "assistant",
        content: "",
        isStreaming: true,
      });

      setLoading(true);

      // Abort previous request if any
      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;

      // Timeout timer: abort if no data received for STREAM_TIMEOUT_MS
      let timeoutId: ReturnType<typeof setTimeout> | null = null;
      const resetTimeout = () => {
        if (timeoutId) clearTimeout(timeoutId);
        timeoutId = setTimeout(() => {
          controller.abort();
        }, STREAM_TIMEOUT_MS);
      };
      resetTimeout();

      try {
        const response = await fetch("/api/v1/assistant/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            flow_id: flowId,
            message,
            selected_model: modelConfig,
            session_id: sessionId,
          }),
          signal: controller.signal,
          credentials: "include",
        });

        if (!response.ok) {
          const errorText = await response.text();
          updateLastAssistantMessage(
            `Error: ${response.status} - ${errorText}`,
          );
          return;
        }

        const reader = response.body?.getReader();
        if (!reader) {
          updateLastAssistantMessage("Error: No response stream");
          return;
        }

        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          resetTimeout();
          buffer += decoder.decode(value, { stream: true });

          const { messages: sseMessages, remaining } = parseSSE(buffer);
          buffer = remaining;

          for (const msg of sseMessages) {
            try {
              const data = JSON.parse(msg.data);
              handleSSEEvent(msg.event, data, {
                appendToLastAssistantMessage,
                addToolCallToLastMessage,
                updateToolCall,
                setSessionId,
              });
            } catch {
              // Ignore parse errors for incomplete data
            }
          }
        }

        // Process any remaining buffer
        if (buffer.trim()) {
          const { messages: sseMessages } = parseSSE(buffer + "\n\n");
          for (const msg of sseMessages) {
            try {
              const data = JSON.parse(msg.data);
              handleSSEEvent(msg.event, data, {
                appendToLastAssistantMessage,
                addToolCallToLastMessage,
                updateToolCall,
                setSessionId,
              });
            } catch {
              // Ignore
            }
          }
        }

        // Mark streaming complete
        const lastMsg = useAssistantChatStore.getState().messages;
        if (
          lastMsg.length > 0 &&
          lastMsg[lastMsg.length - 1].role === "assistant"
        ) {
          updateLastAssistantMessage(lastMsg[lastMsg.length - 1].content);
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          // Check if it was a timeout abort vs user abort
          const lastMsg = useAssistantChatStore.getState().messages;
          const lastAssistant = lastMsg[lastMsg.length - 1];
          if (lastAssistant?.isStreaming && !lastAssistant.content) {
            updateLastAssistantMessage(
              "Request timed out. The model may be unavailable or the API endpoint is unreachable.",
            );
          }
          return;
        }
        updateLastAssistantMessage(`Error: ${err}`);
      } finally {
        if (timeoutId) clearTimeout(timeoutId);
        setLoading(false);
        abortControllerRef.current = null;
      }
    },
    [
      messages,
      modelConfig,
      sessionId,
      addMessage,
      appendToLastAssistantMessage,
      updateLastAssistantMessage,
      addToolCallToLastMessage,
      updateToolCall,
      setSessionId,
      setLoading,
    ],
  );

  const stopStreaming = useCallback(() => {
    abortControllerRef.current?.abort();
    setLoading(false);
  }, [setLoading]);

  return { sendMessage, stopStreaming };
}

interface SSEHandlers {
  appendToLastAssistantMessage: (text: string) => void;
  addToolCallToLastMessage: (toolCall: {
    id: string;
    name: string;
    input: Record<string, unknown>;
    status: "running" | "completed" | "error";
    result?: string;
  }) => void;
  updateToolCall: (id: string, updates: Record<string, unknown>) => void;
  setSessionId: (id: string | null) => void;
}

function handleSSEEvent(
  event: string,
  data: Record<string, unknown>,
  handlers: SSEHandlers,
) {
  switch (event) {
    case "token": {
      const text = data.text as string;
      if (text) {
        handlers.appendToLastAssistantMessage(text);
      }
      break;
    }

    case "session_id": {
      const sid = data.session_id as string;
      if (sid) {
        handlers.setSessionId(sid);
      }
      break;
    }

    case "tool_start": {
      const toolId = String((data.name as string) + "_" + Date.now());
      handlers.addToolCallToLastMessage({
        id: toolId,
        name: data.name as string,
        input: (data.input as Record<string, unknown>) || {},
        status: "running",
      });
      break;
    }

    case "tool_result": {
      const name = data.name as string;
      const result = data.result as string;
      const isError = result?.startsWith("Error");

      // Update the last tool call with this name
      const store = useAssistantChatStore.getState();
      const lastMsg = store.messages[store.messages.length - 1];
      if (lastMsg?.toolCalls) {
        const tc = [...lastMsg.toolCalls]
          .reverse()
          .find((t) => t.name === name && t.status === "running");
        if (tc) {
          handlers.updateToolCall(tc.id, {
            status: isError ? "error" : "completed",
            result: result,
          });
        }
      }
      break;
    }

    case "canvas_update": {
      // Apply canvas updates to flowStore
      const nodes = data.nodes as Array<Record<string, unknown>>;
      const edges = data.edges as Array<Record<string, unknown>>;
      if (nodes) {
        const flowStore = useFlowStore.getState();
        const currentNodes = flowStore.nodes;

        const newNodeMap = new Map(nodes.map((n) => [n.id as string, n]));
        const mergedNodes = [
          ...currentNodes.filter((cn) => !newNodeMap.has(cn.id)),
          ...(nodes as never[]),
        ];

        // setNodes first — cleanEdges inside may remove edges with mismatched handles,
        // so we restore them afterwards via setEdges.
        flowStore.setNodes(mergedNodes as never);

        if (edges && edges.length > 0) {
          const currentEdges = flowStore.edges;
          const newEdgeMap = new Map(edges.map((e) => [e.id as string, e]));
          const mergedEdges = [
            ...currentEdges.filter((ce) => !newEdgeMap.has(ce.id)),
            ...(edges as never[]),
          ];
          flowStore.setEdges(mergedEdges as never);
        }
      }
      break;
    }

    case "error": {
      handlers.appendToLastAssistantMessage(
        `\n\nError: ${data.error as string}`,
      );
      break;
    }

    case "done": {
      // Streaming complete
      break;
    }
  }
}
