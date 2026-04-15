import { create } from "zustand";

export interface ToolCall {
  id: string;
  name: string;
  input: Record<string, unknown>;
  status: "running" | "completed" | "error";
  result?: string;
}

export interface AssistantMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  toolCalls?: ToolCall[];
  isStreaming?: boolean;
}

interface AssistantChatState {
  isOpen: boolean;
  isExpanded: boolean;
  messages: AssistantMessage[];
  isLoading: boolean;
  modelConfig: Record<string, unknown> | null;

  toggleOpen: () => void;
  setOpen: (open: boolean) => void;
  setExpanded: (expanded: boolean) => void;
  addMessage: (message: AssistantMessage) => void;
  updateLastAssistantMessage: (text: string) => void;
  appendToLastAssistantMessage: (text: string) => void;
  addToolCallToLastMessage: (toolCall: ToolCall) => void;
  updateToolCall: (id: string, updates: Partial<ToolCall>) => void;
  setLoading: (loading: boolean) => void;
  clearMessages: () => void;
  setModelConfig: (config: Record<string, unknown> | null) => void;
}

let messageCounter = 0;

export const useAssistantChatStore = create<AssistantChatState>((set) => ({
  isOpen: false,
  isExpanded: false,
  messages: [],
  isLoading: false,
  modelConfig: null,

  toggleOpen: () => set((state) => ({ isOpen: !state.isOpen })),

  setOpen: (open) => set({ isOpen: open }),

  setExpanded: (expanded) => set({ isExpanded: expanded }),

  addMessage: (message) =>
    set((state) => ({
      messages: [
        ...state.messages,
        { ...message, id: message.id || String(++messageCounter) },
      ],
    })),

  updateLastAssistantMessage: (text) =>
    set((state) => {
      const messages = [...state.messages];
      const lastIdx = messages.length - 1;
      if (lastIdx >= 0 && messages[lastIdx].role === "assistant") {
        messages[lastIdx] = {
          ...messages[lastIdx],
          content: text,
          isStreaming: false,
        };
      }
      return { messages };
    }),

  appendToLastAssistantMessage: (text) =>
    set((state) => {
      const messages = [...state.messages];
      const lastIdx = messages.length - 1;
      if (lastIdx >= 0 && messages[lastIdx].role === "assistant") {
        messages[lastIdx] = {
          ...messages[lastIdx],
          content: messages[lastIdx].content + text,
          isStreaming: true,
        };
      }
      return { messages };
    }),

  addToolCallToLastMessage: (toolCall) =>
    set((state) => {
      const messages = [...state.messages];
      const lastIdx = messages.length - 1;
      if (lastIdx >= 0 && messages[lastIdx].role === "assistant") {
        const existing = messages[lastIdx].toolCalls || [];
        messages[lastIdx] = {
          ...messages[lastIdx],
          toolCalls: [...existing, toolCall],
        };
      }
      return { messages };
    }),

  updateToolCall: (id, updates) =>
    set((state) => {
      const messages = [...state.messages];
      const lastIdx = messages.length - 1;
      if (lastIdx >= 0 && messages[lastIdx].role === "assistant") {
        const toolCalls = (messages[lastIdx].toolCalls || []).map((tc) =>
          tc.id === id ? { ...tc, ...updates } : tc,
        );
        messages[lastIdx] = { ...messages[lastIdx], toolCalls };
      }
      return { messages };
    }),

  setLoading: (loading) => set({ isLoading: loading }),

  clearMessages: () => set({ messages: [] }),

  setModelConfig: (config) => set({ modelConfig: config }),
}));
