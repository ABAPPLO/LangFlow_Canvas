import { useEffect, useState } from "react";
import { useAssistantChatStore } from "@/stores/assistantChatStore";

interface ModelOption {
  name: string;
  icon: string;
  provider: string;
  category?: string;
  metadata?: Record<string, unknown>;
}

export function AssistantModelSelect() {
  const modelConfig = useAssistantChatStore((s) => s.modelConfig);
  const setModelConfig = useAssistantChatStore((s) => s.setModelConfig);
  const [options, setOptions] = useState<ModelOption[]>([]);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    loadOptions();
  }, []);

  async function loadOptions() {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/model_options/language", {
        credentials: "include",
      });
      if (res.ok) {
        const data = await res.json();
        const all = Array.isArray(data) ? data : [];
        setOptions(all);
      }
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }

  const selectedName = (modelConfig as ModelOption)?.name as string;
  const selectedProvider = (modelConfig as ModelOption)?.provider as string;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => {
          setOpen(!open);
          if (!open && options.length === 0) loadOptions();
        }}
        className="flex items-center gap-1.5 rounded-md border border-border bg-background px-2 py-1 text-xs text-foreground hover:bg-muted"
        title="Select model"
      >
        {selectedName ? (
          <>
            <span className="max-w-[140px] truncate font-medium">
              {selectedName}
            </span>
            <span className="text-muted-foreground">({selectedProvider})</span>
          </>
        ) : loading ? (
          <span className="text-muted-foreground">Loading...</span>
        ) : (
          <span className="text-muted-foreground">Select Model</span>
        )}
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="12"
          height="12"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="ml-auto text-muted-foreground"
        >
          <path d="m6 9 6 6 6-6" />
        </svg>
      </button>

      {open && (
        <div className="absolute left-0 top-full z-50 mt-1 max-h-60 w-64 overflow-y-auto rounded-md border border-border bg-background shadow-lg">
          {loading ? (
            <div className="px-3 py-2 text-xs text-muted-foreground">
              Loading models...
            </div>
          ) : options.length === 0 ? (
            <div className="px-3 py-2 text-xs text-muted-foreground">
              No models found. Configure a provider in Settings.
            </div>
          ) : (
            options.map((opt) => (
              <button
                key={`${opt.provider}-${opt.name}`}
                type="button"
                className={`flex w-full items-center gap-2 px-3 py-1.5 text-left text-xs hover:bg-muted ${
                  selectedName === opt.name && selectedProvider === opt.provider
                    ? "bg-muted font-medium"
                    : ""
                }`}
                onClick={() => {
                  setModelConfig(opt as Record<string, unknown>);
                  setOpen(false);
                }}
              >
                <span className="flex-1 truncate">{opt.name}</span>
                <span className="shrink-0 text-muted-foreground">
                  {opt.provider}
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}
