import { useState } from "react";
import ForwardedIconComponent from "@/components/common/genericIconComponent";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  useAddCustomModel,
  useRemoveCustomModel,
} from "@/controllers/API/queries/models/use-custom-models";
import { useGetEnabledModels } from "@/controllers/API/queries/models/use-get-enabled-models";
import { useGetModelProviders } from "@/controllers/API/queries/models/use-get-model-providers";

import { Model } from "@/modals/modelProviderModal/components/types";
import { cn } from "@/utils/utils";

export interface ModelProviderSelectionProps {
  availableModels: Model[];
  onModelToggle: (modelName: string, enabled: boolean) => void;
  modelType: "llm" | "embeddings" | "all";
  providerName?: string;
  isEnabledModel?: boolean;
}

/**
 * Displays lists of LLM and embedding models with toggle switches.
 * Allows users to enable/disable individual models for a provider.
 * Also allows adding custom model names.
 */
const ModelSelection = ({
  modelType = "llm",
  availableModels,
  onModelToggle,
  providerName,
  isEnabledModel,
}: ModelProviderSelectionProps) => {
  const { data: enabledModelsData } = useGetEnabledModels();
  const addCustomModel = useAddCustomModel();
  const removeCustomModel = useRemoveCustomModel();
  const { refetch: refetchProviders } = useGetModelProviders();
  const [newModelName, setNewModelName] = useState("");

  const isModelEnabled = (modelName: string): boolean => {
    if (!providerName || !enabledModelsData?.enabled_models) return false;
    return enabledModelsData.enabled_models[providerName]?.[modelName] ?? false;
  };

  const handleAddCustomModel = () => {
    const trimmed = newModelName.trim();
    if (!trimmed || !providerName) return;
    addCustomModel.mutate(
      { provider: providerName, model_name: trimmed },
      {
        onSuccess: () => {
          setNewModelName("");
          refetchProviders();
        },
      },
    );
  };

  const handleRemoveCustomModel = (modelName: string) => {
    if (!providerName) return;
    removeCustomModel.mutate(
      { provider: providerName, model_name: modelName },
      {
        onSuccess: () => {
          refetchProviders();
        },
      },
    );
  };

  const llmModels = availableModels.filter(
    (model) =>
      model.metadata?.model_type === "llm" || !model.metadata?.is_custom,
  );
  const embeddingModels = availableModels.filter(
    (model) => model.metadata?.model_type === "embeddings",
  );

  const renderModelSection = (
    title: string,
    models: Model[],
    testIdPrefix: string,
  ) => {
    if (models.length === 0) return null;
    return (
      <div data-testid={`${testIdPrefix}-models-section`}>
        <div className="text-[13px] font-semibold text-muted-foreground">
          {title}
        </div>
        <div className="flex flex-col gap-2 pt-4">
          {models.map((model) => (
            <div
              key={model.model_name}
              className="flex flex-row items-center justify-between h-[24px]"
            >
              <div className="flex flex-row items-center gap-2">
                <ForwardedIconComponent
                  name={model.metadata?.icon || "Bot"}
                  className={cn("w-5 h-5", { grayscale: !isEnabledModel })}
                />
                <span
                  className={cn("text-sm", {
                    "text-muted-foreground": !isEnabledModel,
                  })}
                >
                  {model.model_name}
                </span>
                {model.metadata?.is_custom && (
                  <span className="text-[10px] font-medium px-1.5 py-0.5 rounded bg-primary/10 text-primary">
                    Custom
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                {model.metadata?.is_custom && isEnabledModel && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="h-5 w-5 p-0 text-muted-foreground hover:text-destructive"
                    onClick={() => handleRemoveCustomModel(model.model_name)}
                    data-testid={`remove-custom-${model.model_name}`}
                  >
                    <ForwardedIconComponent name="X" className="w-3 h-3" />
                  </Button>
                )}
                {isEnabledModel && !model.metadata?.is_custom && (
                  <Switch
                    checked={isModelEnabled(model.model_name)}
                    onCheckedChange={(checked) =>
                      onModelToggle(model.model_name, checked)
                    }
                    data-testid={`${testIdPrefix}-toggle-${model.model_name}`}
                  />
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const isOllama = providerName?.toLowerCase() === "ollama";
  const noModelsAvailable =
    (modelType === "llm" && llmModels.length === 0) ||
    (modelType === "embeddings" && embeddingModels.length === 0) ||
    (modelType === "all" && availableModels.length === 0);

  // Show "Add custom model" input for configured providers (except Ollama which fetches live)
  const showAddModel = isEnabledModel && !isOllama && providerName;

  return (
    <div data-testid="model-provider-selection" className="flex flex-col gap-6">
      {isOllama && noModelsAvailable ? (
        <div className="flex flex-col items-center justify-center p-8 text-center border border-dashed rounded-lg bg-muted/30">
          <ForwardedIconComponent
            name="Info"
            className="w-10 h-10 mb-4 text-muted-foreground"
          />
          <h3 className="mb-2 text-sm font-semibold text-foreground">
            No models available
          </h3>
          <p className="max-w-[300px] text-xs text-muted-foreground leading-relaxed">
            It looks like you don't have any
            {modelType === "llm"
              ? " language"
              : modelType === "embeddings"
                ? " embedding"
                : ""}{" "}
            models installed for Ollama. Please pull the models you want to use.
          </p>
          <a
            href="https://ollama.com/library"
            target="_blank"
            rel="noreferrer"
            className="mt-6 text-xs font-medium text-primary underline underline-offset-4 hover:opacity-80 transition-opacity"
          >
            Check Ollama Library
          </a>
        </div>
      ) : (
        <>
          {modelType === "all" ? (
            <>
              {renderModelSection("Language Models", llmModels, "llm")}
              {renderModelSection(
                "Embedding Models",
                embeddingModels,
                "embeddings",
              )}
            </>
          ) : modelType === "llm" ? (
            renderModelSection("Language Models", llmModels, "llm")
          ) : (
            renderModelSection(
              "Embedding Models",
              embeddingModels,
              "embeddings",
            )
          )}
          {showAddModel && (
            <div className="flex flex-col gap-2 pt-2 border-t">
              <div className="text-[13px] font-semibold text-muted-foreground">
                Add Custom Model
              </div>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={newModelName}
                  onChange={(e) => setNewModelName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleAddCustomModel();
                  }}
                  placeholder="Enter model name"
                  className="flex-1 h-8 px-2 text-sm border rounded-md bg-background"
                  data-testid="custom-model-input"
                />
                <Button
                  size="sm"
                  onClick={handleAddCustomModel}
                  disabled={!newModelName.trim() || addCustomModel.isPending}
                  data-testid="add-custom-model-btn"
                >
                  Add
                </Button>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ModelSelection;
