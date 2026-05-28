import type { ReactFlowJsonObject } from "@xyflow/react";
import { useGetFlow } from "@/controllers/API/queries/flows/use-get-flow";
import { usePatchUpdateFlow } from "@/controllers/API/queries/flows/use-patch-update-flow";
import useAlertStore from "@/stores/alertStore";
import useFlowStore from "@/stores/flowStore";
import useFlowsManagerStore from "@/stores/flowsManagerStore";
import type { AllNodeType, EdgeType, FlowType } from "@/types/flow";
import { customStringify } from "@/utils/reactflowUtils";

const useSaveFlow = () => {
  const setFlows = useFlowsManagerStore((state) => state.setFlows);
  const setErrorData = useAlertStore((state) => state.setErrorData);
  const setSaveLoading = useFlowsManagerStore((state) => state.setSaveLoading);
  const setCurrentFlow = useFlowStore((state) => state.setCurrentFlow);

  const { mutate: getFlow } = useGetFlow();
  const { mutate } = usePatchUpdateFlow();

  const saveFlow = async (flow?: FlowType): Promise<void> => {
    const currentFlow = useFlowStore.getState().currentFlow;
    const currentSavedFlow = useFlowsManagerStore.getState().currentFlow;
    if (
      customStringify(flow || currentFlow) !== customStringify(currentSavedFlow)
    ) {
      setSaveLoading(true);

      const flowData = currentFlow?.data;
      const nodes = useFlowStore.getState().nodes;
      const edges = useFlowStore.getState().edges;
      const reactFlowInstance = useFlowStore.getState().reactFlowInstance;

      return new Promise<void>((resolve, reject) => {
        if (currentFlow) {
          flow = flow || {
            ...currentFlow,
            data: {
              ...flowData,
              nodes,
              edges,
              viewport: reactFlowInstance?.getViewport() ?? {
                zoom: 1,
                x: 0,
                y: 0,
              },
            },
          };
        }

        if (flow) {
          if (!flow?.data) {
            getFlow(
              { id: flow!.id },
              {
                onSuccess: (flowResponse) => {
                  flow!.data = flowResponse.data as ReactFlowJsonObject<
                    AllNodeType,
                    EdgeType
                  >;
                },
              },
            );
          }

          const {
            id,
            name,
            data,
            description,
            folder_id,
            endpoint_name,
            locked,
          } = flow;
          const payload =
            currentSavedFlow?.locked && locked === false
              ? { id, locked }
              : {
                  id,
                  name,
                  data: data!,
                  description,
                  folder_id,
                  endpoint_name,
                  locked,
                };
          mutate(payload, {
            onSuccess: (updatedFlow) => {
              const flows = useFlowsManagerStore.getState().flows;
              setSaveLoading(false);
              if (flows) {
                // updates flow in state
                setFlows(
                  flows.map((flow) => {
                    if (flow.id === updatedFlow.id) {
                      return updatedFlow;
                    }
                    return flow;
                  }),
                );
                setCurrentFlow(updatedFlow);
                resolve();
              } else {
                setErrorData({
                  title: "Failed to save flow",
                  list: ["Flows variable undefined"],
                });
                reject(new Error("Flows variable undefined"));
              }
            },
            onError: (e) => {
              const status = e?.response?.status;
              const detail = e?.response?.data?.detail;
              const message =
                status === 423 && typeof detail === "string"
                  ? detail
                  : e.message;
              setErrorData({
                title:
                  status === 423 ? "当前工作流已锁定" : "Failed to save flow",
                list: [message],
              });
              setSaveLoading(false);
              reject(e);
            },
          });
        } else {
          setErrorData({
            title: "Failed to save flow",
            list: ["Flow not found"],
          });
          reject(new Error("Flow not found"));
        }
      });
    }
  };

  return saveFlow;
};

export default useSaveFlow;
