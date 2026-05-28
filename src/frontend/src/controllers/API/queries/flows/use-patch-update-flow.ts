import type { UseMutationResult } from "@tanstack/react-query";
import type { ReactFlowJsonObject } from "@xyflow/react";
import type { useMutationFunctionType } from "@/types/api";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

interface IPatchUpdateFlow {
  id: string;
  name?: string;
  data?: ReactFlowJsonObject;
  description?: string;
  folder_id?: string | null | undefined;
  endpoint_name?: string | null | undefined;
  locked?: boolean | null | undefined;
  access_type?: "PUBLIC" | "PRIVATE" | "PROTECTED";
}

type FlowPatchError = Error & {
  response?: {
    status?: number;
    data?: {
      detail?: unknown;
    };
  };
};

export const usePatchUpdateFlow: useMutationFunctionType<
  undefined,
  IPatchUpdateFlow,
  IPatchUpdateFlow,
  FlowPatchError
> = (options?) => {
  const { mutate, queryClient } = UseRequestProcessor();

  const PatchUpdateFlowFn = async ({
    id,
    ...payload
  }: IPatchUpdateFlow): Promise<IPatchUpdateFlow> => {
    const response = await api.patch(`${getURL("FLOWS")}/${id}`, payload);

    return response.data;
  };

  const shouldRetry = (failureCount: number, error: FlowPatchError) => {
    if (error?.response?.status === 423) {
      return false;
    }

    const optionRetry = options?.retry;
    if (typeof optionRetry === "function") {
      return optionRetry(failureCount, error);
    }
    if (typeof optionRetry === "number") {
      return failureCount < optionRetry;
    }
    if (typeof optionRetry === "boolean") {
      return optionRetry;
    }
    return failureCount < 3;
  };

  const mutation: UseMutationResult<
    IPatchUpdateFlow,
    FlowPatchError,
    IPatchUpdateFlow
  > = mutate(["usePatchUpdateFlow"], PatchUpdateFlowFn, {
    onSettled: (res) => {
      if (res) {
        queryClient.refetchQueries({
          queryKey: ["useGetFolders", res.folder_id],
        });
      }
      queryClient.refetchQueries({
        queryKey: ["useGetFolder"],
      });
    },
    ...options,
    retry: shouldRetry,
  });

  return mutation;
};
