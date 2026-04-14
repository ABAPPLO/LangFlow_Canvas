import { UseMutationResult } from "@tanstack/react-query";
import { useQueryFunctionType } from "@/types/api";
import { api } from "../../api";
import { getURL } from "../../helpers/constants";
import { UseRequestProcessor } from "../../services/request-processor";

export interface CustomModelsResponse {
  custom_models: Record<string, string[]>;
}

export interface CustomModelRequest {
  provider: string;
  model_name: string;
}

export const useGetCustomModels: useQueryFunctionType<
  undefined,
  CustomModelsResponse
> = (options) => {
  const { query } = UseRequestProcessor();

  const getCustomModelsFn = async (): Promise<CustomModelsResponse> => {
    const response = await api.get<CustomModelsResponse>(
      `${getURL("MODELS")}/custom_models`,
    );
    return response.data;
  };

  const queryResult = query(["useGetCustomModels"], getCustomModelsFn, options);

  return queryResult;
};

export const useAddCustomModel: useMutationFunctionType<
  undefined,
  CustomModelRequest,
  CustomModelsResponse,
  Error
> = (options?) => {
  const { mutate } = UseRequestProcessor();

  const addCustomModelFn = async (
    data: CustomModelRequest,
  ): Promise<CustomModelsResponse> => {
    const response = await api.post<CustomModelsResponse>(
      `${getURL("MODELS")}/custom_models`,
      data,
    );
    return response.data;
  };

  const mutation: UseMutationResult<
    CustomModelsResponse,
    Error,
    CustomModelRequest
  > = mutate(["useAddCustomModel"], addCustomModelFn, options);

  return mutation;
};

export const useRemoveCustomModel: useMutationFunctionType<
  undefined,
  CustomModelRequest,
  CustomModelsResponse,
  Error
> = (options?) => {
  const { mutate } = UseRequestProcessor();

  const removeCustomModelFn = async (
    data: CustomModelRequest,
  ): Promise<CustomModelsResponse> => {
    const response = await api.delete<CustomModelsResponse>(
      `${getURL("MODELS")}/custom_models`,
      { data },
    );
    return response.data;
  };

  const mutation: UseMutationResult<
    CustomModelsResponse,
    Error,
    CustomModelRequest
  > = mutate(["useRemoveCustomModel"], removeCustomModelFn, options);

  return mutation;
};
