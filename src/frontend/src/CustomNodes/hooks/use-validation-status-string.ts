import { useEffect } from "react";
import type { VertexBuildTypeAPI } from "../../types/api";
import { isErrorLog } from "../../types/utils/typeCheckingUtils";

const useValidationStatusString = (
  validationStatus: VertexBuildTypeAPI | null,
  setValidationString: (value: any) => void,
) => {
  useEffect(() => {
    if (validationStatus && validationStatus.data?.outputs) {
      let newValidationString = "";
      Object.values(validationStatus?.data?.outputs).forEach((output: any) => {
        if (isErrorLog(output)) {
          newValidationString += `${output.message.errorMessage}\n`;
        }
      });
      setValidationString(newValidationString);
    }
  }, [validationStatus, validationStatus?.data?.outputs, setValidationString]);
};

export function extractStackTrace(
  validationStatus: VertexBuildTypeAPI | null,
): string {
  if (!validationStatus?.data?.outputs) return "";
  let stackTrace = "";
  Object.values(validationStatus.data.outputs).forEach((output: any) => {
    if (isErrorLog(output) && output.message.stackTrace) {
      stackTrace += `${output.message.stackTrace}\n`;
    }
  });
  return stackTrace.trim();
}

export default useValidationStatusString;
