import { useState, type MouseEvent } from "react";
import type { ColDef } from "ag-grid-community";
import IconComponent from "@/components/common/genericIconComponent";
import useAlertStore from "@/stores/alertStore";
import { formatSmartTimestamp } from "@/utils/dateTime";
import { formatTotalLatency, getStatusIconProps } from "../traceViewHelpers";
import {
  formatObjectValue,
  formatRunValue,
  pickFirstNumber,
} from "./flowTraceColumnsHelpers";

type TaskIdCellParams = {
  value?: string | null;
  data?: {
    taskId?: string | null;
    task_id?: string | null;
  };
};

function TaskIdCopyCell(params: TaskIdCellParams) {
  const [copied, setCopied] = useState(false);
  const setSuccessData = useAlertStore((state) => state.setSuccessData);
  const setErrorData = useAlertStore((state) => state.setErrorData);
  const taskId = String(
    params.value ?? params.data?.taskId ?? params.data?.task_id ?? "",
  ).trim();

  if (!taskId) {
    return null;
  }

  const handleCopy = async (event: MouseEvent<HTMLButtonElement>) => {
    event.preventDefault();
    event.stopPropagation();

    try {
      await navigator.clipboard.writeText(taskId);
      setCopied(true);
      setSuccessData({ title: "Task ID copied to clipboard" });
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      setErrorData({
        title: "Copy failed",
        list: ["Unable to copy Task ID. Please copy manually."],
      });
    }
  };

  return (
    <div className="flex h-full w-full min-w-0 items-center gap-2">
      <span className="min-w-0 flex-1 truncate font-mono text-xs" title={taskId}>
        {taskId}
      </span>
      <button
        type="button"
        className="ml-auto inline-flex h-6 w-6 shrink-0 items-center justify-center rounded text-muted-foreground transition hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
        aria-label="Copy Task ID"
        data-testid="copy-task-id-button"
        title="Copy Task ID"
        onClick={handleCopy}
      >
        <IconComponent
          name={copied ? "Check" : "Copy"}
          className="h-3.5 w-3.5"
          skipFallback
        />
      </button>
    </div>
  );
}

export function createFlowTracesColumns({
  flowId,
  flowName,
}: {
  flowId?: string | null;
  flowName?: string | null;
} = {}): ColDef[] {
  return [
    {
      headerName: "Run",
      field: "run",
      flex: 1.0,
      minWidth: 240,
      filter: false,
      sortable: false,
      editable: false,
      valueGetter: () => formatRunValue(flowName, flowId),
    },
    {
      headerName: "Trace ID",
      field: "id",
      flex: 0.3,
      minWidth: 240,
      filter: false,
      sortable: false,
      editable: false,
    },
    {
      headerName: "Task ID",
      field: "taskId",
      flex: 0.3,
      minWidth: 200,
      filter: false,
      sortable: false,
      editable: false,
      valueGetter: (params) =>
        params.data?.taskId ?? params.data?.task_id ?? "",
      cellRenderer: (params: TaskIdCellParams) => (
        <TaskIdCopyCell {...params} />
      ),
    },

    {
      headerName: "Timestamp (UTC)",
      field: "startTime",
      flex: 0.5,
      minWidth: 70,
      filter: false,
      sortable: false,
      editable: false,
      valueGetter: (params) => formatSmartTimestamp(params.data?.startTime),
    },
    {
      headerName: "Input",
      field: "input",
      flex: 1,
      minWidth: 150,
      filter: false,
      sortable: false,
      editable: false,
      valueGetter: (params) => formatObjectValue(params.data?.input),
    },
    {
      headerName: "Output",
      field: "output",
      flex: 1,
      minWidth: 150,
      filter: false,
      sortable: false,
      editable: false,
      valueGetter: (params) => formatObjectValue(params.data?.output),
    },
    {
      headerName: "Token",
      field: "totalTokens",
      flex: 0.5,
      minWidth: 50,
      filter: false,
      sortable: false,
      editable: false,
      valueGetter: (params) => {
        const tokens = pickFirstNumber(
          params.data?.totalTokens,
          params.data?.total_tokens,
        );
        return tokens === null ? "" : String(tokens);
      },
    },
    {
      headerName: "Latency",
      field: "totalLatencyMs",
      flex: 0.6,
      minWidth: 50,
      filter: false,
      sortable: false,
      editable: false,
      valueGetter: (params) => {
        const latencyMs = pickFirstNumber(
          params.data?.totalLatencyMs,
          params.data?.total_latency_ms,
        );
        return formatTotalLatency(latencyMs);
      },
    },
    {
      headerName: "Status",
      field: "status",
      flex: 0.6,
      minWidth: 100,
      filter: false,
      sortable: false,
      editable: false,
      cellRenderer: (params: { value: string | null | undefined }) => {
        const status = params.value ?? "unknown";
        const { colorClass, iconName, shouldSpin } = getStatusIconProps(status);

        return (
          <div className="flex items-center">
            <IconComponent
              name={iconName}
              className={`h-4 w-4 ${colorClass} ${shouldSpin ? "animate-spin" : ""}`}
              aria-label={status}
              dataTestId={`flow-log-status-${status}`}
              skipFallback
            />
          </div>
        );
      },
    },
  ];
}
