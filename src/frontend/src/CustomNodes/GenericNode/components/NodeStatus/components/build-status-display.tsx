import { useState } from "react";
import {
  RUN_TIMESTAMP_PREFIX,
  STATUS_BUILD,
  STATUS_BUILDING,
  STATUS_INACTIVE,
  STATUS_MISSING_FIELDS_ERROR,
} from "@/constants/constants";
import { BuildStatus } from "@/constants/enums";

const StatusMessage = ({ children, className = "text-foreground" }) => (
  <span className={`flex ${className}`}>{children}</span>
);

const TimeStamp = ({ prefix, time }) => (
  <div className="flex items-center text-sm text-secondary-foreground">
    <div>{prefix}</div>
    <div className="ml-1 text-secondary-foreground">{time}</div>
  </div>
);

const Duration = ({ duration }) => (
  <div className="flex items-center text-secondary-foreground">
    <div>Duration:</div>
    <div className="ml-1">{duration}</div>
  </div>
);

function formatStackTrace(trace: string): string {
  const lines = trace.trim().split("\n");
  if (lines.length <= 8) return trace.trim();
  return (
    lines.slice(0, 8).join("\n") + `\n  ... (${lines.length - 8} more lines)`
  );
}

const StackTraceSection = ({ stackTrace }: { stackTrace: string }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="mt-1 border-t border-border pt-1">
      <button
        type="button"
        onClick={() => setExpanded(!expanded)}
        className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
      >
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
          className={`transition-transform ${expanded ? "rotate-90" : ""}`}
        >
          <polyline points="9 18 15 12 9 6" />
        </svg>
        {expanded ? "Hide Stack Trace" : "Show Stack Trace"}
      </button>
      {expanded && (
        <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted/50 p-2 font-mono text-[10px] leading-tight text-muted-foreground">
          {formatStackTrace(stackTrace)}
        </pre>
      )}
    </div>
  );
};

const ValidationDetails = ({
  validationString,
  lastRunTime,
  validationStatus,
  stackTrace,
}) => (
  <div className="max-h-100 px-1 py-2.5">
    <div className="flex max-h-80 flex-col gap-2">
      {validationString && (
        <div className="break-words text-sm text-foreground">
          {validationString}
        </div>
      )}
      {stackTrace && <StackTraceSection stackTrace={stackTrace} />}
      {lastRunTime && (
        <TimeStamp prefix={RUN_TIMESTAMP_PREFIX} time={lastRunTime} />
      )}
      <Duration duration={validationStatus?.data.duration} />
    </div>
  </div>
);

const BuildStatusDisplay = ({
  buildStatus,
  validationStatus,
  validationString,
  lastRunTime,
  stackTrace,
}) => {
  if (buildStatus === BuildStatus.BUILDING) {
    return <StatusMessage>{STATUS_BUILDING}</StatusMessage>;
  }

  if (buildStatus === BuildStatus.INACTIVE) {
    return <StatusMessage>{STATUS_INACTIVE}</StatusMessage>;
  }

  if (buildStatus === BuildStatus.ERROR && !validationStatus) {
    // If the build status is error and there is no validation status, it means that it failed before building, so show the Missing Required Fields error message
    return <StatusMessage>{STATUS_MISSING_FIELDS_ERROR}</StatusMessage>;
  }

  if (!validationStatus) {
    return <StatusMessage>{STATUS_BUILD}</StatusMessage>;
  }

  return (
    <ValidationDetails
      validationString={validationString}
      lastRunTime={lastRunTime}
      validationStatus={validationStatus}
      stackTrace={stackTrace}
    />
  );
};

export default BuildStatusDisplay;
