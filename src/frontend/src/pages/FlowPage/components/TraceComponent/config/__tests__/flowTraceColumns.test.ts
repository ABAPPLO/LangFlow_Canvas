jest.mock("@/utils/dateTime", () => ({
  formatSmartTimestamp: jest.fn(() => "formatted-time"),
}));

import { formatSmartTimestamp } from "@/utils/dateTime";
import { createFlowTracesColumns } from "../flowTraceColumns";

describe("createFlowTracesColumns", () => {
  it("formats the timestamp column in UTC+8", () => {
    const columns = createFlowTracesColumns();
    const timestampColumn = columns.find(
      (column) => column.field === "startTime",
    );
    const startTime = "2026-05-29T03:52:55Z";

    expect(timestampColumn?.headerName).toBe("Timestamp (UTC+8)");
    expect(
      (timestampColumn?.valueGetter as (params: {
        data: { startTime: string };
      }) => string)({
        data: { startTime },
      }),
    ).toBe("formatted-time");
    expect(formatSmartTimestamp).toHaveBeenCalledWith(startTime, {
      timeZone: "Asia/Shanghai",
    });
  });
});
