const pad2 = (num: number): string => String(num).padStart(2, "0");

const DEFAULT_TIME_ZONE = "UTC";

const hasExplicitTimezone = (value: string): boolean =>
  /([zZ]|[+-]\d{2}:?\d{2})$/.test(value);

type DateParts = {
  year: number;
  month: number;
  day: number;
};

const getDateParts = (date: Date, timeZone: string): DateParts => {
  const parts = new Intl.DateTimeFormat("en-US", {
    day: "2-digit",
    month: "2-digit",
    timeZone,
    year: "numeric",
  }).formatToParts(date);

  const valueByType = Object.fromEntries(
    parts.map((part) => [part.type, part.value]),
  );

  return {
    day: Number(valueByType.day),
    month: Number(valueByType.month),
    year: Number(valueByType.year),
  };
};

export const parseApiTimestamp = (value: unknown): Date | null => {
  if (value === null || value === undefined) return null;
  if (value instanceof Date) {
    return Number.isNaN(value.getTime()) ? null : value;
  }

  const raw = String(value).trim();
  if (!raw) return null;

  const normalized = hasExplicitTimezone(raw)
    ? raw
    : raw.includes("T")
      ? `${raw}Z`
      : raw;

  const date = new Date(normalized);
  return Number.isNaN(date.getTime()) ? null : date;
};

type FormatSmartTimestampOptions = {
  timeZone?: string;
};

export const formatSmartTimestamp = (
  value: unknown,
  options: FormatSmartTimestampOptions = {},
): string => {
  const date = parseApiTimestamp(value);
  if (!date) return value ? String(value) : "";

  const timeZone = options.timeZone ?? DEFAULT_TIME_ZONE;
  const now = new Date();
  const dateParts = getDateParts(date, timeZone);
  const nowParts = getDateParts(now, timeZone);

  const time = new Intl.DateTimeFormat(undefined, {
    hour: "2-digit",
    hour12: false,
    minute: "2-digit",
    second: "2-digit",
    timeZone,
  }).format(date);

  const isToday =
    dateParts.year === nowParts.year &&
    dateParts.month === nowParts.month &&
    dateParts.day === nowParts.day;

  if (isToday) return time;

  const sameYear = dateParts.year === nowParts.year;
  if (sameYear) {
    return new Intl.DateTimeFormat(undefined, {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      hour12: false,
      minute: "2-digit",
      second: "2-digit",
      timeZone,
    }).format(date);
  }

  const ddmmyyyy = [
    pad2(dateParts.day),
    pad2(dateParts.month),
    dateParts.year,
  ].join("/");
  return `${ddmmyyyy} ${time}`;
};
