import { Textarea } from "../../../components/ui/textarea";

const TextOutputView = ({
  left,
  value,
}: {
  left: boolean | undefined;
  value: any;
}) => {
  if (typeof value === "object" && Object.keys(value).includes("text")) {
    value = value.text;
  }

  const isTruncated = value?.length > 20000;

  return (
    <div className={`flex flex-col ${left ? "" : "min-h-0 flex-1"}`}>
      <Textarea
        className={`w-full custom-scroll ${left ? "min-h-32 resize-none" : "min-h-[600px] flex-1 resize-y"}`}
        placeholder={"Empty"}
        readOnly
        value={value}
      />
      {isTruncated && (
        <div className="mt-2 text-xs text-muted-foreground">
          This output has been truncated due to its size.
        </div>
      )}
    </div>
  );
};

export default TextOutputView;
