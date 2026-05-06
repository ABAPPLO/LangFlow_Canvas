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

  return (
    <div className={`flex flex-col ${left ? "" : "min-h-0 flex-1"}`}>
      <Textarea
        className={`w-full custom-scroll ${left ? "min-h-32 resize-none" : "min-h-[600px] flex-1 resize-y"}`}
        placeholder={"Empty"}
        readOnly
        value={value}
      />
    </div>
  );
};

export default TextOutputView;
