import { Panel } from "@xyflow/react";
import { AnimatePresence, motion } from "framer-motion";
import { memo, useEffect, useMemo, useState } from "react";
import { shouldRenderInspectionPanelField } from "@/CustomNodes/helpers/parameter-filtering";
import { Separator } from "@/components/ui/separator";
import type { AllNodeType } from "@/types/flow";
import { cn } from "@/utils/utils";
import InspectionPanelFields from "./components/InspectionPanelFields";
import InspectionPanelHeader from "./components/InspectionPanelHeader";

interface InspectionPanelProps {
  selectedNode: AllNodeType | null;
}

const InspectionPanel = memo(function InspectionPanel({
  selectedNode,
}: InspectionPanelProps) {
  const [isEditingFields, setIsEditingFields] = useState(false);

  // Reset edit mode when panel closes or node changes
  useEffect(() => {
    setIsEditingFields(false);
  }, [selectedNode?.id]);

  const hasAdvancedFields = useMemo(() => {
    if (!selectedNode || selectedNode.type !== "genericNode") {
      return false;
    }

    const template = selectedNode.data?.node?.template ?? {};
    const isToolMode = selectedNode.data?.node?.tool_mode;

    return Object.entries(template).some(([templateField, fieldTemplate]) =>
      shouldRenderInspectionPanelField(
        templateField,
        fieldTemplate,
        isToolMode,
      ),
    );
  }, [selectedNode]);

  return (
    <AnimatePresence mode="wait">
      {selectedNode &&
        selectedNode.type === "genericNode" &&
        hasAdvancedFields && (
          <Panel
            position="top-right"
            className={cn(
              "!top-[3rem] !-right-2 !bottom-10 relative",
              "w-[340px]",
              "pointer-events-none",
            )}
          >
            <motion.div
              initial={{ x: "100%", opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: "100%", opacity: 0 }}
              transition={{ duration: 0, ease: "easeInOut" }}
              className={cn(
                "max-h-full w-[320px] ml-auto",
                "rounded-xl border bg-background shadow-lg",
                "overflow-y-auto overflow-x-visible flex flex-col pointer-events-auto",
              )}
            >
              <InspectionPanelHeader
                data={selectedNode.data}
                isEditingFields={isEditingFields}
                setIsEditingFields={setIsEditingFields}
              />
              <Separator className="my-0.5" />
              <InspectionPanelFields
                data={selectedNode.data}
                key={selectedNode.id}
                isEditingFields={isEditingFields}
              />
            </motion.div>
          </Panel>
        )}
    </AnimatePresence>
  );
});

export default InspectionPanel;
