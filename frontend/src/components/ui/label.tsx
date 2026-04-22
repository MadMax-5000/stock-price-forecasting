"use client";

import * as React from "react";
import { cn } from "@/lib/utils";

const Label = React.forwardRef<HTMLLabelElement, React.LabelHTMLAttributes<HTMLLabelElement>>(
  ({ className, ...props }, ref) => (
    <label
      ref={ref}
      className={cn(
        "text-xs uppercase tracking-[2.4px] font-normal",
        className
      )}
      style={{ color: "var(--stone-gray)" }}
      {...props}
    />
  )
);
Label.displayName = "Label";

export { Label };