"use client";

interface SliderProps {
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  label?: string;
  displayValue?: string;
  minLabel?: string;
  maxLabel?: string;
}

export function Slider({ value, min, max, onChange, label, displayValue, minLabel, maxLabel }: SliderProps) {
  const percentage = ((value - min) / (max - min)) * 100;

  return (
    <div className="w-full">
      {(label || displayValue) && (
        <div className="flex justify-between items-center mb-2">
          {label && (
            <span className="text-sm" style={{ color: "var(--ash-gray)" }}>
              {label}
            </span>
          )}
          {displayValue && (
            <span className="text-sm font-medium" style={{ color: "var(--warm-parchment)" }}>
              {displayValue}
            </span>
          )}
        </div>
      )}
      <div className="relative h-1 w-full rounded-full" style={{ background: "#e5e5e5" }}>
        <div
          className="absolute h-full rounded-full"
          style={{
            width: `${percentage}%`,
            background: "var(--warm-parchment)",
          }}
        />
        <input
          type="range"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          className="absolute inset-0 w-full cursor-pointer opacity-0"
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 w-4 h-4 rounded-full border-2 cursor-pointer transition-transform hover:scale-110"
          style={{
            left: `calc(${percentage}% - 8px)`,
            background: "var(--deep-void)",
            borderColor: "var(--warm-parchment)",
          }}
        />
      </div>
      {(minLabel || maxLabel) && (
        <div className="flex justify-between text-xs mt-1" style={{ color: "var(--stone-gray)" }}>
          <span>{minLabel}</span>
          <span>{maxLabel}</span>
        </div>
      )}
    </div>
  );
}