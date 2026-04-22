"use client";

import { Label } from "./ui/label";
import { Card, CardContent } from "./ui/card";
import { Slider } from "./ui/slider";

interface SettingsPanelProps {
  startDate: string;
  endDate: string;
  predictionHorizon: number;
  trainRatio: number;
  threshold: number;
  onStartDateChange: (date: string) => void;
  onEndDateChange: (date: string) => void;
  onPredictionHorizonChange: (value: number) => void;
  onTrainRatioChange: (value: number) => void;
  onThresholdChange: (value: number) => void;
}

export function SettingsPanel({
  startDate,
  endDate,
  predictionHorizon,
  trainRatio,
  threshold,
  onStartDateChange,
  onEndDateChange,
  onPredictionHorizonChange,
  onTrainRatioChange,
  onThresholdChange,
}: SettingsPanelProps) {
  return (
    <Card className="mt-6">
      <CardContent className="p-6 space-y-6">
        <div>
          <Label className="mb-3 block">Période</Label>
          <div className="flex gap-3">
            <div className="flex-1">
              <input
                type="date"
                value={startDate}
                onChange={(e) => onStartDateChange(e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm outline-none border transition-colors duration-200"
                style={{
                  background: "var(--frosted-veil)",
                  borderColor: "var(--mist-border)",
                  color: "var(--warm-parchment)",
                }}
              />
            </div>
            <div className="flex-1">
              <input
                type="date"
                value={endDate}
                onChange={(e) => onEndDateChange(e.target.value)}
                className="w-full px-3 py-2 rounded-lg text-sm outline-none border transition-colors duration-200"
                style={{
                  background: "var(--frosted-veil)",
                  borderColor: "var(--mist-border)",
                  color: "var(--warm-parchment)",
                }}
              />
            </div>
          </div>
        </div>

        <div>
          <div className="flex gap-1 mb-3">
            {[7, 14, 30, 60, 90].map((days) => (
              <button
                key={days}
                onClick={() => onPredictionHorizonChange(days)}
                className="flex-1 py-1.5 text-xs rounded-md transition-colors"
                style={{
                  background:
                    predictionHorizon === days
                      ? "var(--warm-parchment)"
                      : "var(--frosted-veil)",
                  color:
                    predictionHorizon === days
                      ? "var(--deep-void)"
                      : "var(--stone-gray)",
                  fontWeight: predictionHorizon === days ? 500 : 400,
                }}
              >
                {days}d
              </button>
            ))}
          </div>
          <Slider
            value={predictionHorizon}
            min={7}
            max={90}
            onChange={onPredictionHorizonChange}
            label="Horizon de prédiction"
            displayValue={`${predictionHorizon} jours`}
            minLabel="7 jours"
            maxLabel="90 jours"
          />
        </div>

        <div>
          <Slider
            value={trainRatio * 100}
            min={50}
            max={90}
            onChange={(val) => onTrainRatioChange(val / 100)}
            label="Ratio d'entraînement"
            displayValue={`${Math.round(trainRatio * 100)}%`}
            minLabel="50%"
            maxLabel="90%"
          />
        </div>

        <div>
          <Slider
            value={threshold * 100}
            min={30}
            max={70}
            onChange={(val) => onThresholdChange(val / 100)}
            label="Seuil de décision"
            displayValue={`${(threshold * 100).toFixed(0)}%`}
            minLabel="30%"
            maxLabel="70%"
          />
        </div>
      </CardContent>
    </Card>
  );
}