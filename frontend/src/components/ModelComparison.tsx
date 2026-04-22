"use client";

import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { ModelComparison as ModelComparisonType } from "@/types";
import { Trophy, TrendingUp, Crosshair, Activity } from "lucide-react";

interface ModelComparisonProps {
  models: ModelComparisonType[];
  bestModel: { model_name: string; precision: number };
}

export function ModelComparison({ models, bestModel }: ModelComparisonProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Trophy className="w-5 h-5" style={{ color: "var(--warm-parchment)" }} />
          Comparaison des modèles
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid gap-4">
          {models.map((model, index) => {
            const isBest = model.name === bestModel.model_name;
            
            return (
              <motion.div
                key={model.name}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
                className={`relative p-4 rounded-xl border transition-all duration-200 ${
                  isBest ? "border-mist-border" : "border-transparent"
                }`}
                style={{
                  backgroundColor: isBest ? "var(--frosted-veil)" : "transparent",
                }}
              >
                {isBest && (
                  <div
                    className="absolute top-2 right-2 px-2 py-0.5 rounded text-xs font-medium"
                    style={{ backgroundColor: "var(--success)", color: "var(--deep-void)" }}
                  >
                    Meilleur
                  </div>
                )}
                
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center text-xs font-medium"
                      style={{ 
                        backgroundColor: isBest ? "var(--earth-gray)" : "var(--frosted-veil)",
                        color: isBest ? "var(--warm-parchment)" : "var(--stone-gray)"
                      }}
                    >
                      {index + 1}
                    </div>
                    <span
                      className="font-medium"
                      style={{ color: isBest ? "var(--warm-parchment)" : "var(--ash-gray)" }}
                    >
                      {model.name}
                    </span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Crosshair className="w-4 h-4" style={{ color: "var(--stone-gray)" }} />
                    <span className="text-sm" style={{ color: "var(--warm-parchment)" }}>
                      {(model.precision * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-4 gap-2">
                  <MetricItem
                    label="Précision"
                    value={model.precision}
                    icon={Crosshair}
                    isHighlighted={isBest}
                    delay={0.1}
                  />
                  <MetricItem
                    label="Exactitude"
                    value={model.accuracy}
                    icon={Activity}
                    isHighlighted={false}
                    delay={0.15}
                  />
                  <MetricItem
                    label="Rappel"
                    value={model.recall}
                    icon={TrendingUp}
                    isHighlighted={false}
                    delay={0.2}
                  />
                  <MetricItem
                    label="F1"
                    value={model.f1}
                    icon={Trophy}
                    isHighlighted={false}
                    delay={0.25}
                  />
                </div>
              </motion.div>
            );
          })}
        </div>
      </CardContent>
    </Card>
  );
}

function MetricItem({
  label,
  value,
  icon: Icon,
  isHighlighted,
  delay,
}: {
  label: string;
  value: number;
  icon: React.ElementType;
  isHighlighted: boolean;
  delay: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay }}
      className="flex flex-col items-center gap-1 p-2 rounded-lg"
      style={{ backgroundColor: "var(--frosted-veil)" }}
    >
      <Icon
        className="w-3 h-3"
        style={{ color: isHighlighted ? "var(--warm-parchment)" : "var(--stone-gray)" }}
      />
      <span
        className="text-xs"
        style={{ color: isHighlighted ? "var(--warm-parchment)" : "var(--ash-gray)" }}
      >
        {(value * 100).toFixed(1)}%
      </span>
      <span className="text-[10px] uppercase tracking-wider" style={{ color: "var(--stone-gray)" }}>
        {label}
      </span>
    </motion.div>
  );
}