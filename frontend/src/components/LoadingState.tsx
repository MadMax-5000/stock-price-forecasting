"use client";

import { motion } from "framer-motion";
import { Database, Cpu, TrendingUp, Loader2 } from "lucide-react";

interface LoadingStateProps {
  step: "idle" | "downloading" | "cleaning" | "features" | "models" | "training" | "complete";
}

const steps = [
  { key: "downloading", label: "Téléchargement des données", icon: Database },
  { key: "cleaning", label: "Nettoyage et prétraitement", icon: Database },
  { key: "features", label: "Génération des caractéristiques", icon: Cpu },
  { key: "models", label: "Comparaison des modèles", icon: TrendingUp },
  { key: "training", label: "Entraînement du meilleur modèle", icon: Cpu },
];

export function LoadingState({ step }: LoadingStateProps) {
  const currentIndex = steps.findIndex((s) => s.key === step);

  return (
    <div className="flex flex-col items-center justify-center py-16">
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.3 }}
        className="relative"
      >
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
        >
          <Loader2 className="w-12 h-12" style={{ color: "var(--warm-parchment)" }} />
        </motion.div>
      </motion.div>

      <div className="mt-8 space-y-3">
        {steps.map((s, index) => {
          const Icon = s.icon;
          const isComplete = index < currentIndex;
          const isActive = index === currentIndex;

          return (
            <motion.div
              key={s.key}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className="flex items-center gap-3"
            >
              <div
                className="w-8 h-8 rounded-full flex items-center justify-center transition-all duration-300"
                style={{
                  backgroundColor: isComplete
                    ? "var(--success)"
                    : isActive
                    ? "var(--earth-gray)"
                    : "transparent",
                  border: `1px solid ${
                    isComplete
                      ? "var(--success)"
                      : isActive
                      ? "var(--warm-parchment)"
                      : "var(--mist-border)"
                  }`,
                }}
              >
                {isComplete ? (
                  <motion.svg
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    className="w-4 h-4 text-deep-void"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <motion.path
                      initial={{ pathLength: 0 }}
                      animate={{ pathLength: 1 }}
                      transition={{ duration: 0.3 }}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M5 13l4 4L19 7"
                    />
                  </motion.svg>
                ) : isActive ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <Loader2 className="w-4 h-4" style={{ color: "var(--warm-parchment)" }} />
                  </motion.div>
                ) : (
                  <Icon className="w-4 h-4" style={{ color: "var(--stone-gray)" }} />
                )}
              </div>
              <span
                className="text-sm transition-all duration-300"
                style={{
                  color: isActive ? "var(--warm-parchment)" : isComplete ? "var(--ash-gray)" : "var(--stone-gray)",
                  fontWeight: isActive ? 500 : 400,
                }}
              >
                {s.label}
              </span>
              {isActive && (
                <motion.span
                  initial={{ opacity: 0 }}
                  animate={{ opacity: [0, 1, 0] }}
                  transition={{ duration: 1, repeat: Infinity }}
                  className="ml-1"
                  style={{ color: "var(--warm-parchment)" }}
                >
                  ...
                </motion.span>
              )}
            </motion.div>
          );
        })}
      </div>

      {step === "complete" && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-8"
        >
          <div
            className="px-6 py-3 rounded-full text-sm font-medium"
            style={{ backgroundColor: "var(--success)", color: "var(--deep-void)" }}
          >
            Analyse terminée !
          </div>
        </motion.div>
      )}
    </div>
  );
}