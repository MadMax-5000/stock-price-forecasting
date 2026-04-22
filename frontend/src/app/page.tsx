"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Play, Settings2 } from "lucide-react";
import { StockPicker } from "@/components/StockPicker";
import { SettingsPanel } from "@/components/SettingsPanel";
import { ModelComparison } from "@/components/ModelComparison";
import { PredictionChart } from "@/components/PredictionChart";
import { LoadingState } from "@/components/LoadingState";
import { Button } from "@/components/ui/button";
import { fetchStocks, predict } from "@/lib/utils";
import { Stock, PredictionResponse } from "@/types";

type PipelineStep = "idle" | "downloading" | "cleaning" | "features" | "models" | "training" | "complete";

export default function Home() {
  const [stocks, setStocks] = useState<Stock[]>([]);
  const [selectedStock, setSelectedStock] = useState("AAPL");
  const [startDate, setStartDate] = useState("2015-01-01");
  const [endDate, setEndDate] = useState("2025-04-21");
  const [predictionHorizon, setPredictionHorizon] = useState(30);
  const [trainRatio, setTrainRatio] = useState(0.7);
  const [threshold, setThreshold] = useState(0.51);
  
  const [isLoading, setIsLoading] = useState(false);
  const [pipelineStep, setPipelineStep] = useState<PipelineStep>("idle");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const pipelineTimerRef = useRef<NodeJS.Timeout | null>(null);

  const startPipelineAnimation = () => {
    const steps: PipelineStep[] = ["downloading", "cleaning", "features", "models", "training", "complete"];
    let currentStep = 0;
    
    const advanceStep = () => {
      if (currentStep < steps.length) {
        setPipelineStep(steps[currentStep]);
        currentStep++;
        
        const delays: Record<PipelineStep, number> = {
          idle: 0,
          downloading: 1500,
          cleaning: 1200,
          features: 1000,
          models: 2000,
          training: 1500,
          complete: 0,
        };
        
        pipelineTimerRef.current = setTimeout(advanceStep, delays[steps[currentStep - 1]] || 1000);
      }
    };
    
    advanceStep();
  };

  const stopPipelineAnimation = () => {
    if (pipelineTimerRef.current) {
      clearTimeout(pipelineTimerRef.current);
      pipelineTimerRef.current = null;
    }
    setPipelineStep("idle");
  };

  useEffect(() => {
    fetchStocks().then(setStocks).catch(console.error);
  }, []);

  useEffect(() => {
    return () => stopPipelineAnimation();
  }, []);

  const handleRunPipeline = async () => {
    setError(null);
    setResult(null);
    setIsLoading(true);
    startPipelineAnimation();
    
    try {
      const data = await predict({
        ticker: selectedStock,
        start_date: startDate,
        end_date: endDate,
        prediction_horizon: predictionHorizon,
        train_ratio: trainRatio,
        threshold: threshold,
      });
      setResult(data);
      setPipelineStep("complete");
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
      stopPipelineAnimation();
    } finally {
      setIsLoading(false);
      setTimeout(() => {
        if (!error) {
          setPipelineStep("idle");
        }
      }, 2000);
    }
  };

  return (
    <div className="min-h-screen">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
          <motion.aside
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-4 space-y-6"
          >
            <StockPicker
              stocks={stocks}
              selectedStock={selectedStock}
              onSelect={setSelectedStock}
            />

            <SettingsPanel
              startDate={startDate}
              endDate={endDate}
              predictionHorizon={predictionHorizon}
              trainRatio={trainRatio}
              threshold={threshold}
              onStartDateChange={setStartDate}
              onEndDateChange={setEndDate}
              onPredictionHorizonChange={setPredictionHorizon}
              onTrainRatioChange={setTrainRatio}
              onThresholdChange={setThreshold}
            />

            <Button
              onClick={handleRunPipeline}
              disabled={isLoading}
              className="w-full h-12 text-base"
              variant="primary"
            >
              {isLoading ? (
                <>
                  <span className="w-4 h-4 mr-2 border-2 border-current border-t-transparent rounded-full animate-spin" />
                  Analyse en cours...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  Lancer la prédiction
                </>
              )}
            </Button>
          </motion.aside>

          <motion.main
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="lg:col-span-8"
          >
            <AnimatePresence mode="wait">
              {error && (
                <motion.div
                  key="error"
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  className="p-4 rounded-xl border border-red-500/50 bg-red-500/10 text-red-400"
                >
                  {error}
                </motion.div>
              )}

              {isLoading && !result && !error && (
                <motion.div
                  key="loading"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <LoadingState step={pipelineStep} />
                </motion.div>
              )}

              {result && !error && (
                <motion.div
                  key="result"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-6"
                >
                  <PredictionChart
                    historical={result.historical}
                    predictions={result.predictions}
                    metrics={result.metrics}
                    ticker={selectedStock}
                  />

                  <ModelComparison
                    models={result.model_comparison}
                    bestModel={result.best_model}
                  />
                </motion.div>
              )}

              {!result && !isLoading && !error && (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center py-24"
                >
                  <div
                    className="w-20 h-20 rounded-2xl flex items-center justify-center mb-6"
                    style={{ backgroundColor: "var(--frosted-veil)" }}
                  >
                    <Settings2
                      className="w-10 h-10"
                      style={{ color: "var(--stone-gray)" }}
                    />
                  </div>
                  <h3
                    className="text-xl font-normal mb-2"
                    style={{ color: "var(--warm-parchment)" }}
                  >
                    Prêt à prédire
                  </h3>
                  <p
                    className="text-center max-w-md"
                    style={{ color: "var(--stone-gray)" }}
                  >
                    Sélectionnez une action et configurez vos paramètres, puis cliquez sur {"\""}Lancer la prédiction{"\""} 
                    pour démarrer le pipeline d'analyse.
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.main>
        </div>
      </div>
    </div>
  );
}