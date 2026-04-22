"use client";

import { useState } from "react";
import {
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Area,
    ComposedChart,
} from "recharts";
import { motion } from "framer-motion";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { TrendingUp, TrendingDown, DollarSign } from "lucide-react";
import { PredictionData, Metrics } from "@/types";

interface PredictionChartProps {
    historical: PredictionData[];
    predictions: PredictionData[];
    metrics: Metrics;
    ticker: string;
}

export function PredictionChart({
    historical,
    predictions,
    metrics,
    ticker,
}: PredictionChartProps) {
    const [showHistorical, setShowHistorical] = useState(true);
    const [showPrediction, setShowPrediction] = useState(true);

    const trainRatio = 0.7;
    const trainSize = Math.floor(historical.length * trainRatio);
    const trainData = historical.slice(0, trainSize);
    const testData = historical.slice(trainSize);
    const lastTrainPoint = trainData[trainData.length - 1];

    const unifiedData = [
        ...trainData.map((d) => ({
            date: d.date,
            trainClose: d.close,
            testClose: undefined as number | undefined,
        })),
        ...(lastTrainPoint ? [
            {
                date: lastTrainPoint.date,
                trainClose: lastTrainPoint.close,
                testClose: lastTrainPoint.close,
            },
        ] : []),
        ...testData.map((d) => ({
            date: d.date,
            trainClose: undefined as number | undefined,
            testClose: d.close,
        })),
    ];

    const formatDate = (date: string) => {
        const d = new Date(date);
        return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
    };

    const formatPrice = (value: number) => `$${value.toFixed(2)}`;

    return (
        <Card>
            <CardHeader>
                <CardTitle className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <TrendingUp className="w-5 h-5" style={{ color: "var(--warm-parchment)" }} />
                        Prévision du prix {ticker}
                    </div>
                    <div className="flex gap-2">
                        <button
                            onClick={() => setShowHistorical(!showHistorical)}
                            className={`px-3 py-1 text-xs rounded-full transition-colors ${showHistorical
                                    ? "bg-[#22c55e] text-deep-void"
                                    : "bg-frosted-veil text-stone-gray"
                                }`}
                        >
                            Train data
                        </button>
                        <button
                            onClick={() => setShowPrediction(!showPrediction)}
                            className={`px-3 py-1 text-xs rounded-full transition-colors ${showPrediction
                                    ? "bg-[#2196f3] text-deep-void"
                                    : "bg-frosted-veil text-stone-gray"
                                }`}
                        >
                            Test data
                        </button>
                    </div>
                </CardTitle>
            </CardHeader>
            <CardContent>
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                    className="h-[400px]"
                >
                    <ResponsiveContainer width="100%" height="100%">
                        {/* Single unified dataset — no separate allData merges */}
                        <ComposedChart data={unifiedData}>
                            <defs>
                                <linearGradient id="historicalGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#2196f3" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#2196f3" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="predictionGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid
                                strokeDasharray="3 3"
                                stroke="rgba(226,226,226,0.15)"
                                vertical={false}
                            />
                            <XAxis
                                dataKey="date"
                                tickFormatter={formatDate}
                                tick={{ fill: "#868584", fontSize: 10 }}
                                axisLine={{ stroke: "rgba(226,226,226,0.3)" }}
                                tickLine={false}
                                interval="preserveStartEnd"
                            />
                            <YAxis
                                tickFormatter={(v) => `$${v.toFixed(0)}`}
                                tick={{ fill: "#868584", fontSize: 10 }}
                                axisLine={false}
                                tickLine={false}
                                domain={["auto", "auto"]}
                                width={60}
                            />
                            <Tooltip
                                contentStyle={{
                                    backgroundColor: "#1a1918",
                                    border: "1px solid rgba(226,226,226,0.35)",
                                    borderRadius: "8px",
                                    color: "#faf9f6",
                                }}
                                labelStyle={{ color: "#afaeac" }}
                                formatter={(value, name) => [
                                    `$${Number(value).toFixed(2)}`,
                                    name === "trainClose" ? "Train data" : name === "testClose" ? "Test data" : name,
                                ]}
                                labelFormatter={(label) => formatDate(String(label))}
                            />

                            {/* Train data line */}
                            {showHistorical && (
                                <Area
                                    type="monotone"
                                    dataKey="trainClose"
                                    stroke="#22c55e"
                                    strokeWidth={2}
                                    fill="url(#historicalGradient)"
                                    dot={false}
                                    connectNulls={false}
                                />
                            )}

                            {/* Test data line */}
                            {showPrediction && (
                                <Area
                                    type="monotone"
                                    dataKey="testClose"
                                    stroke="#2196f3"
                                    strokeWidth={2}
                                    fill="url(#predictionGradient)"
                                    dot={false}
                                    connectNulls={false}
                                />
                            )}
                        </ComposedChart>
                    </ResponsiveContainer>
                </motion.div>

                <div className="grid grid-cols-3 gap-4 mt-6">
                    <MetricCard
                        label="Prix actuel"
                        value={formatPrice(metrics.last_historical_price)}
                        icon={DollarSign}
                    />
                    <MetricCard
                        label="Prix prédit"
                        value={formatPrice(metrics.predicted_end_price)}
                        icon={metrics.predicted_change_pct >= 0 ? TrendingUp : TrendingDown}
                        change={metrics.predicted_change_pct}
                    />
                    <MetricCard
                        label="Variation"
                        value={`${metrics.predicted_change_pct >= 0 ? "+" : ""}${metrics.predicted_change_pct.toFixed(2)}%`}
                        icon={metrics.predicted_change_pct >= 0 ? TrendingUp : TrendingDown}
                        isPositive={metrics.predicted_change_pct >= 0}
                    />
                </div>
            </CardContent>
        </Card>
    );
}

function MetricCard({
    label,
    value,
    icon: Icon,
    change,
    isPositive,
}: {
    label: string;
    value: string;
    icon: React.ElementType;
    change?: number;
    isPositive?: boolean;
}) {
    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 rounded-xl"
            style={{ backgroundColor: "var(--frosted-veil)" }}
        >
            <div className="flex items-center gap-2 mb-1">
                <Icon
                    className="w-4 h-4"
                    style={{
                        color:
                            change !== undefined
                                ? isPositive
                                    ? "var(--success)"
                                    : "var(--error)"
                                : "var(--stone-gray)",
                    }}
                />
                <span
                    className="text-xs uppercase tracking-wider"
                    style={{ color: "var(--stone-gray)" }}
                >
                    {label}
                </span>
            </div>
            <div
                className="text-xl font-medium"
                style={{ color: "var(--warm-parchment)" }}
            >
                {value}
            </div>
        </motion.div>
    );
}
