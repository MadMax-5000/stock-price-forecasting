import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
    return twMerge(clsx(inputs));
}

export async function fetchStocks() {
    const res = await fetch("http://localhost:8000/api/stocks");
    if (!res.ok) throw new Error("Failed to fetch stocks");
    return res.json();
}

export async function predict(data: {
    ticker: string;
    start_date: string;
    end_date: string;
    prediction_horizon: number;
    train_ratio: number;
    threshold: number;
}) {
    const res = await fetch("http://localhost:8000/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
    });
    if (!res.ok) throw new Error("Failed to fetch prediction");
    return res.json();
}
