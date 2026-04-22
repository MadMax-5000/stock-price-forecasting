export interface Stock {
  symbol: string;
  name: string;
  sector: string;
}

export interface PredictionData {
  date: string;
  close: number;
  direction?: string;
  probability?: number;
}

export interface ModelComparison {
  name: string;
  precision: number;
  accuracy: number;
  recall: number;
  f1: number;
  total_trades: number;
}

export interface BestModel {
  model_name: string;
  precision: number;
  accuracy: number;
  recall: number;
  f1: number;
}

export interface Metrics {
  last_historical_price: number;
  predicted_end_price: number;
  predicted_change_pct: number;
}

export interface PredictionResponse {
  historical: PredictionData[];
  predictions: PredictionData[];
  model_comparison: ModelComparison[];
  best_model: BestModel;
  metrics: Metrics;
}

export interface PredictionRequest {
  ticker: string;
  start_date: string;
  end_date: string;
  prediction_horizon: number;
  train_ratio: number;
  threshold: number;
}