"""Tests for the models module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestModelsUtils:
    """Tests for model utility functions."""

    @pytest.fixture
    def sample_ml_data(self) -> pd.DataFrame:
        """Create sample data for ML testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        data = {
            "Close": np.cumsum(np.random.randn(500)) + 100,
            "Target": np.random.randint(0, 2, 500),
            "Close_Ratio_5": np.random.uniform(0.9, 1.1, 500),
            "Close_Ratio_10": np.random.uniform(0.9, 1.1, 500),
            "Trend_5": np.random.uniform(0, 5, 500),
            "Trend_10": np.random.uniform(0, 10, 500),
        }
        return pd.DataFrame(data, index=dates)

    def test_engineer_features(self, sample_ml_data: pd.DataFrame) -> None:
        """Test feature engineering function."""
        from models.utils import engineer_features

        horizons = [5, 10]
        result_df, predictors = engineer_features(sample_ml_data.copy(), horizons)

        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(predictors, list)
        assert len(result_df) < len(sample_ml_data)
        assert len(predictors) == 4

    def test_calculate_metrics(self, sample_ml_data: pd.DataFrame) -> None:
        """Test metrics calculation."""
        from models.utils import calculate_metrics

        predictions = pd.DataFrame({
            "Target": sample_ml_data["Target"],
            "Predictions": sample_ml_data["Target"].shift(1).fillna(0).astype(int),
        })

        metrics = calculate_metrics(predictions)

        assert hasattr(metrics, "accuracy")
        assert hasattr(metrics, "precision")
        assert hasattr(metrics, "recall")
        assert hasattr(metrics, "f1_score")
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1

    def test_calculate_metrics_edge_cases(self) -> None:
        """Test metrics calculation with edge cases."""
        from models.utils import calculate_metrics

        predictions_all_zeros = pd.DataFrame({
            "Target": [0, 0, 0, 0],
            "Predictions": [0, 0, 0, 0],
        })

        metrics = calculate_metrics(predictions_all_zeros)

        assert metrics.accuracy == 1.0
        assert metrics.precision == 0.0

    def test_print_benchmark_table(self, sample_ml_data: pd.DataFrame, capsys) -> None:
        """Test benchmark table printing."""
        from models.utils import print_benchmark_table

        predictions = pd.DataFrame({
            "Target": sample_ml_data["Target"],
            "Predictions": sample_ml_data["Target"].astype(int),
        })

        print_benchmark_table(predictions, "Test Model")

        captured = capsys.readouterr()
        assert "Test Model" in captured.out
        assert "Accuracy" in captured.out

    def test_print_benchmark_table_extended(
        self, sample_ml_data: pd.DataFrame, capsys
    ) -> None:
        """Test extended benchmark table printing."""
        from models.utils import print_benchmark_table

        predictions = pd.DataFrame({
            "Target": sample_ml_data["Target"],
            "Predictions": sample_ml_data["Target"].astype(int),
        })

        print_benchmark_table(predictions, "Test Model", show_extended=True)

        captured = capsys.readouterr()
        assert "F1 Score" in captured.out
        assert "ROC-AUC" in captured.out


class TestRunBacktest:
    """Tests for backtest functionality."""

    @pytest.fixture
    def sample_ml_data(self) -> pd.DataFrame:
        """Create sample data for ML testing."""
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=500, freq="D")
        data = {
            "Close": np.cumsum(np.random.randn(500)) + 100,
            "Target": np.random.randint(0, 2, 500),
            "feature1": np.random.randn(500),
            "feature2": np.random.randn(500),
        }
        return pd.DataFrame(data, index=dates)

    def test_run_backtest_structure(self, sample_ml_data: pd.DataFrame) -> None:
        """Test backtest returns correct structure."""
        from models.utils import run_backtest
        from sklearn.linear_model import LogisticRegression

        df = sample_ml_data.dropna()
        predictors = ["feature1", "feature2"]
        model = LogisticRegression()

        results = run_backtest(df, model, predictors, start=350, step=50)

        assert "Target" in results.columns
        assert "Predictions" in results.columns
        assert len(results) <= 150

    def test_run_backtest_threshold(self, sample_ml_data: pd.DataFrame) -> None:
        """Test backtest with custom threshold."""
        from models.utils import run_backtest
        from sklearn.linear_model import LogisticRegression

        df = sample_ml_data.dropna()
        predictors = ["feature1", "feature2"]
        model = LogisticRegression()

        results = run_backtest(
            df, model, predictors, start=350, step=50, threshold=0.3
        )

        assert "Predictions" in results.columns


class TestModelConfigs:
    """Tests for model configurations."""

    def test_random_forest_config(self) -> None:
        """Test Random Forest configuration."""
        from models.random_forest import MODEL_PARAMS, TREND_HORIZONS

        assert "n_estimators" in MODEL_PARAMS
        assert "max_depth" in MODEL_PARAMS
        assert isinstance(TREND_HORIZONS, list)

    def test_xgboost_config(self) -> None:
        """Test XGBoost configuration."""
        from models.xgboost_model import MODEL_PARAMS, TREND_HORIZONS

        assert "n_estimators" in MODEL_PARAMS
        assert "max_depth" in MODEL_PARAMS
        assert isinstance(TREND_HORIZONS, list)

    def test_logistic_regression_config(self) -> None:
        """Test Logistic Regression configuration."""
        from models.logistic_regression import MODEL_PARAMS, TREND_HORIZONS

        assert "penalty" in MODEL_PARAMS
        assert isinstance(TREND_HORIZONS, list)

    def test_baseline_config(self) -> None:
        """Test baseline model configuration."""
        from models.baseline_naive import DECISION_THRESHOLD, BACKTEST_STEP

        assert 0 <= DECISION_THRESHOLD <= 1
        assert BACKTEST_STEP > 0

    def test_arima_config(self) -> None:
        """Test ARIMA configuration."""
        from models.arima import MODEL_PARAMS

        assert "order" in MODEL_PARAMS
        assert isinstance(MODEL_PARAMS["order"], tuple)
