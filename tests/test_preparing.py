"""Tests for the data preparation module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class TestDataCleaning:
    """Tests for data cleaning functions."""

    @pytest.fixture
    def sample_stock_data(self) -> pd.DataFrame:
        """Create sample stock data for testing."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        data = {
            "Date": dates,
            "Open": np.random.uniform(100, 110, 100),
            "High": np.random.uniform(110, 120, 100),
            "Low": np.random.uniform(90, 100, 100),
            "Close": np.random.uniform(100, 110, 100),
            "Volume": np.random.randint(1000000, 10000000, 100),
        }
        df = pd.DataFrame(data)
        df["High"] = df[["Open", "Close", "High"]].max(axis=1) + 1
        df["Low"] = df[["Open", "Close", "Low"]].min(axis=1) - 1
        return df

    def test_get_missing_values_info(self, sample_stock_data: pd.DataFrame) -> None:
        """Test missing values detection."""
        from preparing.data_cleaning import get_missing_values_info

        result = get_missing_values_info(sample_stock_data)

        assert result["total_missing"] == 0
        assert not result["has_missing"]
        assert all(count == 0 for count in result["count"])

    def test_get_missing_values_info_with_nans(self) -> None:
        """Test missing values detection with NaN values."""
        from preparing.data_cleaning import get_missing_values_info

        df = pd.DataFrame({
            "Date": pd.date_range("2020-01-01", periods=5),
            "Open": [1.0, 2.0, np.nan, 4.0, 5.0],
            "Close": [1.1, 2.1, 3.1, 4.1, 5.1],
        })

        result = get_missing_values_info(df)

        assert result["has_missing"]
        assert result["total_missing"] == 1

    def test_validate_data_types(self, sample_stock_data: pd.DataFrame) -> None:
        """Test data type validation."""
        from preparing.data_cleaning import validate_data_types

        mismatches = validate_data_types(sample_stock_data)

        assert len(mismatches) == 0

    def test_validate_data_types_mismatch(self) -> None:
        """Test data type validation with mismatched types."""
        from preparing.data_cleaning import validate_data_types

        df = pd.DataFrame({
            "Date": ["2020-01-01"] * 5,
            "Open": [1.0] * 5,
            "Close": [1.1] * 5,
        })

        mismatches = validate_data_types(df)

        assert "Date" in mismatches

    def test_detect_outliers_iqr(self, sample_stock_data: pd.DataFrame) -> None:
        """Test IQR-based outlier detection."""
        from preparing.data_cleaning import detect_outliers_iqr

        outliers = detect_outliers_iqr(sample_stock_data)

        assert isinstance(outliers, pd.DataFrame)

    def test_detect_outliers_rolling_iqr(self, sample_stock_data: pd.DataFrame) -> None:
        """Test rolling IQR-based outlier detection."""
        from preparing.data_cleaning import detect_outliers_rolling_iqr

        result_df = detect_outliers_rolling_iqr(sample_stock_data)

        assert "Close_outlier" in result_df.columns
        assert isinstance(result_df, pd.DataFrame)

    def test_get_data_quality_report(self, sample_stock_data: pd.DataFrame) -> None:
        """Test data quality report generation."""
        from preparing.data_cleaning import get_data_quality_report

        report = get_data_quality_report(sample_stock_data)

        assert hasattr(report, "is_clean")
        assert hasattr(report, "missing_values")
        assert hasattr(report, "type_mismatches")

    def test_clean_data(self, sample_stock_data: pd.DataFrame) -> None:
        """Test data cleaning function."""
        from preparing.data_cleaning import clean_data

        cleaned = clean_data(sample_stock_data, remove_outliers=True)

        assert isinstance(cleaned, pd.DataFrame)
        assert "Date" in cleaned.index.names
        assert len(cleaned) > 0

    def test_clean_data_inplace(self, sample_stock_data: pd.DataFrame) -> None:
        """Test in-place data cleaning."""
        from preparing.data_cleaning import clean_data

        original_len = len(sample_stock_data)
        cleaned = clean_data(sample_stock_data, remove_outliers=False, inplace=False)

        assert len(sample_stock_data) == original_len
        assert len(cleaned) <= original_len


class TestFeatureEngineering:
    """Tests for feature engineering functions."""

    @pytest.fixture
    def sample_processed_data(self) -> pd.DataFrame:
        """Create sample processed stock data for testing."""
        dates = pd.date_range("2020-01-01", periods=300, freq="D")
        data = {
            "Open": np.random.uniform(100, 110, 300),
            "High": np.random.uniform(110, 120, 300),
            "Low": np.random.uniform(90, 100, 300),
            "Close": np.random.uniform(100, 110, 300),
            "Volume": np.random.randint(1000000, 10000000, 300),
        }
        df = pd.DataFrame(data, index=dates)
        df["High"] = df[["Open", "Close"]].max(axis=1) + np.random.uniform(1, 5, 300)
        df["Low"] = df[["Open", "Close"]].min(axis=1) - np.random.uniform(1, 5, 300)
        return df

    def test_validate_datetime_index(self, sample_processed_data: pd.DataFrame) -> None:
        """Test datetime index validation."""
        from preparing.feature_engineering import validate_datetime_index

        result = validate_datetime_index(sample_processed_data)

        assert result is True

    def test_validate_datetime_index_invalid(self) -> None:
        """Test datetime index validation with invalid index."""
        from preparing.feature_engineering import validate_datetime_index

        df = pd.DataFrame({"Close": [1, 2, 3]})
        result = validate_datetime_index(df)

        assert result is False

    def test_add_base_features(self, sample_processed_data: pd.DataFrame) -> None:
        """Test base features addition."""
        from preparing.feature_engineering import add_base_features

        result = add_base_features(sample_processed_data)

        assert "returns" in result.columns
        assert "RSI" in result.columns
        assert "MACD" in result.columns
        assert "BB_Position" in result.columns
        assert "Target" in result.columns

    def test_add_rolling_horizon_features(self, sample_processed_data: pd.DataFrame) -> None:
        """Test rolling horizon features addition."""
        from preparing.feature_engineering import add_rolling_horizon_features

        df = sample_processed_data.copy()
        df["Target"] = (df["Close"] > df["Close"].shift(1)).astype(int)
        horizons = [5, 10, 20]
        result = add_rolling_horizon_features(df, horizons)

        assert "Close_Ratio_5" in result.columns
        assert "Close_Ratio_10" in result.columns
        assert "Close_Ratio_20" in result.columns
        assert "Trend_5" in result.columns

    def test_add_calendar_features(self, sample_processed_data: pd.DataFrame) -> None:
        """Test calendar features addition."""
        from preparing.feature_engineering import add_calendar_features

        result = add_calendar_features(sample_processed_data)

        assert "Day_of_Week" in result.columns
        assert "Month" in result.columns
        assert result["Day_of_Week"].min() >= 0
        assert result["Day_of_Week"].max() <= 6
        assert result["Month"].min() >= 1
        assert result["Month"].max() <= 12

    def test_add_features_complete_pipeline(
        self, sample_processed_data: pd.DataFrame
    ) -> None:
        """Test complete feature engineering pipeline."""
        from preparing.feature_engineering import add_features

        result = add_features(sample_processed_data)

        assert "returns" in result.columns
        assert "RSI" in result.columns
        assert "MACD" in result.columns
        assert "Target" in result.columns
        assert len(result) < len(sample_processed_data)


class TestDataCollection:
    """Tests for data collection functions."""

    def test_stocks_constant_defined(self) -> None:
        """Test that STOCKS constant is properly defined."""
        from preparing.data_collection import STOCKS

        assert isinstance(STOCKS, list)
        assert len(STOCKS) > 0
        assert all(isinstance(s, str) for s in STOCKS)
        assert "AAPL" in STOCKS
