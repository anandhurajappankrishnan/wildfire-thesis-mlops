"""Tests for temporal feature leakage in silver layer logic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _build_silver_features(df: pd.DataFrame, stride_days: int = 7) -> pd.DataFrame:
    """Mirror step3_silver feature logic for unit testing."""
    lag_steps = max(1, round(7 / stride_days))
    weather_window = max(2, round(7 / stride_days))
    gcols = ["country", "latitude", "longitude"]
    g = df.groupby(gcols, group_keys=False)
    df = df.sort_values(gcols + ["obs_date"]).reset_index(drop=True)
    df["ndvi_lag7"] = g["ndvi"].shift(lag_steps)
    df["temp_7d_mean"] = g["temperature_2m"].transform(
        lambda x: x.rolling(weather_window, min_periods=1).mean()
    )
    df["precip_7d_sum"] = g["total_precipitation"].transform(
        lambda x: x.rolling(weather_window, min_periods=1).sum()
    )
    df["ndvi_delta7"] = df["ndvi"] - df["ndvi_lag7"]
    return df


def test_rolling_features_use_only_past_and_current_rows():
    dates = pd.date_range("2024-01-01", periods=5, freq="7D")
    df = pd.DataFrame(
        {
            "country": ["pt"] * 5,
            "latitude": [40.0] * 5,
            "longitude": [-8.0] * 5,
            "obs_date": dates,
            "ndvi": [0.1, 0.2, 0.3, 0.4, 0.5],
            "temperature_2m": [10, 20, 30, 40, 50],
            "total_precipitation": [1, 2, 3, 4, 5],
        }
    )
    out = _build_silver_features(df)
    # Row index 2: temp_7d_mean should be mean of rows 0..2 (10,20,30) = 20, not include 40,50
    assert out.loc[2, "temp_7d_mean"] == 20.0
    assert out.loc[2, "precip_7d_sum"] == 6.0
    # ndvi_lag7 at row 2 uses row 0 (lag 2 steps with stride 7)
    assert out.loc[2, "ndvi_lag7"] == 0.1
    assert out.loc[2, "ndvi_delta7"] == 0.2


def test_no_future_ndvi_in_lag():
    dates = pd.date_range("2024-06-01", periods=4, freq="7D")
    df = pd.DataFrame(
        {
            "country": ["au"] * 4,
            "latitude": [-36.0] * 4,
            "longitude": [146.0] * 4,
            "obs_date": dates,
            "ndvi": [0.5, 0.6, 0.9, 0.1],
            "temperature_2m": [15] * 4,
            "total_precipitation": [0] * 4,
        }
    )
    out = _build_silver_features(df)
    # At row 1, lag must be row 0 (0.5), not row 2 (0.9)
    assert out.loc[1, "ndvi_lag7"] == 0.5
