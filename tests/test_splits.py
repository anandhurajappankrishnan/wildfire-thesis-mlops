"""Tests for grouped train/test splits (no shared locations)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ml_eval import location_keys, spatial_temporal_split  # noqa: E402


def _synthetic_df(n_locs: int = 20, dates_per_loc: int = 10) -> pd.DataFrame:
    rows = []
    for i in range(n_locs):
        lat, lon = 38.0 + i * 0.01, -120.0 + i * 0.01
        for d in range(dates_per_loc):
            rows.append(
                {
                    "obs_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=7 * d),
                    "latitude": lat,
                    "longitude": lon,
                    "country": "usa",
                    "region_label": "California (USA)",
                    "fire_within_7d": (i + d) % 7 == 0,
                    "ndvi": 0.3,
                    "evi": 0.2,
                    "temperature_2m": 25.0,
                    "total_precipitation": 0.0,
                    "dewpoint_temperature_2m": 10.0,
                    "ndvi_lag7": 0.28,
                    "temp_7d_mean": 24.0,
                    "precip_7d_sum": 0.0,
                    "ndvi_delta7": 0.02,
                    "day_of_year": 180,
                    "season_idx": 1,
                }
            )
    return pd.DataFrame(rows)


def test_train_test_do_not_share_locations():
    df = _synthetic_df()
    feature_cols = [
        "ndvi",
        "evi",
        "temperature_2m",
        "total_precipitation",
        "dewpoint_temperature_2m",
        "ndvi_lag7",
        "temp_7d_mean",
        "precip_7d_sum",
        "ndvi_delta7",
        "day_of_year",
        "season_idx",
    ]
    _, _, _, _, test_meta, test_loc_keys = spatial_temporal_split(df, feature_cols, 0.25, seed=42)
    from ml_eval import add_location_id

    df = add_location_id(df.sort_values("obs_date"))
    split_date = test_meta["obs_date"].min()
    train_used_locs = set(
        df[(df["obs_date"] < split_date) & (~df["loc_key"].isin(test_loc_keys))]["loc_key"]
    )
    test_used_locs = set(test_meta["loc_key"])
    assert train_used_locs.isdisjoint(test_used_locs), "Train and test must not share lat/lon locations"
    assert len(test_loc_keys) > 0
