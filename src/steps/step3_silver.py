"""
Silver layer: feature engineering and 7-day fire target label.

Builds rolling weather features, vegetation lags, seasonality, and
fire_within_7d from FIRMS (preferred) or forward-looking MCD64A1 burn flags.

Input:  data/bronze/bronze_satellite_raw.parquet
Output: data/silver/silver_features_clean.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]


def forward_fire_within_horizon(group: pd.DataFrame, horizon_days: int) -> pd.Series:
    """1 if any burn occurs within horizon_days after obs_date at the same location."""
    dates = group["obs_date"].values
    burns = group["burn_binary"].to_numpy(dtype=float)
    out = np.zeros(len(group), dtype=np.int64)
    for i in range(len(group)):
        t0 = pd.Timestamp(dates[i])
        for j in range(i + 1, len(group)):
            delta = (pd.Timestamp(dates[j]) - t0).days
            if delta > horizon_days:
                break
            if burns[j] > 0.5:
                out[i] = 1
                break
    return pd.Series(out, index=group.index)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    horizon_days = int(cfg["data"].get("prediction_horizon_days", 7))
    stride_days = int(cfg["data"].get("gee", {}).get("temporal_stride_days", 7))
    lag_steps = max(1, round(7 / stride_days))

    bronze_path = ROOT / cfg["paths"]["bronze_dir"] / "bronze_satellite_raw.parquet"
    silver_dir = ROOT / cfg["paths"]["silver_dir"]
    silver_dir.mkdir(parents=True, exist_ok=True)
    silver_path = silver_dir / "silver_features_clean.parquet"

    df = pd.read_parquet(bronze_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    for col in [
        "ndvi",
        "evi",
        "temperature_2m",
        "total_precipitation",
        "dewpoint_temperature_2m",
        "burned_area",
        "firms_fire_7d",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "country" not in df.columns:
        df["country"] = "portugal"
    if "region_label" not in df.columns:
        df["region_label"] = df["country"].str.replace("_", " ").str.title()

    df = df.dropna(subset=["latitude", "longitude", "obs_date"]).copy()
    df = df.sort_values(["country", "latitude", "longitude", "obs_date"]).reset_index(drop=True)

    df["burn_binary"] = (df["burned_area"].fillna(0) > 0.5).astype(int)

    gcols = ["country", "latitude", "longitude"]
    g = df.groupby(gcols, group_keys=False)

    df["ndvi_lag7"] = g["ndvi"].shift(lag_steps)
    weather_window = max(2, round(7 / stride_days))
    df["temp_7d_mean"] = g["temperature_2m"].transform(lambda x: x.rolling(weather_window, min_periods=1).mean())
    df["precip_7d_sum"] = g["total_precipitation"].transform(lambda x: x.rolling(weather_window, min_periods=1).sum())
    df["ndvi_delta7"] = df["ndvi"] - df["ndvi_lag7"]

    df["day_of_year"] = df["obs_date"].dt.dayofyear
    df["season_idx"] = (df["day_of_year"] // 91).astype(int)

    if "firms_fire_7d" in df.columns:
        df["fire_within_7d"] = (df["firms_fire_7d"].fillna(0) > 0.5).astype(int)
    else:
        label_parts = []
        for _, grp in df.groupby(gcols, sort=False):
            label_parts.append(forward_fire_within_horizon(grp, horizon_days))
        df["fire_within_7d"] = pd.concat(label_parts).sort_index()

    df = df.dropna(subset=["ndvi", "temperature_2m", "total_precipitation"]).copy()
    df = df.drop(columns=["burn_binary"], errors="ignore")

    pos = int(df["fire_within_7d"].sum())
    print(f"Silver: {len(df)} rows, {pos} positive fire_within_7d labels ({100 * pos / max(len(df), 1):.2f}%)")
    by_country = df.groupby("region_label")["fire_within_7d"].agg(["count", "sum"])
    print(by_country.to_string())

    df.to_parquet(silver_path, index=False)
    print(f"Saved: {silver_path}")


if __name__ == "__main__":
    main()
