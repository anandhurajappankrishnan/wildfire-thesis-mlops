"""
Silver layer: feature engineering and 7-day fire target label.

Builds rolling weather features, vegetation lags, seasonality, and
fire_within_7d from FIRMS (preferred) or forward-looking MCD64A1 burn flags.

Groups by cell_id when present (areal cells), else lat/lon for legacy extracts.

Input:  data/bronze/bronze_satellite_raw.parquet
Output: data/silver/silver_features_clean.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]

FEATURE_FULL = [
    "ndvi",
    "evi",
    "temperature_2m",
    "total_precipitation",
    "dewpoint_temperature_2m",
    "wind_speed",
    "relative_humidity",
    "vapor_pressure_deficit",
    "elevation",
    "slope",
    "ndvi_lag7",
    "temp_7d_mean",
    "precip_7d_sum",
    "low_precip_days_7d",
    "ndvi_delta7",
    "day_of_year",
    "season_idx",
]


def magnus_vapor_pressure_kpa(temp_c: pd.Series) -> pd.Series:
    """Saturation vapor pressure (kPa) via Magnus/Tetens."""
    return 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))


def compute_relative_humidity(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    es = magnus_vapor_pressure_kpa(temp_c)
    ea = magnus_vapor_pressure_kpa(dewpoint_c)
    return (100.0 * ea / es).clip(0.0, 100.0)


def compute_vpd(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    es = magnus_vapor_pressure_kpa(temp_c)
    ea = magnus_vapor_pressure_kpa(dewpoint_c)
    return (es - ea).clip(lower=0.0)


def forward_fire_within_horizon(group: pd.DataFrame, horizon_days: int) -> pd.Series:
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


def group_columns(df: pd.DataFrame) -> list[str]:
    if "cell_id" in df.columns and df["cell_id"].notna().any():
        return ["country", "cell_id"]
    return ["country", "latitude", "longitude"]


def print_modeling_sanity(df: pd.DataFrame, log_dir: Path | None = None) -> dict:
    """Report retention and lag coverage — mirrors step4 dropna cohort."""
    gcols = group_columns(df)
    unit = "cell" if "cell_id" in gcols else "location"
    loc_obs = df.groupby(gcols).size()
    pre_pos = float(df["fire_within_7d"].mean()) if "fire_within_7d" in df.columns else float("nan")
    model_df = df.dropna(subset=FEATURE_FULL + ["fire_within_7d"])
    post_pos = float(model_df["fire_within_7d"].mean()) if len(model_df) else float("nan")
    pos_after = int(model_df["fire_within_7d"].sum()) if len(model_df) else 0
    stats = {
        "total_silver_rows": len(df),
        "rows_after_dropna": len(model_df),
        "retention_fraction": len(model_df) / max(len(df), 1),
        "positive_rate_before_dropna": pre_pos,
        "positive_rate_after_dropna": post_pos,
        "positive_count_after_dropna": pos_after,
        f"unique_{unit}s": int(loc_obs.shape[0]),
        f"obs_per_{unit}_min": int(loc_obs.min()) if len(loc_obs) else 0,
        f"obs_per_{unit}_mean": float(loc_obs.mean()) if len(loc_obs) else 0.0,
        f"obs_per_{unit}_max": int(loc_obs.max()) if len(loc_obs) else 0,
        f"{unit}s_with_lag_computable": int((loc_obs > 1).sum()),
        "ndvi_lag7_nonnull": int(df["ndvi_lag7"].notna().sum()) if "ndvi_lag7" in df.columns else 0,
    }
    print("\n=== MODELING SANITY (step4 FEATURE_FULL dropna) ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}" if "rate" in k or "fraction" in k or "mean" in k else f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")
    if len(model_df) and "region_label" in model_df.columns:
        region_pos = (
            model_df.groupby("region_label")["fire_within_7d"]
            .agg(["count", "sum"])
            .rename(columns={"count": "rows", "sum": "positives"})
            .astype({"positives": int})
        )
        print("\n  Per-region positives (after dropna):")
        print(region_pos.to_string())
        if log_dir is not None:
            region_pos.to_csv(log_dir / "sanity_per_region.csv")
    null_pct = {col: 100.0 * df[col].isna().mean() for col in FEATURE_FULL if col in df.columns}
    if null_pct:
        print("\n  Per-feature null % (silver, before dropna):")
        for col, pct in null_pct.items():
            print(f"    {col}: {pct:.2f}%")
        if log_dir is not None:
            pd.DataFrame([null_pct]).to_csv(log_dir / "sanity_feature_null_pct.csv", index=False)
    print("=" * 48)
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build silver features from bronze parquet")
    p.add_argument("--bronze-path", type=str, default=None)
    p.add_argument("--silver-path", type=str, default=None)
    p.add_argument("--sanity-report", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    horizon_days = int(cfg["data"].get("prediction_horizon_days", 7))
    stride_days = int(cfg["data"].get("gee", {}).get("temporal_stride_days", 7))
    lag_steps = max(1, round(7 / stride_days))

    bronze_path = Path(args.bronze_path) if args.bronze_path else ROOT / cfg["paths"]["bronze_dir"] / "bronze_satellite_raw.parquet"
    silver_path = Path(args.silver_path) if args.silver_path else ROOT / cfg["paths"]["silver_dir"] / "silver_features_clean.parquet"
    silver_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(bronze_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"])
    for col in [
        "ndvi", "evi", "temperature_2m", "total_precipitation",
        "dewpoint_temperature_2m", "wind_speed", "elevation", "slope",
        "burned_area", "firms_fire_7d",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "country" not in df.columns:
        df["country"] = "portugal"
    if "region_label" not in df.columns:
        df["region_label"] = df["country"].str.replace("_", " ").str.title()

    df = df.dropna(subset=["obs_date"]).copy()
    if "cell_id" not in df.columns or df["cell_id"].isna().all():
        df = df.dropna(subset=["latitude", "longitude"])
        df["latitude"] = df["latitude"].round(5)
        df["longitude"] = df["longitude"].round(5)

    gcols = group_columns(df)
    df = df.sort_values(gcols + ["obs_date"]).reset_index(drop=True)

    df["burn_binary"] = (df["burned_area"].fillna(0) > 0.5).astype(int)
    g = df.groupby(gcols, group_keys=False)

    df["ndvi_lag7"] = g["ndvi"].shift(lag_steps)
    weather_window = max(2, round(7 / stride_days))
    df["temp_7d_mean"] = g["temperature_2m"].transform(
        lambda x: x.rolling(weather_window, min_periods=1).mean()
    )
    df["precip_7d_sum"] = g["total_precipitation"].transform(
        lambda x: x.rolling(weather_window, min_periods=1).sum()
    )
    df["relative_humidity"] = compute_relative_humidity(
        df["temperature_2m"], df["dewpoint_temperature_2m"]
    )
    df["vapor_pressure_deficit"] = compute_vpd(
        df["temperature_2m"], df["dewpoint_temperature_2m"]
    )
    df["_precip_low"] = (df["total_precipitation"].fillna(0) < 0.001).astype(float)
    df["low_precip_days_7d"] = g["_precip_low"].transform(
        lambda x: x.rolling(weather_window, min_periods=1).sum()
    )
    df.drop(columns=["_precip_low"], inplace=True)
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
    print(df.groupby("region_label")["fire_within_7d"].agg(["count", "sum"]).to_string())

    if args.sanity_report:
        stats = print_modeling_sanity(df, log_dir=silver_path.parent)
        pd.DataFrame([stats]).to_csv(silver_path.parent / "sanity_report.csv", index=False)
        print(f"Sanity log: {silver_path.parent / 'sanity_report.csv'}")

    df.to_parquet(silver_path, index=False)
    print(f"Saved: {silver_path}")


if __name__ == "__main__":
    main()
