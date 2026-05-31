"""
Bronze layer: persist raw GEE CSV as typed Parquet with ingestion timestamp.

Input:  data/raw/gee_modis_era5_burnedarea.csv
Output: data/bronze/bronze_satellite_raw.parquet
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from env_setup import load_project_env  # noqa: E402


def main() -> None:
    load_project_env()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    raw_path = ROOT / cfg["paths"]["raw_dir"] / "gee_modis_era5_burnedarea.csv"
    bronze_dir = ROOT / cfg["paths"]["bronze_dir"]
    bronze_dir.mkdir(parents=True, exist_ok=True)
    bronze_path = bronze_dir / "bronze_satellite_raw.parquet"

    df = pd.read_csv(raw_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date
    df["ingested_at"] = pd.Timestamp.utcnow()
    df.to_parquet(bronze_path, index=False)
    print(f"Saved: {bronze_path}")


if __name__ == "__main__":
    main()
