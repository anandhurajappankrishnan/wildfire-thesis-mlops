"""
Bronze layer: persist raw GEE CSV as typed Parquet with ingestion timestamp.

Input:  data/raw/gee_modis_era5_burnedarea.csv
Output: data/bronze/bronze_satellite_raw.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from env_setup import load_project_env  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bronze parquet from raw GEE CSV")
    p.add_argument("--raw-path", type=str, default=None, help="Override raw CSV path")
    p.add_argument("--bronze-path", type=str, default=None, help="Override bronze parquet path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    load_project_env()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))

    raw_path = Path(args.raw_path) if args.raw_path else ROOT / cfg["paths"]["raw_dir"] / "gee_modis_era5_burnedarea.csv"
    bronze_path = Path(args.bronze_path) if args.bronze_path else ROOT / cfg["paths"]["bronze_dir"] / "bronze_satellite_raw.parquet"
    bronze_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date
    df["ingested_at"] = pd.Timestamp.utcnow()
    df.to_parquet(bronze_path, index=False)
    print(f"Saved: {bronze_path}")


if __name__ == "__main__":
    main()
