"""
Optional: load Bronze parquet into SQL Server dbo.bronze_satellite_raw.
Set SQLSERVER_CONN_STR in .env (see .env.example). Skips if unset or connection fails.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from env_setup import load_project_env  # noqa: E402


def main() -> None:
    load_project_env()
    conn = os.getenv("SQLSERVER_CONN_STR")
    if not conn:
        print("SQLSERVER_CONN_STR not set — skipping SQL load.")
        return

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    try:
        import pyodbc
    except ImportError:
        print("pyodbc not installed — skipping SQL load.")
        return

    bronze_path = ROOT / cfg["paths"]["bronze_dir"] / "bronze_satellite_raw.parquet"
    df = pd.read_parquet(bronze_path)
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date

    cols = [
        "obs_date",
        "latitude",
        "longitude",
        "ndvi",
        "evi",
        "temperature_2m",
        "total_precipitation",
        "dewpoint_temperature_2m",
        "burned_area",
        "source_system",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    insert_df = df[cols].copy()
    now = pd.Timestamp.utcnow()

    try:
        with pyodbc.connect(conn, timeout=30) as cx:
            cur = cx.cursor()
            cur.fast_executemany = True
            sql = """
            INSERT INTO dbo.bronze_satellite_raw
            (obs_date, latitude, longitude, ndvi, evi, temperature_2m, total_precipitation,
             dewpoint_temperature_2m, burned_area, source_system, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            batch = []
            for row in insert_df.itertuples(index=False):
                batch.append(
                    (
                        row.obs_date,
                        float(row.latitude),
                        float(row.longitude),
                        row.ndvi,
                        row.evi,
                        row.temperature_2m,
                        row.total_precipitation,
                        row.dewpoint_temperature_2m,
                        row.burned_area,
                        str(row.source_system) if row.source_system is not None else "GEE",
                        now,
                    )
                )
            cur.executemany(sql, batch)
            cx.commit()
        print(f"Inserted {len(batch)} rows into dbo.bronze_satellite_raw.")
    except Exception as exc:
        print(f"SQL load skipped or failed: {exc}")


if __name__ == "__main__":
    main()
