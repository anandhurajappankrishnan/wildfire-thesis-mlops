"""
Main pipeline orchestrator for the wildfire thesis MLOps project.

Author: Anandhu R Krishnan
Runs the medallion workflow: GEE extract → Bronze → Silver → Gold → thesis outputs.

Usage:
    python src/pipeline_real.py          # quick mode if raw CSV exists (~1 min)
    python src/pipeline_real.py --quick  # same as above
    python src/pipeline_real.py --full   # re-download from GEE (~10–20 min)

See config.yaml for dates, regions (config/study_areas.geojson), and model params.
"""
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml


def append_step_log(log_path: Path, row: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run_id", "step_name", "started_at", "duration_sec", "status"],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def run_step(name: str, cmd: list[str], cwd: Path, run_id: str, log_path: Path) -> None:
    """Run a pipeline step subprocess, record timing, and fail fast on error."""
    print(f"[STEP] {name}")
    started = datetime.now(timezone.utc).isoformat()
    t0 = time.perf_counter()
    status = "success"
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError:
        status = "failure"
        append_step_log(
            log_path,
            {
                "run_id": run_id,
                "step_name": name,
                "started_at": started,
                "duration_sec": round(time.perf_counter() - t0, 3),
                "status": status,
            },
        )
        raise
    append_step_log(
        log_path,
        {
            "run_id": run_id,
            "step_name": name,
            "started_at": started,
            "duration_sec": round(time.perf_counter() - t0, 3),
            "status": status,
        },
    )


def write_latest_timing(cfg: dict, root: Path, run_id: str, log_path: Path) -> None:
    """Summarize the current run's step durations for thesis RQ1 tables."""
    if not log_path.exists():
        return
    import pandas as pd

    df = pd.read_csv(log_path)
    run_df = df[df["run_id"] == run_id]
    if run_df.empty:
        return
    gold_dir = root / cfg["paths"]["gold_dir"]
    gold_dir.mkdir(parents=True, exist_ok=True)
    timing = run_df[["step_name", "duration_sec"]].copy()
    timing["duration_min"] = timing["duration_sec"] / 60.0
    timing.to_csv(gold_dir / "pipeline_timing_latest.csv", index=False)


def raw_data_exists(root: Path, cfg: dict) -> bool:
    """True if a non-empty GEE raw CSV is already cached locally."""
    raw_path = root / cfg["paths"]["raw_dir"] / "gee_modis_era5_burnedarea.csv"
    return raw_path.is_file() and raw_path.stat().st_size > 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildfire thesis MLOps pipeline")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Re-download all data from Google Earth Engine (~10–20 min)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip GEE extraction and reuse cached data in data/raw/ (~1 min)",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with (root / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    pipeline_cfg = cfg.get("pipeline", {})
    skip_gee_default = bool(pipeline_cfg.get("skip_gee_if_raw_exists", True))
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = root / cfg["paths"]["gold_dir"] / "pipeline_step_runs.csv"

    if args.full:
        run_gee = True
    elif args.quick:
        run_gee = False
    else:
        run_gee = not (skip_gee_default and raw_data_exists(root, cfg))

    python = sys.executable
    steps: list[tuple[str, list[str]]] = []

    if run_gee:
        print("Mode: FULL (GEE extraction enabled — expect ~10–20 min for 3 regions)")
        steps.append(("Extract GEE data", [python, "src/steps/step1_extract_gee.py"]))
    else:
        raw_path = root / cfg["paths"]["raw_dir"] / "gee_modis_era5_burnedarea.csv"
        if not raw_path.is_file():
            raise SystemExit(
                "No cached raw data found. Run once with: python src/pipeline_real.py --full"
            )
        print(f"Mode: QUICK (skipping GEE — reusing {raw_path.name})")

    steps.extend(
        [
            ("Build Bronze layer", [python, "src/steps/step2_bronze.py"]),
            ("Load Bronze to SQL (optional)", [python, "src/steps/step2b_load_sql.py"]),
            ("Build Silver layer", [python, "src/steps/step3_silver.py"]),
            ("Train models + evaluate", [python, "src/steps/step4_train_eval.py"]),
            ("Generate RQ figures/tables", [python, "src/steps/step5_generate_thesis_outputs.py"]),
        ]
    )

    for name, cmd in steps:
        run_step(name, cmd, root, run_id, log_path)

    write_latest_timing(cfg, root, run_id, log_path)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
