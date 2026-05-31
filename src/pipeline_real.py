"""
Main pipeline orchestrator for the wildfire thesis MLOps project.

Runs the medallion workflow: GEE extract → Bronze → Silver → Gold → thesis outputs.

Usage:
    python src/pipeline_real.py          # quick mode if raw CSV exists (~1 min)
    python src/pipeline_real.py --quick  # same as above
    python src/pipeline_real.py --full   # re-download from GEE (~10–20 min)

See config.yaml for dates, regions (config/study_areas.geojson), and model params.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def run_step(name: str, cmd: list[str], cwd: Path) -> None:
    """Run a pipeline step subprocess and fail fast on error."""
    print(f"[STEP] {name}")
    subprocess.run(cmd, cwd=cwd, check=True)


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

    if args.full:
        run_gee = True
    elif args.quick:
        run_gee = False
    else:
        run_gee = not (skip_gee_default and raw_data_exists(root, cfg))

    python = sys.executable

    if run_gee:
        print("Mode: FULL (GEE extraction enabled — expect ~10–20 min for 3 regions)")
        run_step("Extract GEE data", [python, "src/steps/step1_extract_gee.py"], root)
    else:
        raw_path = root / cfg["paths"]["raw_dir"] / "gee_modis_era5_burnedarea.csv"
        if not raw_path.is_file():
            raise SystemExit(
                "No cached raw data found. Run once with: python src/pipeline_real.py --full"
            )
        print(f"Mode: QUICK (skipping GEE — reusing {raw_path.name})")

    run_step("Build Bronze layer", [python, "src/steps/step2_bronze.py"], root)
    run_step("Load Bronze to SQL (optional)", [python, "src/steps/step2b_load_sql.py"], root)
    run_step("Build Silver layer", [python, "src/steps/step3_silver.py"], root)
    run_step("Train models + evaluate", [python, "src/steps/step4_train_eval.py"], root)
    run_step("Generate RQ figures/tables", [python, "src/steps/step5_generate_thesis_outputs.py"], root)
    print("Pipeline completed.")


if __name__ == "__main__":
    main()
