"""
Smoke verification for fire-aware areal-cell GEE sampling.

Runs a representative SHORT extract into data/smoke/ only.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
sys.path.insert(0, str(ROOT / "src"))
from ml_eval import set_global_seed, spatial_temporal_split  # noqa: E402

# ~90 cells/region, 10 weekly dates, 3 regions
SMOKE = {
    "grid_points": 90,
    "start_date": "2023-08-01",
    "end_date": "2023-10-03",  # 9 weekly dates
    "raw_csv": ROOT / "data" / "smoke" / "raw" / "gee_modis_era5_burnedarea.csv",
    "bronze_parquet": ROOT / "data" / "smoke" / "bronze" / "bronze_satellite_raw.parquet",
    "silver_parquet": ROOT / "data" / "smoke" / "silver" / "silver_features_clean.parquet",
    "sanity_csv": ROOT / "data" / "smoke" / "silver" / "sanity_report.csv",
}


def run(cmd: list[str]) -> None:
    print(f"\n>> {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def project_full_run(cfg: dict, smoke_stats: dict, smoke_raw_rows: int) -> dict:
    """Scale smoke metrics to full config and estimate holdout test positives."""
    gee = cfg["data"]["gee"]
    grid = int(gee.get("grid_points_per_region", 300))
    stride = int(gee.get("temporal_stride_days", 7))
    start = pd.Timestamp(cfg["data"]["start_date"])
    end = pd.Timestamp(cfg["data"]["end_date"])
    full_dates = len(pd.date_range(start, end, freq=f"{stride}D"))
    smoke_dates = len(pd.date_range(SMOKE["start_date"], SMOKE["end_date"], freq=f"{stride}D"))
    n_regions = 3
    test_size = float(cfg["model"].get("test_size", 0.25))

    full_raw = grid * full_dates * n_regions
    scale_raw = full_raw / max(smoke_raw_rows, 1)

    silver_rows = smoke_stats.get("total_silver_rows", 0)
    model_rows = smoke_stats.get("rows_after_dropna", 0)
    pos_count = smoke_stats.get("positive_count_after_dropna", 0)
    retention = smoke_stats.get("retention_fraction", 0)
    pos_rate = smoke_stats.get("positive_rate_after_dropna", 0)

    silver_retention = silver_rows / max(smoke_raw_rows, 1)
    proj_silver = int(full_raw * silver_retention)
    proj_model = int(proj_silver * retention)
    proj_positives = int(round(proj_model * pos_rate))

    # Spatial-temporal holdout: ~25% locations x late-period dates (~7% of model rows observed historically)
    holdout_row_frac = 0.07
    proj_test_rows = max(1, int(proj_model * holdout_row_frac))
    proj_test_positives = int(round(pos_count * scale_raw * holdout_row_frac / max(retention, 1e-9)))
    # simpler: apply pos_rate to projected test rows
    proj_test_positives_alt = int(round(proj_test_rows * pos_rate))

    proj_region_positives: dict[str, int] = {}
    proj_region_holdout_positives: dict[str, int] = {}
    if SMOKE["sanity_csv"].exists():
        per_region_path = SMOKE["sanity_csv"].parent / "sanity_per_region.csv"
        if per_region_path.exists():
            region_df = pd.read_csv(per_region_path, index_col=0)
            for region, row in region_df.iterrows():
                region_smoke_pos = int(row["positives"])
                region_model_rows = int(row["rows"])
                region_pos_rate = region_smoke_pos / max(region_model_rows, 1)
                region_proj_model = int(
                    round((grid * full_dates) * silver_retention * retention)
                )
                region_proj_pos = int(round(region_proj_model * region_pos_rate))
                region_proj_test = int(round(region_proj_model * holdout_row_frac * region_pos_rate))
                proj_region_positives[str(region)] = region_proj_pos
                proj_region_holdout_positives[str(region)] = region_proj_test

    return {
        "full_grid_points_per_region": grid,
        "full_weekly_dates": full_dates,
        "full_raw_rows_est": full_raw,
        "proj_silver_rows_est": proj_silver,
        "proj_model_rows_after_dropna_est": proj_model,
        "proj_total_positives_est": proj_positives,
        "proj_holdout_test_rows_est": proj_test_rows,
        "proj_holdout_test_positives_est": proj_test_positives_alt,
        "smoke_to_full_raw_scale": scale_raw,
        "holdout_row_fraction_assumed": holdout_row_frac,
        "test_location_fraction": test_size,
        "proj_region_positives": proj_region_positives,
        "proj_region_holdout_positives": proj_region_holdout_positives,
    }


FEATURE_LEGACY = [
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


def smoke_xgb_signal_check(silver_path: Path, cfg: dict) -> dict:
    """Quick holdout XGB PR-AUC/ROC-AUC: legacy vs expanded feature set."""
    from steps.step3_silver import FEATURE_FULL

    df = pd.read_parquet(silver_path)
    if "obs_date" in df.columns:
        df["obs_date"] = pd.to_datetime(df["obs_date"])
    seed = int(cfg["project"]["seed"])
    test_size = float(cfg["model"]["test_size"])
    set_global_seed(seed)

    out = {}
    for label, cols in [("legacy", FEATURE_LEGACY), ("expanded", FEATURE_FULL)]:
        sub = df.dropna(subset=cols + ["fire_within_7d"]).copy()
        if len(sub) < 100 or sub["fire_within_7d"].nunique() < 2:
            out[label] = {"pr_auc": float("nan"), "roc_auc": float("nan"), "model_rows": len(sub)}
            continue
        X_train, X_test, y_train, y_test, _, _ = spatial_temporal_split(
            sub, cols, test_size, seed
        )
        if len(X_test) < 10 or y_test.nunique() < 2:
            out[label] = {"pr_auc": float("nan"), "roc_auc": float("nan"), "model_rows": len(sub)}
            continue
        pos = float(y_train.sum())
        neg = float(len(y_train) - pos)
        model = XGBClassifier(
            n_estimators=int(cfg["model"]["xgb_n_estimators"]),
            learning_rate=float(cfg["model"]["xgb_learning_rate"]),
            max_depth=int(cfg["model"]["xgb_max_depth"]),
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            scale_pos_weight=neg / max(pos, 1.0),
        )
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        out[label] = {
            "pr_auc": float(average_precision_score(y_test, y_prob)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)),
            "model_rows": len(sub),
            "test_rows": len(X_test),
            "test_positives": int(y_test.sum()),
        }
    return out


def main() -> None:
    for p in [SMOKE["raw_csv"].parent, SMOKE["bronze_parquet"].parent, SMOKE["silver_parquet"].parent]:
        p.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SMOKE TEST — fire-aware areal cells (data/smoke/ only)")
    print(
        f"  grid_points={SMOKE['grid_points']}, dates {SMOKE['start_date']} to {SMOKE['end_date']}"
    )
    print("=" * 60)

    run(
        [
            PYTHON,
            "src/steps/step1_extract_gee.py",
            "--output-csv",
            str(SMOKE["raw_csv"]),
            "--grid-points",
            str(SMOKE["grid_points"]),
            "--start-date",
            SMOKE["start_date"],
            "--end-date",
            SMOKE["end_date"],
        ]
    )

    run(
        [
            PYTHON,
            "src/steps/step2_bronze.py",
            "--raw-path",
            str(SMOKE["raw_csv"]),
            "--bronze-path",
            str(SMOKE["bronze_parquet"]),
        ]
    )

    run(
        [
            PYTHON,
            "src/steps/step3_silver.py",
            "--bronze-path",
            str(SMOKE["bronze_parquet"]),
            "--silver-path",
            str(SMOKE["silver_parquet"]),
            "--sanity-report",
        ]
    )

    smoke_raw_rows = len(pd.read_csv(SMOKE["raw_csv"]))
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    smoke_stats = pd.read_csv(SMOKE["sanity_csv"]).iloc[0].to_dict() if SMOKE["sanity_csv"].exists() else {}

    null_path = SMOKE["sanity_csv"].parent / "sanity_feature_null_pct.csv"
    if null_path.exists():
        print("\n" + "=" * 60)
        print("NEW FEATURE NULL % (silver, before dropna)")
        print("=" * 60)
        null_row = pd.read_csv(null_path).iloc[0].to_dict()
        for col, pct in null_row.items():
            print(f"  {col}: {pct:.2f}%")

    signal = smoke_xgb_signal_check(SMOKE["silver_parquet"], cfg)
    print("\n" + "=" * 60)
    print("XGB HOLDOUT SIGNAL CHECK (smoke slice, same split)")
    print("=" * 60)
    for label, metrics in signal.items():
        print(f"  {label}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}" if k.endswith("_auc") else f"    {k}: {v:.1f}")
            else:
                print(f"    {k}: {v}")

    proj = project_full_run(cfg, smoke_stats, smoke_raw_rows)

    print("\n" + "=" * 60)
    print("FULL-RUN PROJECTION (from smoke scale-up)")
    print("=" * 60)
    for k, v in proj.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for rk, rv in v.items():
                print(f"    {rk}: {rv}")
        elif isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    pos_rate_pct = 100 * smoke_stats.get("positive_rate_after_dropna", 0)
    retention_pct = 100 * smoke_stats.get("retention_fraction", 0)
    test_pos = proj["proj_holdout_test_positives_est"]
    go_pos = 30 <= test_pos <= 500  # upper bound sanity
    go_rate = 2 <= pos_rate_pct <= 15
    go_ret = retention_pct >= 60

    print("\n" + "=" * 60)
    print("GO / NO-GO")
    print("=" * 60)
    print(f"  Retention >= 60%:           {'PASS' if go_ret else 'FAIL'} ({retention_pct:.1f}%)")
    print(f"  Positive rate 2-15%:        {'PASS' if go_rate else 'FAIL'} ({pos_rate_pct:.2f}%)")
    print(f"  Proj. holdout positives >= 30: {'PASS' if test_pos >= 30 else 'FAIL'} (~{test_pos})")
    print(f"  Overall:                    {'GO' if (go_ret and go_rate and test_pos >= 30) else 'NO-GO'}")
    print("\nSmoke complete. Await confirmation before --full.")


if __name__ == "__main__":
    main()
