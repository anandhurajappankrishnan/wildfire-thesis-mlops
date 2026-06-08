"""
Gold layer: train LR / RF / XGB models, evaluate, and export artifacts.

Uses grouped spatial-temporal split (disjoint locations), validation-tuned
thresholds, dummy baseline, and per-region metrics.

Input:  data/silver/silver_features_clean.parquet
Output: data/gold/*.csv, *.joblib
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from env_setup import load_project_env  # noqa: E402
from ml_eval import (  # noqa: E402
    dummy_baseline_metrics,
    metrics_at_thresholds,
    set_global_seed,
    spatial_temporal_split,
    train_val_slice,
)

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
FEATURE_VEG = ["ndvi", "evi", "ndvi_lag7", "ndvi_delta7", "day_of_year", "season_idx"]
FEATURE_WEATHER = [
    "temperature_2m",
    "total_precipitation",
    "dewpoint_temperature_2m",
    "wind_speed",
    "relative_humidity",
    "vapor_pressure_deficit",
    "temp_7d_mean",
    "precip_7d_sum",
    "low_precip_days_7d",
    "day_of_year",
    "season_idx",
]
FEATURE_TOPO = ["elevation", "slope", "day_of_year", "season_idx"]


def build_models(cfg: dict, y_train: pd.Series, imbalance: str) -> dict[str, object]:
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    spw = neg / max(pos, 1.0)
    seed = int(cfg["project"]["seed"])

    if imbalance == "balanced":
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=cfg["model"]["lr_max_iter"], class_weight="balanced", random_state=seed),
        )
        rf = RandomForestClassifier(
            n_estimators=cfg["model"]["rf_n_estimators"],
            random_state=seed,
            class_weight="balanced",
        )
        xgb = XGBClassifier(
            n_estimators=cfg["model"]["xgb_n_estimators"],
            learning_rate=cfg["model"]["xgb_learning_rate"],
            max_depth=cfg["model"]["xgb_max_depth"],
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            scale_pos_weight=spw,
        )
    elif imbalance == "none":
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=cfg["model"]["lr_max_iter"], class_weight=None, random_state=seed),
        )
        rf = RandomForestClassifier(
            n_estimators=cfg["model"]["rf_n_estimators"],
            random_state=seed,
            class_weight=None,
        )
        xgb = XGBClassifier(
            n_estimators=cfg["model"]["xgb_n_estimators"],
            learning_rate=cfg["model"]["xgb_learning_rate"],
            max_depth=cfg["model"]["xgb_max_depth"],
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=seed,
            scale_pos_weight=1.0,
        )
    else:
        raise ValueError(imbalance)

    return {"LR": lr, "RF": rf, "XGB": xgb}


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: dict,
    imbalance: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    val_frac = float(cfg["model"].get("val_frac", 0.15))
    X_fit, X_val, y_fit, y_val = train_val_slice(X_train, y_train, val_frac)
    models = build_models(cfg, y_fit, imbalance)

    rows = []
    pred_records = pd.DataFrame(index=X_test.index)
    pred_records["y_true"] = y_test.values

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_fit, y_fit)
        elapsed_min = (time.time() - t0) / 60.0

        val_prob = model.predict_proba(X_val)[:, 1] if len(X_val) else None
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred_default = (y_prob >= 0.5).astype(int)

        thresh_metrics = metrics_at_thresholds(
            y_test.to_numpy(),
            y_prob,
            y_val.to_numpy() if len(y_val) else None,
            val_prob,
        )

        row = {
            "run_date": pd.Timestamp.utcnow().date().isoformat(),
            "model_name": name,
            "imbalance_mode": imbalance,
            "train_minutes": elapsed_min,
            "pr_auc": thresh_metrics["pr_auc_default"],
            "roc_auc": thresh_metrics["roc_auc_default"],
            # Backward-compatible aliases (default threshold 0.5)
            "precision_score": thresh_metrics["precision_score_default"],
            "recall_score": thresh_metrics["recall_score_default"],
            "f1_score": thresh_metrics["f1_score_default"],
        }
        row.update(thresh_metrics)
        rows.append(row)

        pred_records[f"{name}_prob"] = y_prob
        pred_records[f"{name}_pred"] = y_pred_default
        pred_records[f"{name}_pred_tuned"] = (y_prob >= row["threshold_best_f1"]).astype(int)
        pred_records[f"{name}_threshold"] = row["threshold_best_f1"]

    return models, pd.DataFrame(rows), pred_records


def eval_ablation_row(
    df: pd.DataFrame,
    cols: list[str],
    label: str,
    cfg: dict,
) -> dict:
    X_train, X_test, y_train, y_test, _, _ = spatial_temporal_split(
        df, cols, cfg["model"]["test_size"], cfg["project"]["seed"]
    )
    X_fit, X_val, y_fit, y_val = train_val_slice(X_train, y_train, cfg["model"].get("val_frac", 0.15))
    xgb = build_models(cfg, y_fit, "balanced")["XGB"]
    xgb.fit(X_fit, y_fit)
    y_prob = xgb.predict_proba(X_test)[:, 1]
    val_prob = xgb.predict_proba(X_val)[:, 1] if len(X_val) else None
    m = metrics_at_thresholds(
        y_test.to_numpy(),
        y_prob,
        y_val.to_numpy() if len(y_val) else None,
        val_prob,
    )
    row = {"dataset": label, "pr_auc": m["pr_auc_default"], "roc_auc": m["roc_auc_default"]}
    row.update({k.replace("_default", ""): v for k, v in m.items() if k.endswith("_default")})
    row["f1_score"] = m["f1_score_default"]
    row["f1_score_best_f1"] = m["f1_score_best_f1"]
    row["threshold_best_f1"] = m["threshold_best_f1"]
    return row


def per_region_metrics(
    preds: pd.DataFrame,
    models_metrics: pd.DataFrame,
    region_col: str = "region_label",
) -> pd.DataFrame:
    from ml_eval import eval_metrics

    rows = []
    if region_col not in preds.columns:
        return pd.DataFrame()
    for region in sorted(preds[region_col].dropna().unique()):
        sub = preds[preds[region_col] == region]
        y_true = sub["y_true"].astype(int).to_numpy()
        if len(sub) < 5 or len(np.unique(y_true)) < 2:
            continue
        for _, mrow in models_metrics[models_metrics["imbalance_mode"] == "balanced"].iterrows():
            name = mrow["model_name"]
            prob_col = f"{name}_prob"
            if prob_col not in sub.columns:
                continue
            y_prob = sub[prob_col].to_numpy()
            y_pred_def = (y_prob >= 0.5).astype(int)
            y_pred_tuned = (y_prob >= mrow["threshold_best_f1"]).astype(int)
            base = {"region_label": region, "model_name": name, "n_test": len(sub), "base_rate": float(y_true.mean())}
            base.update({f"{k}_default": v for k, v in eval_metrics(y_true, y_pred_def, y_prob).items()})
            base.update({f"{k}_best_f1": v for k, v in eval_metrics(y_true, y_pred_tuned, y_prob).items()})
            rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    load_project_env()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    seed = int(cfg["project"]["seed"])
    set_global_seed(seed)

    silver_path = ROOT / cfg["paths"]["silver_dir"] / "silver_features_clean.parquet"
    gold_dir = ROOT / cfg["paths"]["gold_dir"]
    gold_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(silver_path)
    if "obs_date" in df.columns:
        df["obs_date"] = pd.to_datetime(df["obs_date"])
    df = df.dropna(subset=FEATURE_FULL + ["fire_within_7d"]).copy()

    base_rate = float(df["fire_within_7d"].mean())
    print(f"Silver positive rate (full dataset): {100 * base_rate:.2f}%")

    X_train, X_test, y_train, y_test, test_meta, _ = spatial_temporal_split(
        df, FEATURE_FULL, cfg["model"]["test_size"], seed
    )
    print(f"Train: {len(X_train)} rows ({y_train.sum()} positives) | Test: {len(X_test)} rows ({y_test.sum()} positives)")

    models_bal, metrics_bal, preds_bal = train_models(X_train, X_test, y_train, y_test, cfg, "balanced")
    for col in test_meta.columns:
        preds_bal[col] = test_meta[col].values

    metrics_bal.to_csv(gold_dir / "gold_model_results.csv", index=False)
    preds_bal.to_csv(gold_dir / "gold_test_predictions.csv", index=False)
    joblib.dump(models_bal["XGB"], gold_dir / "xgb_model.joblib")
    joblib.dump(models_bal["LR"], gold_dir / "lr_model.joblib")
    joblib.dump(models_bal["RF"], gold_dir / "rf_model.joblib")

    fi = pd.DataFrame({"feature": FEATURE_FULL, "importance": models_bal["XGB"].feature_importances_})
    fi.sort_values("importance", ascending=False).to_csv(gold_dir / "xgb_feature_importance.csv", index=False)

    # Dummy baseline
    baseline = dummy_baseline_metrics(y_train, y_test)
    pd.DataFrame([{"model_name": "DummyClassifier(stratified)", "base_rate": base_rate, **baseline}]).to_csv(
        gold_dir / "gold_baseline.csv", index=False
    )

    # Per-region metrics
    per_region = per_region_metrics(preds_bal, metrics_bal)
    if not per_region.empty:
        per_region.to_csv(gold_dir / "gold_region_metrics.csv", index=False)

    # TimeSeriesSplit CV is disabled for persistent areal cells: every cell recurs
    # across all time windows, so purging test-fold locations from train removes
    # ~100% of training rows and run_time_series_cv yields no valid folds.
    # Holdout spatial_temporal_split (above) is the sole evaluation protocol.

    # RQ2 ablation (same grouped split)
    ablation_rows = [
        eval_ablation_row(df, cols, label, cfg)
        for label, cols in [
            ("NDVI Only", FEATURE_VEG),
            ("Weather Only", FEATURE_WEATHER),
            ("Topography Only", FEATURE_TOPO),
            ("Combined", FEATURE_FULL),
        ]
    ]
    pd.DataFrame(ablation_rows).to_csv(gold_dir / "gold_ablation_rq2.csv", index=False)

    # RQ4 imbalance comparison
    imb_rows = []
    imb_pred_rows = []
    for mode in ["none", "balanced"]:
        _, mdf, pred_df = train_models(X_train, X_test, y_train, y_test, cfg, mode)
        mdf = mdf[mdf["model_name"].isin(["LR", "XGB"])]
        imb_rows.append(mdf)
        xgb_cmp = pred_df[["y_true", "XGB_prob", "XGB_pred", "XGB_pred_tuned", "XGB_threshold"]].copy()
        xgb_cmp["imbalance_mode"] = mode
        imb_pred_rows.append(xgb_cmp)
    pd.concat(imb_rows, ignore_index=True).to_csv(gold_dir / "gold_imbalance_rq4.csv", index=False)
    pd.concat(imb_pred_rows, ignore_index=True).to_csv(gold_dir / "gold_imbalance_predictions.csv", index=False)

    print(f"Saved: {gold_dir / 'gold_model_results.csv'}")


if __name__ == "__main__":
    main()
