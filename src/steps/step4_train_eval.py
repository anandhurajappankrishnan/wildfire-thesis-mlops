"""
Gold layer: train LR / RF / XGB models, evaluate, and export artifacts.

Uses a chronological train/test split. Saves models, predictions, ablation
and imbalance comparison CSVs under data/gold/.

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
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from env_setup import load_project_env  # noqa: E402

FEATURE_FULL = [
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
FEATURE_VEG = ["ndvi", "evi", "ndvi_lag7", "ndvi_delta7", "day_of_year", "season_idx"]
FEATURE_WEATHER = [
    "temperature_2m",
    "total_precipitation",
    "dewpoint_temperature_2m",
    "temp_7d_mean",
    "precip_7d_sum",
    "day_of_year",
    "season_idx",
]


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    out = {
        "precision_score": precision_score(y_true, y_pred, zero_division=0),
        "recall_score": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }
    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return out


def time_based_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float,
    min_rows_train: int = 500,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Chronological split; nudges split earlier if the tail has no positives (common with sparse fire labels)."""
    df = df.sort_values("obs_date").reset_index(drop=True)
    n = len(df)
    y = df["fire_within_7d"].astype(int)
    split_idx = int(n * (1 - test_size))
    step = max(1, n // 200)

    def both_classes(mask: pd.Series) -> bool:
        return int(mask.min()) == 0 and int(mask.max()) == 1

    while split_idx >= min_rows_train:
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        if both_classes(y_test) and both_classes(y_train):
            break
        split_idx -= step

    if split_idx < min_rows_train:
        split_idx = int(n * (1 - test_size))
        print(
            "Warning: could not find a time split with fire labels in both train and test. "
            "Metrics may show NaN. Increase max_dates / date range or fix burn labels."
        )

    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df["fire_within_7d"].astype(int)
    y_test = test_df["fire_within_7d"].astype(int)
    meta_cols = [c for c in ["obs_date", "latitude", "longitude", "country", "region_label"] if c in test_df.columns]
    test_meta = test_df[meta_cols]
    return X_train, X_test, y_train, y_test, test_meta


def train_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    cfg: dict,
    imbalance: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    spw = neg / max(pos, 1.0)

    if imbalance == "balanced":
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=cfg["model"]["lr_max_iter"], class_weight="balanced"),
        )
        rf = RandomForestClassifier(
            n_estimators=cfg["model"]["rf_n_estimators"],
            random_state=cfg["project"]["seed"],
            class_weight="balanced",
        )
        xgb = XGBClassifier(
            n_estimators=cfg["model"]["xgb_n_estimators"],
            learning_rate=cfg["model"]["xgb_learning_rate"],
            max_depth=cfg["model"]["xgb_max_depth"],
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=cfg["project"]["seed"],
            scale_pos_weight=spw,
        )
    elif imbalance == "none":
        lr = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=cfg["model"]["lr_max_iter"], class_weight=None),
        )
        rf = RandomForestClassifier(
            n_estimators=cfg["model"]["rf_n_estimators"],
            random_state=cfg["project"]["seed"],
            class_weight=None,
        )
        xgb = XGBClassifier(
            n_estimators=cfg["model"]["xgb_n_estimators"],
            learning_rate=cfg["model"]["xgb_learning_rate"],
            max_depth=cfg["model"]["xgb_max_depth"],
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=cfg["project"]["seed"],
            scale_pos_weight=1.0,
        )
    else:
        raise ValueError(imbalance)

    models = {"LR": lr, "RF": rf, "XGB": xgb}
    rows = []
    pred_records = pd.DataFrame(index=X_test.index)
    pred_records["y_true"] = y_test.values

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed_min = (time.time() - t0) / 60.0
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        row = {
            "run_date": pd.Timestamp.utcnow().date().isoformat(),
            "model_name": name,
            "imbalance_mode": imbalance,
            "train_minutes": elapsed_min,
        }
        row.update(eval_metrics(y_test.to_numpy(), y_pred, y_prob))
        rows.append(row)
        pred_records[f"{name}_prob"] = y_prob
        pred_records[f"{name}_pred"] = y_pred

    return models, pd.DataFrame(rows), pred_records


def main() -> None:
    load_project_env()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    silver_path = ROOT / cfg["paths"]["silver_dir"] / "silver_features_clean.parquet"
    gold_dir = ROOT / cfg["paths"]["gold_dir"]
    gold_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(silver_path)
    if "obs_date" in df.columns:
        df["obs_date"] = pd.to_datetime(df["obs_date"])

    df = df.dropna(subset=FEATURE_FULL + ["fire_within_7d"]).copy()

    # Primary run: balanced / imbalance-aware
    X_train, X_test, y_train, y_test, test_meta = time_based_split(df, FEATURE_FULL, cfg["model"]["test_size"])
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

    # RQ2 ablation (same split)
    ablation_rows = []
    for label, cols in [("NDVI Only", FEATURE_VEG), ("Weather Only", FEATURE_WEATHER), ("Combined", FEATURE_FULL)]:
        Xt_tr, Xt_te, yt_tr, yt_te, _ = time_based_split(df, cols, cfg["model"]["test_size"])
        xgb = XGBClassifier(
            n_estimators=cfg["model"]["xgb_n_estimators"],
            learning_rate=cfg["model"]["xgb_learning_rate"],
            max_depth=cfg["model"]["xgb_max_depth"],
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=cfg["project"]["seed"],
            scale_pos_weight=float((len(yt_tr) - yt_tr.sum()) / max(yt_tr.sum(), 1)),
        )
        xgb.fit(Xt_tr, yt_tr)
        y_prob = xgb.predict_proba(Xt_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        row = {"dataset": label}
        row.update(eval_metrics(yt_te.to_numpy(), y_pred, y_prob))
        ablation_rows.append(row)
    pd.DataFrame(ablation_rows).to_csv(gold_dir / "gold_ablation_rq2.csv", index=False)

    # RQ4 imbalance comparison (LR + XGB only to keep runtime low)
    imb_rows = []
    imb_pred_rows = []
    for mode in ["none", "balanced"]:
        _, mdf, pred_df = train_models(X_train, X_test, y_train, y_test, cfg, mode)
        mdf = mdf[mdf["model_name"].isin(["LR", "XGB"])]
        imb_rows.append(mdf)
        xgb_cmp = pred_df[["y_true", "XGB_prob", "XGB_pred"]].copy()
        xgb_cmp["imbalance_mode"] = mode
        imb_pred_rows.append(xgb_cmp)
    pd.concat(imb_rows, ignore_index=True).to_csv(gold_dir / "gold_imbalance_rq4.csv", index=False)
    pd.concat(imb_pred_rows, ignore_index=True).to_csv(gold_dir / "gold_imbalance_predictions.csv", index=False)

    print(f"Saved: {gold_dir / 'gold_model_results.csv'}")


if __name__ == "__main__":
    main()
