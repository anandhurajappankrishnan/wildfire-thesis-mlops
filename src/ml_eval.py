"""
Shared ML evaluation utilities: splits, thresholds, CV, baselines, seeds.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit


def stable_int_hash(text: str, modulo: int = 2_000_000_000) -> int:
    """Deterministic hash (unlike built-in hash() which varies with PYTHONHASHSEED)."""
    return int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % modulo


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)


def location_keys(df: pd.DataFrame) -> pd.Series:
    return df["latitude"].astype(str) + "_" + df["longitude"].astype(str)


def add_location_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["loc_key"] = location_keys(out)
    out["loc_id"] = out.groupby("loc_key", sort=False).ngroup()
    return out


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    out = {
        "precision_score": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_score": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if len(np.unique(y_true)) < 2:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    else:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    return out


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Return (threshold, f1) maximizing F1 on the provided slice."""
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 >= best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


def spatial_temporal_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    test_size: float,
    seed: int,
    min_rows_train: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, set[str]]:
    """
    Chronological split with disjoint test locations.

    Train: non-held-out locations, obs_date strictly before split date.
    Test:  held-out locations only, obs_date on/after split date.
    """
    df = add_location_id(df.sort_values("obs_date").reset_index(drop=True))
    n = len(df)
    split_idx = max(min_rows_train, int(n * (1 - test_size)))
    split_idx = min(split_idx, n - 1)
    split_date = df.iloc[split_idx]["obs_date"]

    unique_locs = df["loc_key"].unique()
    rng = np.random.RandomState(seed)
    n_test_locs = max(1, int(len(unique_locs) * test_size))
    test_loc_keys = set(rng.choice(unique_locs, size=n_test_locs, replace=False))

    train_mask = (df["obs_date"] < split_date) & (~df["loc_key"].isin(test_loc_keys))
    test_mask = (df["obs_date"] >= split_date) & (df["loc_key"].isin(test_loc_keys))

    # Ensure both classes in train/test when possible
    y = df["fire_within_7d"].astype(int)
    step = max(1, n // 200)
    while split_idx >= min_rows_train:
        tr, te = y[train_mask], y[test_mask]
        if len(tr) >= min_rows_train and len(te) > 0 and tr.nunique() > 1 and te.nunique() > 1:
            break
        split_idx -= step
        split_date = df.iloc[split_idx]["obs_date"]
        train_mask = (df["obs_date"] < split_date) & (~df["loc_key"].isin(test_loc_keys))
        test_mask = (df["obs_date"] >= split_date) & (df["loc_key"].isin(test_loc_keys))

    train_df = df.loc[train_mask]
    test_df = df.loc[test_mask]
    meta_cols = [c for c in ["obs_date", "latitude", "longitude", "country", "region_label", "loc_key"] if c in test_df.columns]

    return (
        train_df[feature_cols],
        test_df[feature_cols],
        train_df["fire_within_7d"].astype(int),
        test_df["fire_within_7d"].astype(int),
        test_df[meta_cols],
        test_loc_keys,
    )


def metrics_at_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    val_y: np.ndarray | None = None,
    val_prob: np.ndarray | None = None,
) -> dict[str, Any]:
    """Metrics at default 0.5 and at validation-tuned best-F1 threshold."""
    pred_default = (y_prob >= 0.5).astype(int)
    m_default = eval_metrics(y_true, pred_default, y_prob)

    if val_y is not None and val_prob is not None and len(np.unique(val_y)) > 1:
        best_t, _ = best_f1_threshold(val_y, val_prob)
    else:
        best_t = 0.5

    pred_tuned = (y_prob >= best_t).astype(int)
    m_tuned = eval_metrics(y_true, pred_tuned, y_prob)

    return {
        "threshold_default": 0.5,
        **{f"{k}_default": v for k, v in m_default.items()},
        "threshold_best_f1": best_t,
        **{f"{k}_best_f1": v for k, v in m_tuned.items()},
    }


def train_val_slice(
    X_train: pd.DataFrame, y_train: pd.Series, val_frac: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    n = len(X_train)
    if n < 50:
        return X_train, X_train.iloc[:0], y_train, y_train.iloc[:0]
    split = int(n * (1 - val_frac))
    return X_train.iloc[:split], X_train.iloc[split:], y_train.iloc[:split], y_train.iloc[split:]


def dummy_baseline_metrics(y_train: pd.Series, y_test: pd.Series) -> dict[str, float]:
    """Majority-class (stratified) dummy PR-AUC baseline."""
    dummy = DummyClassifier(strategy="stratified", random_state=0)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    y_prob = dummy.predict_proba(np.zeros((len(y_test), 1)))[:, 1]
    y_pred = dummy.predict(np.zeros((len(y_test), 1)))
    return eval_metrics(y_test.to_numpy(), y_pred, y_prob)


def run_time_series_cv(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_factory,
    n_splits: int,
    seed: int,
) -> pd.DataFrame:
    """TimeSeriesSplit CV with locations held out from train when they appear in test."""
    df = add_location_id(df.sort_values("obs_date").reset_index(drop=True))
    tscv = TimeSeriesSplit(n_splits=min(n_splits, max(2, len(df) // 500)))

    rows = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]
        test_locs = set(test_df["loc_key"])
        train_df = train_df[~train_df["loc_key"].isin(test_locs)]

        X_tr, y_tr = train_df[feature_cols], train_df["fire_within_7d"].astype(int)
        X_te, y_te = test_df[feature_cols], test_df["fire_within_7d"].astype(int)
        if len(X_te) < 10 or len(X_tr) < 50 or y_tr.nunique() < 2 or y_te.nunique() < 2:
            continue

        model = model_factory(y_tr)
        model.fit(X_tr, y_tr)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        row = {"fold": fold, "n_train": len(X_tr), "n_test": len(X_te)}
        row.update(eval_metrics(y_te.to_numpy(), y_pred, y_prob))
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    cv = pd.DataFrame(rows)
    summary = cv.drop(columns=["fold"]).agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean", "std"]
    return summary
