# Methodology

This document describes the wildfire risk pipeline implemented in this repository for a Master's thesis on 7-day wildfire prediction across Portugal, California (USA), and Southeast Australia.

## Problem and label

**Target:** `fire_within_7d` — binary indicator that an active fire (FIRMS thermal anomaly, ≥325 K, 3 km focal max) occurs within the **next 7 days** after `obs_date` at the sample location.

**Observation date:** Each row is a point-in-time snapshot used to predict forward fire activity. Features must use only information available at or before `obs_date`.

## Data sources (Google Earth Engine)

| Source | Collection | Role |
|--------|------------|------|
| Vegetation | MODIS/061/MOD13A1 | NDVI, EVI |
| Weather | ECMWF/ERA5_LAND/DAILY_AGGR | 2 m temperature, precipitation, dewpoint (same calendar day as `obs_date`) |
| Burn history | MODIS/061/MCD64A1 | Monthly burned-area flag (context only; label prefers FIRMS) |
| Fire label | FIRMS | Forward 7-day fire mask |

Spatial resolution for sampling: **500 m** (`gee_scale_meters` in `config.yaml`).

## Sampling strategy

Uniform random points across each region's bounding box yield ~0.2% positive labels because fires are spatially sparse.

**Current approach (honest oversampling of location, not labels):**

- Per region and date, sample **`positive_points_per_region`** points via `stratifiedSample` on the FIRMS forward-fire mask (fire-biased locations).
- Sample **`background_points_per_region`** uniform random points across the AOI (negatives).
- Labels are computed from FIRMS at each point — no label duplication or fabrication.

Configurable in `config.yaml` under `data.gee`. Goal: **5–15% positive rate** in silver while retaining real negatives.

Seeds use `hashlib.md5` per region (not Python `hash()`) for cross-run reproducibility.

## Feature engineering (Silver)

All rolling/lag features are **trailing-only** at each `(latitude, longitude, country)` group:

- `ndvi_lag7`: `shift` by ~7 days (stride-aware)
- `temp_7d_mean`, `precip_7d_sum`: pandas trailing rolling window (includes current row, no future rows)
- `ndvi_delta7`: current NDVI minus lagged NDVI

MODIS vegetation in Step 1 uses a **trailing 32-day median** ending on `obs_date` (no ±16-day centered window).

## Leakage mitigations

1. **Temporal:** Features exclude post-`obs_date` MODIS composites and forward weather.
2. **Label:** FIRMS label window starts the day after `obs_date`.
3. **Split:** Train/test uses a **grouped spatial-temporal split** — held-out locations never appear in training; training rows are strictly before the split date.

## Evaluation protocol

- **Primary metrics (threshold-independent):** PR-AUC, ROC-AUC.
- **Secondary (threshold-dependent):** Precision, recall, F1 at **0.5** and at **validation-tuned best-F1** threshold.
- **Baseline:** `DummyClassifier(strategy="stratified")` — PR-AUC near the dataset base rate.
- **Cross-validation:** `TimeSeriesSplit` on chronologically ordered rows; test locations removed from train in each fold. Reported as mean ± std in `gold_cv_results.csv`.
- **Per-region metrics:** `gold_region_metrics.csv` — fire dynamics differ across regions.

Random seeds: `project.seed` in `config.yaml` (numpy, sklearn, xgboost).

## Limitations

- **Base rate / rarity:** Even with fire-biased sampling, labels remain noisy at 500 m; PR-AUC should be compared to the **positive base rate**, not to 1.0.
- **FIRMS detection limits:** Small or smoldering fires may be missed; focal smoothing introduces spatial ambiguity.
- **Point samples ≠ areal forecasting:** Results reflect risk at sampled locations, not full regional wall-to-wall maps.
- **Three regions, one model pool:** Pooled training assumes some transferability; per-region metrics should be inspected.
- **Date range:** 2023-06-01 to 2024-10-31 — conclusions may not generalize to other fire seasons or climates without re-extraction.

## Reproducibility

```bash
python src/pipeline_real.py --full   # GEE re-extraction (requires Earth Engine auth)
python src/pipeline_real.py --quick  # Reuse cached raw CSV
python -m pytest tests/
python -m streamlit run dashboard/app.py
```

Pipeline step timing and success/failure are logged to `data/gold/pipeline_timing_latest.csv` and `data/gold/pipeline_step_runs.csv`.
