# Wildfire Thesis MLOps & Dashboard

![Status](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B) ![GEE](https://img.shields.io/badge/Data-Google%20Earth%20Engine-green)

**Author:** [Anandhu R Krishnan](https://github.com/anandhurajappankrishnan)

End-to-end MLOps pipeline and interactive dashboard for predicting **7-day wildfire risk** across **Portugal**, **California (USA)**, and **Southeast Australia**. Master's thesis implementation in Data Science & Machine Learning.

**Repository:** [github.com/anandhurajappankrishnan/wildfire-thesis-mlops](https://github.com/anandhurajappankrishnan/wildfire-thesis-mlops)

---

## Overview

| Stage | Description |
|-------|-------------|
| **Extract** | MODIS vegetation, ERA5-Land weather, FIRMS fire labels via Google Earth Engine |
| **Transform** | Medallion architecture (Bronze → Silver → Gold) with leakage-safe feature engineering |
| **Model** | XGBoost, Random Forest, Logistic Regression with class-weight imbalance handling |
| **Evaluate** | Spatial-temporal holdout, TimeSeriesSplit CV, dummy baseline, per-region metrics |
| **Deliver** | Thesis figures/tables (RQ1–RQ7) + Streamlit dashboard with Gemini assistant |

### Data sources

| Source | Use |
|--------|-----|
| **MODIS MOD13A1** | NDVI, EVI (trailing 32-day composite, no future pixels) |
| **ERA5-Land** | Temperature, precipitation, dewpoint (same calendar day as `obs_date`) |
| **FIRMS** | Active fire detections (7-day forward labels) |
| **MCD64A1** | Monthly burned area (supplementary) |

### Sampling strategy

GEE extraction uses **fire-biased + background** sampling per region and date (not uniform random points):

- **`positive_points_per_region`** — `stratifiedSample` on the FIRMS forward-fire mask
- **`background_points_per_region`** — uniform random negatives across the AOI

This raises the positive label rate honestly (target ~5–15%) without fabricating labels. See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for full details on leakage mitigations and evaluation protocol.

### Data flow

```
GEE Extract → Bronze (Parquet) → Silver (Features) → Gold (Models + metrics) → Dashboard / Thesis outputs
```

---

## Quick start

### 1. Prerequisites

- **Python 3.9 or 3.10** (recommended; tested on 3.9)
- [Google Earth Engine](https://earthengine.google.com/) account
- (Optional) Gemini API key for the dashboard chatbot

### 2. Clone & install

```bash
git clone https://github.com/anandhurajappankrishnan/wildfire-thesis-mlops.git
cd wildfire-thesis-mlops
pip install -r requirements.txt
```

### 3. Configure secrets

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

| Variable | Required | Purpose |
|----------|----------|---------|
| `EE_PROJECT_ID` | Yes (for `--full`) | Google Cloud project linked to Earth Engine |
| `GEMINI_API_KEY` | No | Dashboard LLM assistant |
| `SQLSERVER_CONN_STR` | No | Optional bronze SQL load (step 2b) |

Authenticate Earth Engine once:

```bash
earthengine authenticate
```

### 4. Run the pipeline

**First run** — downloads satellite data with fire-biased sampling (~10–20 min, one-time per dataset):

```bash
python src/pipeline_real.py --full
```

**Later runs** — reuses cached `data/raw/` (~1 min):

```bash
python src/pipeline_real.py
# or explicitly:
python src/pipeline_real.py --quick
```

Re-run `--full` when you change dates, regions in `config/study_areas.geojson`, sampling settings, or want fresh GEE data.

### 5. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501).

> **Windows tip:** If you see `No module named streamlit`, ensure you use the same Python that installed the requirements (e.g. Python 3.9, not 3.14).

### 6. Run tests

```bash
python -m pytest tests/ -q
```

---

## Configuration

Main settings in `config.yaml`:

| Setting | Description |
|---------|-------------|
| `data.start_date` / `end_date` | Extraction window (default: Jun 2023 – Oct 2024) |
| `config/study_areas.geojson` | Country bounding boxes |
| `data.gee.positive_points_per_region` | Fire-biased sample points per region/date (default: 100) |
| `data.gee.background_points_per_region` | Random background points per region/date (default: 300) |
| `data.gee.grid_points_per_region` | Legacy fallback if pos/bg not set |
| `model.test_size` | Fraction of locations held out for spatial test set (default: 0.25) |
| `model.val_frac` | Tail of train used for threshold tuning (default: 0.15) |
| `model.cv_folds` | TimeSeriesSplit folds for XGB CV summary (default: 5) |
| `project.seed` | Random seed for numpy, sklearn, xgboost |
| `pipeline.skip_gee_if_raw_exists` | Auto quick-mode when raw CSV exists |

---

## Evaluation & gold outputs

The pipeline produces two complementary evaluation views:

| File | Split | Purpose |
|------|-------|---------|
| `gold_model_results.csv` | **Spatial-temporal holdout** | Primary pooled test metrics (LR/RF/XGB) |
| `gold_test_predictions.csv` | Same holdout | Per-row predictions at threshold 0.5 and tuned best-F1 |
| `gold_region_metrics.csv` | Same holdout | Per-region breakdown (PR-AUC, F1, etc.) |
| `gold_baseline.csv` | Same holdout | DummyClassifier (stratified) baseline |
| `gold_cv_results.csv` | **TimeSeriesSplit CV** (XGB only) | Mean ± std across chronological folds |
| `gold_ablation_rq2.csv` | Holdout | NDVI / Weather / Combined feature ablation |
| `gold_imbalance_rq4.csv` | Holdout | Class-weight vs no-weight comparison |
| `pipeline_timing_latest.csv` | — | Measured step durations (RQ1) |
| `pipeline_step_runs.csv` | — | Append-only success/failure log (RQ6) |

**Primary metrics:** PR-AUC and ROC-AUC (threshold-independent). F1/precision/recall are reported at both **0.5** and **validation-tuned best-F1** thresholds.

**Holdout split:** ~25% of locations held out entirely; train uses non-held-out locations before split date; test uses held-out locations on/after split date. No location overlap between train and test.

**CV split:** Chronological `TimeSeriesSplit` with location purge — supplementary stability estimate for XGB only; not directly comparable to holdout row counts.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for limitations and interpretation guidance (e.g. compare PR-AUC to base rate, not to 1.0).

---

## Project structure

```text
├── .env.example              # Secret template (copy to .env — never commit .env)
├── config.yaml               # Pipeline parameters
├── config/study_areas.geojson
├── docs/METHODOLOGY.md       # Sampling, leakage, evaluation protocol
├── dashboard/app.py          # Streamlit UI
├── src/
│   ├── pipeline_real.py      # Main entry (--full / --quick) + step timing log
│   ├── ml_eval.py            # Splits, thresholds, CV, baseline utilities
│   ├── env_setup.py
│   └── steps/
│       ├── step1_extract_gee.py      # GEE fire-biased + background sampling
│       ├── step2_bronze.py           # Raw → Bronze parquet
│       ├── step2b_load_sql.py        # Optional SQL Server load
│       ├── step3_silver.py           # Trailing-only feature engineering
│       ├── step4_train_eval.py       # Train LR/RF/XGB, export gold metrics
│       └── step5_generate_thesis_outputs.py
├── tests/
│   ├── test_leakage.py       # Silver features use no forward-looking data
│   └── test_splits.py        # Train/test location disjointness
├── airflow/dags/               # Optional daily orchestration
├── sql/medallion_schema.sql
├── data/                       # Generated locally (git-ignored)
└── outputs/                    # Thesis artifacts (git-ignored)
```

---

## Pipeline steps

| Step | Script | Output |
|------|--------|--------|
| 1 | `step1_extract_gee.py` | `data/raw/gee_modis_era5_burnedarea.csv` |
| 2 | `step2_bronze.py` | `data/bronze/bronze_satellite_raw.parquet` |
| 2b | `step2b_load_sql.py` | SQL Server (optional, skips if unset) |
| 3 | `step3_silver.py` | `data/silver/silver_features_clean.parquet` |
| 4 | `step4_train_eval.py` | `data/gold/*.csv`, `*.joblib` |
| 5 | `step5_generate_thesis_outputs.py` | `outputs/figures`, `outputs/tables`, `outputs/reports` |

---

## What is NOT in GitHub

These are git-ignored and must be generated locally after clone:

- `.env` — your API keys
- `data/` — datasets and trained models
- `outputs/` — thesis figures and tables

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named streamlit` | Use Python 3.9: `py -3.9 -m pip install -r requirements.txt` |
| `No prediction data found` | Run `python src/pipeline_real.py --full` (or `--quick` if raw data exists) |
| Very low positive rate / PR-AUC | Re-run `--full` after updating sampling config; old raw CSV used uniform sampling |
| GEE auth error | Run `earthengine authenticate` and set `EE_PROJECT_ID` in `.env` |
| SQL step fails | Expected if SQL Server is not running; step 2b is optional |
| Dashboard map empty | Select a region tab and a date that exists in the data |
| Plotly chart title "undefined" | Fixed in current dashboard; restart Streamlit after pulling latest |

---

## Git workflow (updates)

```bash
git add .
git status          # confirm .env and data/ are NOT listed
git commit -m "Your message"
git push
```

**Never commit:** `.env`, `data/`, `outputs/`, or any file containing real API keys.

---

## Data citations

When publishing thesis results, cite:

- NASA FIRMS active fire data ([earthdata.nasa.gov/firms](https://earthdata.nasa.gov/firms))
- Google Earth Engine datasets (MODIS, ERA5-Land)

---

## License

Academic / thesis use. © 2024–2026 **Anandhu R Krishnan**.
