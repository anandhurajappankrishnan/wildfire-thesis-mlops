# Wildfire Thesis MLOps & Dashboard

![Status](https://img.shields.io/badge/Status-Active-success) ![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B) ![GEE](https://img.shields.io/badge/Data-Google%20Earth%20Engine-green)

End-to-end MLOps pipeline and interactive dashboard for predicting **7-day wildfire risk** across **Portugal**, **California (USA)**, and **Southeast Australia**. Built for a Master's thesis in Data Science & Machine Learning.

## Overview

| Stage | Description |
|-------|-------------|
| **Extract** | MODIS vegetation, ERA5-Land weather, FIRMS fire labels via Google Earth Engine |
| **Transform** | Medallion architecture (Bronze → Silver → Gold) with feature engineering |
| **Model** | XGBoost, Random Forest, Logistic Regression with class-weight imbalance handling |
| **Deliver** | Thesis figures/tables (RQ1–RQ7) + Streamlit dashboard with Gemini assistant |

### Data sources

- **MODIS MOD13A1** — NDVI, EVI (vegetation / fuel)
- **ERA5-Land** — temperature, precipitation, dewpoint
- **FIRMS** — active fire detections (7-day forward labels)
- **MCD64A1** — monthly burned area (supplementary)

### Data flow

```
GEE Extract → Bronze (Parquet) → Silver (Features) → Gold (Models) → Dashboard / Thesis outputs
```

## Quick start

### 1. Prerequisites

- Python 3.9+
- [Google Earth Engine](https://earthengine.google.com/) account
- (Optional) Gemini API key for the dashboard chatbot

### 2. Install

```bash
git clone https://github.com/YOUR_USERNAME/wildfire-thesis-mlops.git
cd wildfire-thesis-mlops
pip install -r requirements.txt
```

### 3. Configure secrets

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS / Linux
```

Edit `.env`:

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

**First run** — downloads satellite data (~10–20 min, one-time per dataset):

```bash
python src/pipeline_real.py --full
```

**Later runs** — reuses cached `data/raw/` (~1 min):

```bash
python src/pipeline_real.py
```

Re-run `--full` only when you change dates, regions, or want fresh GEE data.

### 5. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501).

## Configuration

Main settings live in `config.yaml`:

- **Dates** — `data.start_date` / `data.end_date` (default: Jun 2023 – Oct 2024)
- **Regions** — `config/study_areas.geojson` (add/edit country bounding boxes)
- **Sampling** — `data.gee.grid_points_per_region`, `temporal_stride_days`
- **Models** — hyperparameters under `model:`

## Project structure

```text
├── .env.example              # Secret template (copy to .env — never commit .env)
├── config.yaml               # Pipeline parameters
├── config/
│   └── study_areas.geojson   # Multi-country regions (Portugal, California, Australia)
├── dashboard/
│   └── app.py                  # Streamlit UI
├── src/
│   ├── pipeline_real.py        # Main entry point (--full / --quick)
│   ├── env_setup.py            # Loads .env from project root
│   └── steps/
│       ├── step1_extract_gee.py
│       ├── step2_bronze.py
│       ├── step2b_load_sql.py  # Optional SQL Server load
│       ├── step3_silver.py
│       ├── step4_train_eval.py
│       └── step5_generate_thesis_outputs.py
├── airflow/dags/               # Optional daily orchestration
├── sql/                        # SQL Server medallion schema
├── data/                       # Generated locally (git-ignored)
└── outputs/                    # Thesis figures & tables (git-ignored)
```

## What gets git-ignored

These folders are **not** pushed to GitHub (see `.gitignore`):

- `.env` — secrets
- `data/` — raw/bronze/silver/gold datasets and models
- `outputs/` — generated thesis artifacts

After cloning, run `python src/pipeline_real.py --full` once to regenerate them.

## Pushing to GitHub

From the project root (first time):

```bash
git init
git add .
git status                    # verify .env and data/ are NOT listed
git commit -m "Initial commit: wildfire thesis MLOps pipeline and dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/wildfire-thesis-mlops.git
git push -u origin main
```

**Before pushing, confirm:**

- [ ] `.env` is not staged (only `.env.example`)
- [ ] `data/` and `outputs/` are not staged
- [ ] No API keys in code or config files

## Pipeline steps reference

| Step | Script | Output |
|------|--------|--------|
| 1 | `step1_extract_gee.py` | `data/raw/gee_modis_era5_burnedarea.csv` |
| 2 | `step2_bronze.py` | `data/bronze/bronze_satellite_raw.parquet` |
| 2b | `step2b_load_sql.py` | SQL Server (optional) |
| 3 | `step3_silver.py` | `data/silver/silver_features_clean.parquet` |
| 4 | `step4_train_eval.py` | `data/gold/*.joblib`, `gold_test_predictions.csv` |
| 5 | `step5_generate_thesis_outputs.py` | `outputs/figures`, `outputs/tables`, `outputs/reports` |

## License & thesis use

Academic / thesis project. Cite FIRMS and GEE data sources when publishing results.

---
*Built for a Master's Thesis in Data Science & Machine Learning.*
