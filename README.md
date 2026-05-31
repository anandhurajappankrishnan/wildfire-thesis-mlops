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
| **Transform** | Medallion architecture (Bronze → Silver → Gold) with feature engineering |
| **Model** | XGBoost, Random Forest, Logistic Regression with class-weight imbalance handling |
| **Deliver** | Thesis figures/tables (RQ1–RQ7) + Streamlit dashboard with Gemini assistant |

### Data sources

| Source | Use |
|--------|-----|
| **MODIS MOD13A1** | NDVI, EVI (vegetation / fuel) |
| **ERA5-Land** | Temperature, precipitation, dewpoint |
| **FIRMS** | Active fire detections (7-day forward labels) |
| **MCD64A1** | Monthly burned area (supplementary) |

### Data flow

```
GEE Extract → Bronze (Parquet) → Silver (Features) → Gold (Models) → Dashboard / Thesis outputs
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

**First run** — downloads satellite data (~10–20 min, one-time per dataset):

```bash
python src/pipeline_real.py --full
```

**Later runs** — reuses cached `data/raw/` (~1 min):

```bash
python src/pipeline_real.py
```

Re-run `--full` only when you change dates, regions in `config/study_areas.geojson`, or want fresh GEE data.

### 5. Launch the dashboard

```bash
python -m streamlit run dashboard/app.py
```

Open [http://localhost:8501](http://localhost:8501).

> **Windows tip:** If you see `No module named streamlit`, ensure you use the same Python that installed the requirements (e.g. Python 3.9, not 3.14).

---

## Configuration

Main settings in `config.yaml`:

| Setting | Description |
|---------|-------------|
| `data.start_date` / `end_date` | Extraction window (default: Jun 2023 – Oct 2024) |
| `config/study_areas.geojson` | Country bounding boxes |
| `data.gee.grid_points_per_region` | Sample points per country per date |
| `model.*` | LR / RF / XGB hyperparameters |
| `pipeline.skip_gee_if_raw_exists` | Auto quick-mode when raw CSV exists |

---

## Project structure

```text
├── .env.example              # Secret template (copy to .env — never commit .env)
├── config.yaml               # Pipeline parameters
├── config/study_areas.geojson
├── dashboard/app.py          # Streamlit UI
├── src/
│   ├── pipeline_real.py      # Main entry (--full / --quick)
│   ├── env_setup.py
│   └── steps/
│       ├── step1_extract_gee.py      # GEE multi-country extraction
│       ├── step2_bronze.py           # Raw → Bronze parquet
│       ├── step2b_load_sql.py        # Optional SQL Server load
│       ├── step3_silver.py           # Feature engineering + labels
│       ├── step4_train_eval.py       # Train LR/RF/XGB, export gold
│       └── step5_generate_thesis_outputs.py
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
| 4 | `step4_train_eval.py` | `data/gold/*.joblib`, `gold_test_predictions.csv` |
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
| GEE auth error | Run `earthengine authenticate` and set `EE_PROJECT_ID` in `.env` |
| SQL step fails | Expected if SQL Server is not running; step 2b is optional |
| Dashboard map empty | Select a region tab and a date that exists in the data |

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
