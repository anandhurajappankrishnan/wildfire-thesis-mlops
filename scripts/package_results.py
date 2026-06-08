"""Assemble review_package/ and review_package.zip from figures, gold CSVs, and notebook."""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES_SRC = ROOT / "outputs" / "figures"
GOLD_DIR = ROOT / "data" / "gold"
NOTEBOOK_SRC = ROOT / "notebooks" / "thesis_experiment.ipynb"
PACKAGE_DIR = ROOT / "review_package"
ZIP_PATH = ROOT / "review_package.zip"

GOLD_CSVS = [
    "gold_model_results.csv",
    "gold_region_metrics.csv",
    "gold_baseline.csv",
    "gold_ablation_rq2.csv",
    "gold_imbalance_rq4.csv",
    "gold_imbalance_predictions.csv",
    "xgb_feature_importance.csv",
    "gold_test_predictions.csv",
    "pipeline_timing_latest.csv",
    "pipeline_step_runs.csv",
]

# One-line descriptions keyed by package-relative path
DESCRIPTIONS: dict[str, str] = {
    # RQ1
    "figures/rq1/figure_1_1_system_architecture.pdf": (
        "End-to-end medallion pipeline architecture (GEE → Bronze → Silver → Gold → Dashboard)."
    ),
    "figures/rq1/figure_1_2_daily_stability.pdf": (
        "Daily ingestion stability metric from an earlier pipeline run."
    ),
    "figures/rq1/figure_1_2_dataflow_timeline.pdf": (
        "Measured wall-clock latency per pipeline stage (latest run)."
    ),
    "figures/rq1/figure_1_2_pipeline_latency.pdf": (
        "Pipeline stage latency chart (alternate run artifact)."
    ),
    "tables/pipeline_timing_latest.csv": (
        "Latest measured duration (minutes) for each pipeline step."
    ),
    # RQ2
    "figures/rq2/figure_2_1_performance_bar.pdf": (
        "XGB holdout PR-AUC and F1 for NDVI / Weather / Topography / Combined feature sets."
    ),
    "figures/rq2/figure_2_2_feature_correlation_heatmap.pdf": (
        "Pearson correlation heatmap for a subset of combined features."
    ),
    "tables/gold_ablation_rq2.csv": (
        "RQ2 ablation metrics: single-source vs combined feature sets (balanced XGB holdout)."
    ),
    # RQ3
    "figures/rq3/figure_3_1_feature_importance.pdf": (
        "XGB gain-based feature importance for the combined 17-feature model."
    ),
    "figures/rq3/figure_3_2_temporal_trends.pdf": (
        "NDVI time series with fire-within-14d labels overlaid."
    ),
    "tables/xgb_feature_importance.csv": (
        "Numeric XGB feature importance rankings for the combined model."
    ),
    # RQ4
    "figures/rq4/figure_4_1_imbalance_comparison.pdf": (
        "Class-weight balancing comparison (legacy figure from earlier run)."
    ),
    "figures/rq4/figure_4_1_pr_curves.pdf": (
        "Precision–recall curves for balanced LR and XGB on spatial holdout."
    ),
    "figures/rq4/figure_4_2_confusion_matrix_comparison.pdf": (
        "Confusion matrices for LR and XGB at threshold 0.5 on holdout."
    ),
    "tables/gold_imbalance_rq4.csv": (
        "Balanced vs unbalanced LR/XGB metrics on the same holdout split."
    ),
    "tables/gold_imbalance_predictions.csv": (
        "Holdout predictions used for RQ4 imbalance analysis."
    ),
    # RQ5
    "figures/rq5/figure_5_1_model_comparison.pdf": (
        "Holdout PR-AUC, ROC-AUC, and tuned F1 for LR, RF, and XGB vs dummy baseline."
    ),
    "figures/rq5/figure_5_1_roc_curves.pdf": (
        "ROC curves for all three classifiers on spatial holdout."
    ),
    "figures/rq5/figure_5_2_model_stability_over_time.pdf": (
        "Model score stability over time (legacy figure from earlier run)."
    ),
    "figures/rq5/figure_5_3_region_pr_auc.pdf": (
        "Random Forest PR-AUC broken down by study region on holdout."
    ),
    "tables/gold_model_results.csv": (
        "Primary holdout metrics for LR, RF, and XGB (balanced, spatial-temporal split)."
    ),
    "tables/gold_region_metrics.csv": (
        "Per-region holdout PR-AUC and ROC-AUC for each classifier."
    ),
    "tables/gold_baseline.csv": (
        "Stratified DummyClassifier baseline PR-AUC and ROC-AUC on holdout."
    ),
    "tables/gold_test_predictions.csv": (
        "Cell-level holdout predictions and probabilities for all models."
    ),
    # RQ6
    "figures/rq6/figure_6_1_airflow_dag_visualization.pdf": (
        "Conceptual Airflow DAG mirroring the Python pipeline steps."
    ),
    "figures/rq6/figure_6_2_pipeline_success_rate.pdf": (
        "Weekly pipeline step success rate from orchestration logs."
    ),
    "tables/pipeline_step_runs.csv": (
        "Timestamped log of each pipeline step run, status, and duration."
    ),
    # RQ7
    "figures/rq7/figure_7_1_dashboard_snapshot.pdf": (
        "Conceptual Streamlit dashboard readout from gold artifacts."
    ),
    "figures/rq7/figure_7_2_risk_map_visualization.pdf": (
        "RF predicted risk scores over time by region on holdout cells."
    ),
    # Notebook
    "notebooks/thesis_experiment.ipynb": (
        "Self-contained live experiment notebook: feature engineering, training, RQ1–RQ7, reproducibility checks."
    ),
}


def snapshot_data_mtimes() -> dict[str, float]:
    data_dir = ROOT / "data"
    if not data_dir.exists():
        return {}
    return {
        str(path.relative_to(ROOT)): path.stat().st_mtime
        for path in sorted(data_dir.rglob("*"))
        if path.is_file()
    }


def collect_pdfs() -> list[Path]:
    copied: list[Path] = []
    for rq in range(1, 8):
        src_dir = FIGURES_SRC / f"rq{rq}"
        if not src_dir.is_dir():
            continue
        dest_dir = PACKAGE_DIR / "figures" / f"rq{rq}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for pdf in sorted(src_dir.glob("*.pdf")):
            dest = dest_dir / pdf.name
            shutil.copy2(pdf, dest)
            copied.append(dest)
    return copied


def copy_gold_csvs() -> list[Path]:
    dest_dir = PACKAGE_DIR / "tables"
    dest_dir.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for name in GOLD_CSVS:
        src = GOLD_DIR / name
        if not src.is_file():
            raise FileNotFoundError(f"Required gold CSV not found: {src}")
        dest = dest_dir / name
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def copy_notebook() -> Path:
    dest_dir = PACKAGE_DIR / "notebooks"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / NOTEBOOK_SRC.name
    shutil.copy2(NOTEBOOK_SRC, dest)
    return dest


def rel_package_path(path: Path) -> str:
    return path.relative_to(PACKAGE_DIR).as_posix()


def write_readme(packaged_files: list[Path]) -> Path:
    readme = PACKAGE_DIR / "README.txt"
    lines = [
        "Wildfire Thesis MLOps — Review Package",
        "Author: Anandhu Rajappan Krishnan",
        "Regions: Portugal, California (USA), Southeast Australia",
        "Study window: 2023-06-01 to 2024-10-31 | 14-day fire horizon | 0.1 deg areal cells",
        "",
        "Contents are read-only copies of finalized pipeline outputs.",
        "Live reproduction: run notebooks/thesis_experiment.ipynb from repo root.",
        "",
    ]

    rq_sections: dict[str, list[str]] = {f"RQ{n}": [] for n in range(1, 8)}
    rq_sections["Notebook"] = []

    for path in sorted(packaged_files, key=lambda p: rel_package_path(p)):
        rel = rel_package_path(path)
        desc = DESCRIPTIONS.get(rel, "Review artifact.")
        entry = f"  {rel} — {desc}"

        if rel.startswith("figures/rq") or rel.startswith("tables/"):
            rq_num = None
            if rel.startswith("figures/rq"):
                rq_num = rel.split("/")[1].upper()
            else:
                table_rq_map = {
                    "gold_ablation_rq2.csv": "RQ2",
                    "xgb_feature_importance.csv": "RQ3",
                    "gold_imbalance_rq4.csv": "RQ4",
                    "gold_imbalance_predictions.csv": "RQ4",
                    "gold_model_results.csv": "RQ5",
                    "gold_region_metrics.csv": "RQ5",
                    "gold_baseline.csv": "RQ5",
                    "gold_test_predictions.csv": "RQ5",
                    "pipeline_timing_latest.csv": "RQ1",
                    "pipeline_step_runs.csv": "RQ6",
                }
                rq_num = table_rq_map.get(path.name, "RQ5")
            rq_sections.setdefault(rq_num, []).append(entry)
        elif rel.startswith("notebooks/"):
            rq_sections["Notebook"].append(entry)

    rq_titles = {
        "RQ1": "RQ1 — Multi-region ingestion and pipeline operability",
        "RQ2": "RQ2 — Feature-set fusion ablation (NDVI / Weather / Topography / Combined)",
        "RQ3": "RQ3 — Environmental drivers and feature importance",
        "RQ4": "RQ4 — Class imbalance handling (balanced vs unbalanced)",
        "RQ5": "RQ5 — Classifier comparison and regional holdout performance",
        "RQ6": "RQ6 — MLOps reproducibility and orchestration logs",
        "RQ7": "RQ7 — Operational decision scenarios and risk visualization",
        "Notebook": "Notebook",
    }

    for key in ["RQ1", "RQ2", "RQ3", "RQ4", "RQ5", "RQ6", "RQ7", "Notebook"]:
        entries = rq_sections.get(key, [])
        if not entries:
            continue
        lines.append(rq_titles[key])
        lines.extend(entries)
        lines.append("")

    readme.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return readme


def create_zip() -> Path:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=path.relative_to(ROOT).as_posix())
    return ZIP_PATH


def main() -> None:
    if not NOTEBOOK_SRC.is_file():
        raise FileNotFoundError(f"Notebook not found: {NOTEBOOK_SRC}")

    before_mtimes = snapshot_data_mtimes()

    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True)

    pdfs = collect_pdfs()
    csvs = copy_gold_csvs()
    notebook = copy_notebook()
    packaged = pdfs + csvs + [notebook]
    readme = write_readme(packaged)
    packaged.append(readme)

    zip_path = create_zip()

    after_mtimes = snapshot_data_mtimes()
    for rel, mtime in before_mtimes.items():
        assert rel in after_mtimes, f"Missing data file after packaging: {rel}"
        assert abs(after_mtimes[rel] - mtime) < 0.01, f"Source data modified: {rel}"

    print(f"Package directory: {PACKAGE_DIR}")
    print(f"Zip archive:       {zip_path}")
    print(f"PDF figures:       {len(pdfs)}")
    print(f"Gold CSVs:         {len(csvs)}")
    print(f"Notebook:          1")
    print(f"README:            {readme.name}")
    print("Source data/ untouched.")


if __name__ == "__main__":
    main()
