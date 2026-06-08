Wildfire Thesis MLOps — Review Package
Author: Anandhu Rajappan Krishnan
Regions: Portugal, California (USA), Southeast Australia
Study window: 2023-06-01 to 2024-10-31 | 14-day fire horizon | 0.1 deg areal cells

Contents are read-only copies of finalized pipeline outputs.
Live reproduction: run notebooks/thesis_experiment.ipynb from repo root.

RQ1 — Multi-region ingestion and pipeline operability
  figures/rq1/figure_1_1_system_architecture.pdf — End-to-end medallion pipeline architecture (GEE → Bronze → Silver → Gold → Dashboard).
  figures/rq1/figure_1_2_daily_stability.pdf — Daily ingestion stability metric from an earlier pipeline run.
  figures/rq1/figure_1_2_dataflow_timeline.pdf — Measured wall-clock latency per pipeline stage (latest run).
  figures/rq1/figure_1_2_pipeline_latency.pdf — Pipeline stage latency chart (alternate run artifact).
  tables/pipeline_timing_latest.csv — Latest measured duration (minutes) for each pipeline step.

RQ2 — Feature-set fusion ablation (NDVI / Weather / Topography / Combined)
  figures/rq2/figure_2_1_performance_bar.pdf — XGB holdout PR-AUC and F1 for NDVI / Weather / Topography / Combined feature sets.
  figures/rq2/figure_2_2_feature_correlation_heatmap.pdf — Pearson correlation heatmap for a subset of combined features.
  tables/gold_ablation_rq2.csv — RQ2 ablation metrics: single-source vs combined feature sets (balanced XGB holdout).

RQ3 — Environmental drivers and feature importance
  figures/rq3/figure_3_1_feature_importance.pdf — XGB gain-based feature importance for the combined 17-feature model.
  figures/rq3/figure_3_2_temporal_trends.pdf — NDVI time series with fire-within-14d labels overlaid.
  tables/xgb_feature_importance.csv — Numeric XGB feature importance rankings for the combined model.

RQ4 — Class imbalance handling (balanced vs unbalanced)
  figures/rq4/figure_4_1_imbalance_comparison.pdf — Class-weight balancing comparison (legacy figure from earlier run).
  figures/rq4/figure_4_1_pr_curves.pdf — Precision–recall curves for balanced LR and XGB on spatial holdout.
  figures/rq4/figure_4_2_confusion_matrix_comparison.pdf — Confusion matrices for LR and XGB at threshold 0.5 on holdout.
  tables/gold_imbalance_predictions.csv — Holdout predictions used for RQ4 imbalance analysis.
  tables/gold_imbalance_rq4.csv — Balanced vs unbalanced LR/XGB metrics on the same holdout split.

RQ5 — Classifier comparison and regional holdout performance
  figures/rq5/figure_5_1_model_comparison.pdf — Holdout PR-AUC, ROC-AUC, and tuned F1 for LR, RF, and XGB vs dummy baseline.
  figures/rq5/figure_5_1_roc_curves.pdf — ROC curves for all three classifiers on spatial holdout.
  figures/rq5/figure_5_2_model_stability_over_time.pdf — Model score stability over time (legacy figure from earlier run).
  figures/rq5/figure_5_3_region_pr_auc.pdf — Random Forest PR-AUC broken down by study region on holdout.
  tables/gold_baseline.csv — Stratified DummyClassifier baseline PR-AUC and ROC-AUC on holdout.
  tables/gold_model_results.csv — Primary holdout metrics for LR, RF, and XGB (balanced, spatial-temporal split).
  tables/gold_region_metrics.csv — Per-region holdout PR-AUC and ROC-AUC for each classifier.
  tables/gold_test_predictions.csv — Cell-level holdout predictions and probabilities for all models.

RQ6 — MLOps reproducibility and orchestration logs
  figures/rq6/figure_6_1_airflow_dag_visualization.pdf — Conceptual Airflow DAG mirroring the Python pipeline steps.
  figures/rq6/figure_6_2_pipeline_success_rate.pdf — Weekly pipeline step success rate from orchestration logs.
  tables/pipeline_step_runs.csv — Timestamped log of each pipeline step run, status, and duration.

RQ7 — Operational decision scenarios and risk visualization
  figures/rq7/figure_7_1_dashboard_snapshot.pdf — Conceptual Streamlit dashboard readout from gold artifacts.
  figures/rq7/figure_7_2_risk_map_visualization.pdf — RF predicted risk scores over time by region on holdout cells.

Notebook
  notebooks/thesis_experiment.ipynb — Self-contained live experiment notebook: feature engineering, training, RQ1–RQ7, reproducibility checks.
