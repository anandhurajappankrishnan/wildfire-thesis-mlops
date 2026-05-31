"""
Generate thesis figures, tables, and research-question report (RQ1–RQ7).

Reads gold-layer metrics and writes outputs/figures, outputs/tables,
and outputs/reports/research_question_answers.md.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from env_setup import load_project_env  # noqa: E402


def _figure_format(cfg: dict) -> str:
    fmt = str(cfg.get("paths", {}).get("figure_format", "pdf")).lower().lstrip(".")
    return fmt if fmt in ("pdf", "svg", "png") else "pdf"


def save_thesis_figure(fig_root: Path, rq: str, basename: str, fmt: str) -> None:
    out = fig_root / rq / f"{basename}.{fmt}"
    plt.savefig(out, format=fmt, bbox_inches="tight")
    plt.close()


def ensure_dirs(root: Path) -> None:
    for rq in range(1, 8):
        (root / "outputs" / "figures" / f"rq{rq}").mkdir(parents=True, exist_ok=True)
        (root / "outputs" / "tables" / f"rq{rq}").mkdir(parents=True, exist_ok=True)
    (root / "outputs" / "reports").mkdir(parents=True, exist_ok=True)


def main() -> None:
    load_project_env()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    ensure_dirs(ROOT)
    gold = ROOT / cfg["paths"]["gold_dir"]
    fig = ROOT / "outputs" / "figures"
    tbl = ROOT / "outputs" / "tables"
    fig_fmt = _figure_format(cfg)

    results = pd.read_csv(gold / "gold_model_results.csv")
    preds = pd.read_csv(gold / "gold_test_predictions.csv")
    ablation = pd.read_csv(gold / "gold_ablation_rq2.csv")
    imbalance = pd.read_csv(gold / "gold_imbalance_rq4.csv")
    fi = pd.read_csv(gold / "xgb_feature_importance.csv")
    silver_path = ROOT / cfg["paths"]["silver_dir"] / "silver_features_clean.parquet"
    sdf = pd.read_parquet(silver_path)

    # --- RQ1 (measured pipeline timing from pipeline_timing_latest.csv) ---
    timing_path = gold / "pipeline_timing_latest.csv"
    if timing_path.exists():
        timing_df = pd.read_csv(timing_path)
        pd.DataFrame(
            {
                "Stage": timing_df["step_name"],
                "Duration (min)": timing_df["duration_min"].round(3),
            }
        ).to_csv(tbl / "rq1" / "table_1_1_pipeline_latency.csv", index=False)
        train_min = float(timing_df["duration_min"].sum())
    else:
        train_min = float(results["train_minutes"].sum())
        pd.DataFrame(
            {
                "Stage": ["Model training (only measured stage)"],
                "Duration (min)": [round(train_min, 3)],
            }
        ).to_csv(tbl / "rq1" / "table_1_1_pipeline_latency.csv", index=False)

    preds["obs_date"] = pd.to_datetime(preds["obs_date"], errors="coerce")
    daily = (
        preds.groupby(preds["obs_date"].dt.date)
        .agg(
            predictions=("y_true", "count"),
            fire_events=("y_true", "sum"),
            avg_xgb_prob=("XGB_prob", "mean"),
        )
        .reset_index()
        .rename(columns={"obs_date": "date"})
        .tail(10)
    )
    daily.to_csv(tbl / "rq1" / "table_1_2_daily_prediction_stability.csv", index=False)

    plt.figure(figsize=(9, 3))
    plt.axis("off")
    regions = sdf["region_label"].unique() if "region_label" in sdf.columns else ["multi-region"]
    plt.text(0.02, 0.55, f"GEE ({', '.join(regions)}) -> Bronze -> Silver -> Gold ML -> Dashboard", fontsize=11)
    plt.tight_layout()
    save_thesis_figure(fig, "rq1", "figure_1_1_system_architecture", fig_fmt)

    plt.figure(figsize=(10, 4))
    if timing_path.exists():
        timing_df = pd.read_csv(timing_path)
        stages = timing_df["step_name"].tolist()
        durations = timing_df["duration_min"].tolist()
    else:
        stages = ["Model training"]
        durations = [train_min]
    plt.barh(stages, durations, color="#4C72B0")
    plt.xlabel("Minutes (measured, latest pipeline run)")
    plt.tight_layout()
    save_thesis_figure(fig, "rq1", "figure_1_2_dataflow_timeline", fig_fmt)

    # --- RQ2 (report actual ablation direction from data) ---
    ablation.to_csv(tbl / "rq2" / "table_2_1_model_performance.csv", index=False)
    best_row = ablation.loc[ablation["pr_auc"].idxmax()]
    combined_row = ablation[ablation["dataset"] == "Combined"].iloc[0]
    baseline_name = best_row["dataset"]
    delta_pct = 100 * (combined_row["pr_auc"] - best_row["pr_auc"]) / max(best_row["pr_auc"], 1e-9)
    pd.DataFrame(
        {
            "Comparison": [f"Combined vs best ({baseline_name})"],
            "PR-AUC delta (%)": [round(delta_pct, 2)],
            "Best feature set (PR-AUC)": [baseline_name],
            "Combined PR-AUC": [round(combined_row["pr_auc"], 4)],
        }
    ).to_csv(tbl / "rq2" / "table_2_2_contribution_gain.csv", index=False)

    plt.figure(figsize=(8, 4))
    x = np.arange(len(ablation))
    w = 0.35
    plt.bar(x - w / 2, ablation["pr_auc"], w, label="PR-AUC")
    plt.bar(x + w / 2, ablation["f1_score"], w, label="F1")
    plt.xticks(x, ablation["dataset"], rotation=15)
    plt.legend()
    plt.tight_layout()
    save_thesis_figure(fig, "rq2", "figure_2_1_performance_bar", fig_fmt)

    corr_cols = [c for c in ["ndvi", "evi", "temperature_2m", "total_precipitation", "dewpoint_temperature_2m"] if c in sdf.columns]
    if len(corr_cols) >= 2:
        plt.figure(figsize=(6, 5))
        sns.heatmap(sdf[corr_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
        plt.tight_layout()
        save_thesis_figure(fig, "rq2", "figure_2_2_feature_correlation_heatmap", fig_fmt)

    # --- RQ3 ---
    plt.figure(figsize=(8, 5))
    top = fi.head(10)
    sns.barplot(data=top, x="importance", y="feature", color="#55A868")
    plt.tight_layout()
    save_thesis_figure(fig, "rq3", "figure_3_1_feature_importance", fig_fmt)

    sample = sdf[sdf["fire_within_7d"] == 1].sort_values("obs_date")
    if sample.empty:
        sample = sdf.sort_values("obs_date")
    sample = sample.head(400)
    if len(sample) > 1:
        plt.figure(figsize=(10, 4))
        plt.plot(sample["obs_date"], sample["ndvi"], label="NDVI", alpha=0.7)
        fires = sample[sample["fire_within_7d"] == 1]
        if len(fires):
            plt.scatter(fires["obs_date"], fires["ndvi"], s=20, c="red", label="Fire within 7d", zorder=5)
        plt.legend()
        plt.ylabel("NDVI")
        plt.tight_layout()
        save_thesis_figure(fig, "rq3", "figure_3_2_temporal_trends", fig_fmt)

    veg_only = ablation[ablation["dataset"] == "NDVI Only"].iloc[0]
    weather_only = ablation[ablation["dataset"] == "Weather Only"].iloc[0]
    combined = ablation[ablation["dataset"] == "Combined"].iloc[0]
    pd.DataFrame(
        {
            "Feature Set": ["Weather Only", "NDVI Only", "Combined"],
            "F1 Score": [weather_only["f1_score"], veg_only["f1_score"], combined["f1_score"]],
            "PR-AUC": [weather_only["pr_auc"], veg_only["pr_auc"], combined["pr_auc"]],
        }
    ).to_csv(tbl / "rq3" / "table_3_1_feature_ablation.csv", index=False)
    fi.head(3).to_csv(tbl / "rq3" / "table_3_2_top_features.csv", index=False)

    # --- RQ4 ---
    plt.figure(figsize=(8, 5))
    y_true = preds["y_true"].values
    for name in ["LR", "XGB"]:
        col = f"{name}_prob"
        if col in preds.columns and len(np.unique(y_true)) > 1:
            p, r, _ = precision_recall_curve(y_true, preds[col].values)
            plt.plot(r, p, label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    save_thesis_figure(fig, "rq4", "figure_4_1_pr_curves", fig_fmt)

    imbalance.to_csv(tbl / "rq4" / "table_4_1_strategy_comparison.csv", index=False)

    plt.figure(figsize=(9, 4))
    ax1 = plt.subplot(1, 2, 1)
    if "LR_pred" in preds.columns:
        cm1 = confusion_matrix(y_true, preds["LR_pred"].values)
        sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=ax1)
    ax1.set_title("LR predictions")
    ax2 = plt.subplot(1, 2, 2)
    if "XGB_pred" in preds.columns:
        cm2 = confusion_matrix(y_true, preds["XGB_pred"].values)
        sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=ax2)
    ax2.set_title("XGB predictions")
    plt.tight_layout()
    save_thesis_figure(fig, "rq4", "figure_4_2_confusion_matrix_comparison", fig_fmt)

    fn_none, fn_bal, reduction = 0, 0, 0.0
    imb_preds_path = gold / "gold_imbalance_predictions.csv"
    if imb_preds_path.exists():
        imb_preds = pd.read_csv(imb_preds_path)
        y = imb_preds[imb_preds["imbalance_mode"] == "none"]
        b = imb_preds[imb_preds["imbalance_mode"] == "balanced"]
        fn_none = int(((y["y_true"] == 1) & (y["XGB_pred"] == 0)).sum())
        fn_bal = int(((b["y_true"] == 1) & (b["XGB_pred"] == 0)).sum())
        reduction = 100 * (fn_none - fn_bal) / max(fn_none, 1)
        pd.DataFrame(
            {
                "Method": ["No class weight", "Class weight (balanced)"],
                "FN Count (threshold=0.5)": [fn_none, fn_bal],
                "FN reduction vs none (%)": ["0.0", f"{reduction:.1f}"],
            }
        ).to_csv(tbl / "rq4" / "table_4_2_false_negative_reduction.csv", index=False)
    else:
        fn_bal = int(((preds["y_true"] == 1) & (preds["XGB_pred"] == 0)).sum())
        pd.DataFrame(
            {"Method": ["Class weight (balanced)"], "FN Count (threshold=0.5)": [fn_bal], "FN reduction vs none (%)": ["-"]}
        ).to_csv(tbl / "rq4" / "table_4_2_false_negative_reduction.csv", index=False)

    # --- RQ5 ---
    tbl5 = results[["model_name", "f1_score", "pr_auc", "roc_auc"]].copy()
    tbl5.columns = ["Model", "F1", "PR-AUC", "ROC-AUC"]
    tbl5.to_csv(tbl / "rq5" / "table_5_1_performance_summary.csv", index=False)
    order = ["LR", "RF", "XGB"]
    ridx = results.set_index("model_name").reindex(order)
    pd.DataFrame(
        {
            "Model": order,
            "Training Time (min)": ridx["train_minutes"].values,
        }
    ).to_csv(tbl / "rq5" / "table_5_2_computational_cost.csv", index=False)

    plt.figure(figsize=(8, 5))
    for name in ["LR", "RF", "XGB"]:
        col = f"{name}_prob"
        if col in preds.columns and len(np.unique(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, preds[col].values)
            plt.plot(fpr, tpr, label=name)
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.legend()
    plt.tight_layout()
    save_thesis_figure(fig, "rq5", "figure_5_1_roc_curves", fig_fmt)

    if "obs_date" in preds.columns and len(preds) > 20:
        preds_ts = preds.copy()
        preds_ts["obs_date"] = pd.to_datetime(preds_ts["obs_date"])
        preds_ts["month"] = preds_ts["obs_date"].dt.to_period("M").astype(str)
        monthly_f1 = []
        for model in ["LR", "RF", "XGB"]:
            col = f"{model}_pred"
            if col not in preds_ts.columns:
                continue
            monthly = preds_ts.groupby("month")[["y_true", col]].apply(
                lambda g: f1_score(g["y_true"], g[col], zero_division=0)
            )
            mf1 = monthly.reset_index(name="f1")
            mf1["model"] = model
            monthly_f1.append(mf1)
        if monthly_f1:
            mf1_df = pd.concat(monthly_f1, ignore_index=True)
            plt.figure(figsize=(8, 4))
            for model in mf1_df["model"].unique():
                sub = mf1_df[mf1_df["model"] == model]
                plt.plot(sub["month"], sub["f1"], marker="o", label=model)
            plt.xticks(rotation=45)
            plt.ylabel("F1 score")
            plt.xlabel("Month")
            plt.legend()
            plt.tight_layout()
            save_thesis_figure(fig, "rq5", "figure_5_2_model_stability_over_time", fig_fmt)

    # --- RQ6 (measured step success rates from pipeline_step_runs.csv) ---
    plt.figure(figsize=(9, 3))
    plt.axis("off")
    plt.text(0.02, 0.55, "ingest -> bronze -> silver -> train -> evaluate -> report", fontsize=13)
    plt.tight_layout()
    save_thesis_figure(fig, "rq6", "figure_6_1_airflow_dag_visualization", fig_fmt)

    runs_path = gold / "pipeline_step_runs.csv"
    if runs_path.exists():
        runs = pd.read_csv(runs_path)
        runs["started_at"] = pd.to_datetime(runs["started_at"], errors="coerce", utc=True)
        runs["week"] = runs["started_at"].dt.to_period("W").astype(str)
        weekly = (
            runs.groupby("week")["status"]
            .apply(lambda s: 100 * (s == "success").mean())
            .reset_index(name="success_rate_pct")
            .tail(4)
        )
        plt.figure(figsize=(7, 4))
        if len(weekly):
            plt.bar(weekly["week"], weekly["success_rate_pct"], color="#C44E52")
            plt.ylabel("Success rate (%)")
            plt.xticks(rotation=30)
        else:
            plt.text(0.5, 0.5, "No pipeline runs logged yet", ha="center")
        plt.tight_layout()
        save_thesis_figure(fig, "rq6", "figure_6_2_pipeline_success_rate", fig_fmt)

        failures = (
            runs[runs["status"] == "failure"]
            .groupby("step_name")
            .agg(failure_count=("status", "count"), avg_recovery_min=("duration_sec", lambda s: s.mean() / 60))
            .reset_index()
            .rename(columns={"step_name": "Task"})
        )
        if failures.empty:
            failures = pd.DataFrame({"Task": ["(none)"], "failure_count": [0], "avg_recovery_min": [0.0]})
        failures.to_csv(tbl / "rq6" / "table_6_1_failure_analysis.csv", index=False)
    else:
        plt.figure(figsize=(7, 4))
        plt.text(0.5, 0.5, "No pipeline_step_runs.csv — run pipeline_real.py first", ha="center")
        plt.axis("off")
        plt.tight_layout()
        save_thesis_figure(fig, "rq6", "figure_6_2_pipeline_success_rate", fig_fmt)
        pd.DataFrame(
            {"Task": ["(no runs logged)"], "failure_count": [0], "avg_recovery_min": [0.0]}
        ).to_csv(tbl / "rq6" / "table_6_1_failure_analysis.csv", index=False)
    pd.DataFrame(
        {
            "Metric": ["Test predictions", "Positive fire labels", "Countries"],
            "Value": [
                len(preds),
                int((preds["y_true"] == 1).sum()),
                preds["region_label"].nunique() if "region_label" in preds.columns else 1,
            ],
        }
    ).to_csv(tbl / "rq6" / "table_6_2_reproducibility_metrics.csv", index=False)

    # --- RQ7 ---
    plt.figure(figsize=(9, 4))
    plt.axis("off")
    plt.text(0.03, 0.7, "Dashboard: PR-AUC | Drift | Daily Runs", fontsize=14)
    plt.text(0.03, 0.45, "Risk alerts and map layers (Power BI / web)", fontsize=12)
    plt.tight_layout()
    save_thesis_figure(fig, "rq7", "figure_7_1_dashboard_snapshot", fig_fmt)

    map_df = preds.dropna(subset=["latitude", "longitude"]).copy()
    if len(map_df) > 0 and "XGB_prob" in map_df.columns:
        sample_map = map_df.sort_values("XGB_prob", ascending=False).head(500)
        plt.figure(figsize=(8, 5))
        sc = plt.scatter(
            sample_map["longitude"],
            sample_map["latitude"],
            c=sample_map["XGB_prob"],
            cmap="hot",
            s=12,
            vmin=0,
            vmax=1,
        )
        plt.colorbar(sc, label="XGB fire probability")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title("Real test-set fire risk (top 500 by probability)")
        plt.tight_layout()
        save_thesis_figure(fig, "rq7", "figure_7_2_risk_map_visualization", fig_fmt)

    if "region_label" in preds.columns:
        region_risk = (
            preds.groupby("region_label")
            .agg(avg_risk=("XGB_prob", "mean"), fire_rate=("y_true", "mean"), n=("y_true", "count"))
            .reset_index()
        )
        scenarios = []
        for _, row in region_risk.iterrows():
            risk = "High" if row["avg_risk"] >= 0.3 else ("Medium" if row["avg_risk"] >= 0.1 else "Low")
            action = "Deploy resources" if risk == "High" else ("Monitor" if risk == "Medium" else "No action")
            scenarios.append(
                {
                    "Scenario": f"{row['region_label']} (avg risk {row['avg_risk']:.1%})",
                    "Predicted Risk": risk,
                    "Action": action,
                }
            )
        pd.DataFrame(scenarios).to_csv(tbl / "rq7" / "table_7_1_decision_scenarios.csv", index=False)
    else:
        pd.DataFrame(
            {
                "Scenario": ["High risk grid cells", "Moderate risk", "Low risk"],
                "Predicted Risk": ["High", "Medium", "Low"],
                "Action": ["Deploy resources", "Monitor", "No action"],
            }
        ).to_csv(tbl / "rq7" / "table_7_1_decision_scenarios.csv", index=False)

    xgb_res = results[results["model_name"] == "XGB"].iloc[0]
    tuned_recall = xgb_res.get("recall_score_best_f1", xgb_res["recall_score"])
    pd.DataFrame(
        {
            "Metric": [
                "Best model PR-AUC",
                "Best model Recall (threshold=0.5)",
                "Best model Recall (tuned F1 threshold)",
                "Test fire events caught (XGB, threshold=0.5)",
                "Test fire events caught (XGB, tuned threshold)",
            ],
            "Value": [
                f"{xgb_res['pr_auc']:.3f}",
                f"{xgb_res['recall_score']:.3f}",
                f"{tuned_recall:.3f}",
                int(((preds["y_true"] == 1) & (preds["XGB_pred"] == 1)).sum()),
                int(((preds["y_true"] == 1) & (preds["XGB_pred_tuned"] == 1)).sum())
                if "XGB_pred_tuned" in preds.columns
                else int(((preds["y_true"] == 1) & (preds["XGB_pred"] == 1)).sum()),
            ],
        }
    ).to_csv(tbl / "rq7" / "table_7_2_summary_evaluation.csv", index=False)

    best = tbl5.sort_values("PR-AUC", ascending=False).iloc[0]
    base_rate = float(sdf["fire_within_7d"].mean())
    rq2_best = ablation.loc[ablation["pr_auc"].idxmax()]
    rq2_combined = ablation[ablation["dataset"] == "Combined"].iloc[0]
    if rq2_combined["pr_auc"] >= rq2_best["pr_auc"]:
        rq2_text = (
            f"Multi-source fusion (Combined PR-AUC={rq2_combined['pr_auc']:.3f}) "
            f"matches or exceeds the best single-source set ({rq2_best['dataset']}, PR-AUC={rq2_best['pr_auc']:.3f})."
        )
    else:
        rq2_text = (
            f"Combined features (PR-AUC={rq2_combined['pr_auc']:.3f}) did **not** beat the best single-source set "
            f"({rq2_best['dataset']}, PR-AUC={rq2_best['pr_auc']:.3f}) in this run — see `gold_ablation_rq2.csv`."
        )

    fn_reduction_text = "See `table_4_2_false_negative_reduction.csv` for measured FN counts at threshold=0.5."
    if imb_preds_path.exists():
        fn_reduction_text = (
            f"At threshold=0.5, balanced class weight changed XGB false negatives from {fn_none} to {fn_bal} "
            f"({reduction:.1f}% reduction vs none)."
        )

    report = [
        "# Research Question Answers (from latest pipeline run)",
        "",
        f"- Regions: `{', '.join(sorted(preds['region_label'].unique())) if 'region_label' in preds.columns else cfg['data'].get('study_area_name')}`",
        f"- Date range: `{cfg['data']['start_date']}` to `{cfg['data']['end_date']}`",
        f"- Dataset base rate (silver): `{100 * base_rate:.2f}%` positive",
        f"- Best model (PR-AUC): `{best['Model']}` = `{best['PR-AUC']:.3f}` (compare to base rate, not 1.0)",
        "",
        "## RQ1",
        "End-to-end pipeline stages are implemented. Step durations are measured in `data/gold/pipeline_timing_latest.csv`.",
        "",
        "## RQ2",
        rq2_text,
        "",
        "## RQ3",
        "Temporal and environmental drivers are ranked in `xgb_feature_importance.csv` and RQ3 figures.",
        "",
        "## RQ4",
        f"Imbalance modes compared in `gold_imbalance_rq4.csv`. {fn_reduction_text}",
        "",
        "## RQ5",
        "LR / RF / XGB compared in `table_5_1_performance_summary.csv`. Threshold-independent PR-AUC/ROC-AUC are primary; "
        "F1 at 0.5 and at validation-tuned thresholds are in `gold_model_results.csv`.",
        "",
        "## RQ6",
        "Airflow DAG runs the same Python steps. Success/failure is logged to `data/gold/pipeline_step_runs.csv`.",
        "",
        "## RQ7",
        "Dashboard reads gold CSVs; decision tables use real test-set predictions.",
        "",
    ]
    (ROOT / "outputs" / "reports" / "research_question_answers.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Thesis outputs generated under outputs/ (figures as .{fig_fmt}).")


if __name__ == "__main__":
    main()
