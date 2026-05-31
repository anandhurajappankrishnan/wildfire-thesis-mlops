"""
Apache Airflow DAG — daily orchestration of the wildfire thesis pipeline.

Deploy inside an Airflow environment; each task runs the matching step script.
For local development, prefer: python src/pipeline_real.py
"""
from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator

# Task graph: GEE → Bronze → SQL (optional) → Silver → Train → Report
with DAG(
    dag_id="wildfire_thesis_pipeline",
    start_date=datetime(2026, 4, 1),
    schedule="@daily",
    catchup=False,
    tags=["thesis", "wildfire", "mlops"],
) as dag:
    ingest = BashOperator(task_id="data_ingestion", bash_command="python src/steps/step1_extract_gee.py")
    bronze = BashOperator(task_id="bronze_layer", bash_command="python src/steps/step2_bronze.py")
    bronze_sql = BashOperator(task_id="bronze_sql_optional", bash_command="python src/steps/step2b_load_sql.py")
    silver = BashOperator(task_id="silver_layer", bash_command="python src/steps/step3_silver.py")
    train = BashOperator(task_id="model_training", bash_command="python src/steps/step4_train_eval.py")
    report = BashOperator(task_id="reporting", bash_command="python src/steps/step5_generate_thesis_outputs.py")

    ingest >> bronze >> bronze_sql >> silver >> train >> report
