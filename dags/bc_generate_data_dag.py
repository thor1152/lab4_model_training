from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), "../scripts"))
from bc_generate_data import generate_data_bc

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="bc_generate_data_only",
    default_args=default_args,
    description="Generate breast cancer dataset only",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    generate_task = PythonOperator(
        task_id="generate_data_bc",
        python_callable=generate_data_bc,
        op_kwargs={"output_path": "data/breast_cancer.csv"},
    )