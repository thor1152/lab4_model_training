from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Ensure src/ is on path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.bc_data import load_data_bc
from ml_pipeline.bc_model import train_model_bc

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="bc_train_model_only",
    default_args=default_args,
    description="Train breast cancer ML model only (expects data to already exist)",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def train_model_wrapper_bc(data_path: str, model_path: str):
        df = load_data_bc(data_path)
        return train_model_bc(df, model_path)

    train_task = PythonOperator(
        task_id="train_model_bc",
        python_callable=train_model_wrapper_bc,
        op_kwargs={
            "data_path": "data/breast_cancer.csv",
            "model_path": "models/breast_cancer_model.pkl",
        },
    )