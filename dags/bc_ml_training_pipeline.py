from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Add src to path so DAGs can import ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.bc_data import generate_data_bc, load_data_bc
from ml_pipeline.bc_model import train_model_bc

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="bc_ml_training_pipeline",
    default_args=default_args,
    description="Pipeline: generate breast cancer data -> train breast cancer model",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    generate_task = PythonOperator(
        task_id="generate_data_bc",
        python_callable=generate_data_bc,
        op_kwargs={"output_path": "data/breast_cancer.csv"},
    )

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

    generate_task >> train_task