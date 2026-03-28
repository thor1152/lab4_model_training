from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Add src to path so DAGs can import ml_pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from ml_pipeline.data import load_data
from ml_pipeline.v2_model import train_model_v2, promote_model_v2

default_args = {"owner": "airflow", "retries": 1}

with DAG(
    dag_id="ml_training_pipeline_v2",
    default_args=default_args,
    description="Pipeline: train model -> eval model -> promote model",
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
) as dag:

    def train_model_wrapper_v2(
        data_path: str,
        model_path: str,
        metrics_path: str,
        metadata_path: str,
    ):
        df = load_data(data_path)
        return train_model_v2(
            df=df,
            model_path=model_path,
            metrics_path=metrics_path,
            metadata_path=metadata_path,
        )

    train_task = PythonOperator(
        task_id="train_model_v2",
        python_callable=train_model_wrapper_v2,
        op_kwargs={
            "data_path": "data/iris.csv",
            "model_path": "models/iris_model.pkl",
            "metrics_path": "eval/iris_metrics.json",
            "metadata_path": "models/iris_metadata.json",
        },
    )

    def promote_model_wrapper_v2(
        bucket_name: str,
        model_path: str,
        metrics_path: str,
        metadata_path: str,
        accuracy_threshold: float,
    ):
        return promote_model_v2(
            bucket_name=bucket_name,
            model_path=model_path,
            metrics_path=metrics_path,
            metadata_path=metadata_path,
            accuracy_threshold=accuracy_threshold,
        )

    promote_task = PythonOperator(
        task_id="promote_model_v2",
        python_callable=promote_model_wrapper_v2,
        op_kwargs={
            "bucket_name": "your-s3-bucket-name",
            "model_path": "models/iris_model.pkl",
            "metrics_path": "eval/iris_metrics.json",
            "metadata_path": "models/iris_metadata.json",
            "accuracy_threshold": 0.94,
        },
    )

    train_task >> promote_task