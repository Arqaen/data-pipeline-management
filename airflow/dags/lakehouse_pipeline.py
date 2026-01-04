from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow import DAG
from datetime import datetime, timedelta

default_args = {
    "owner": "data-platform",
    "start_date": datetime(2024, 1, 1),
    "retries": 5,
    "retry_delay": timedelta(minutes=1)
}

with DAG(
    dag_id="lakehouse_pipeline",
    default_args=default_args,
    schedule_interval="@hourly",
    catchup=False
) as dag:

    bronze = SparkSubmitOperator(
        task_id="bronze",
        application="/opt/spark-apps/spark_bronze.py",
        conn_id="spark_default",
        conf={
            "spark.bronze.window.start": "{{ data_interval_start.int_timestamp * 1000 }}",
            "spark.bronze.window.end": "{{ data_interval_end.int_timestamp * 1000 }}",
        },
    )

    # silver = SparkSubmitOperator(
    #     task_id="silver",
    #     application="/opt/spark-apps/spark_silver.py",
    #     conn_id="spark_default",
    # )

    # gold = SparkSubmitOperator(
    #     task_id="gold",
    #     application="/opt/spark-apps/spark_gold.py",
    #     conn_id="spark_default",
    # )

    # bronze >> silver >> gold

    bronze
