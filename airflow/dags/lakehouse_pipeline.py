from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta
from dotenv import load_dotenv
from airflow import DAG
import os

MINIO_USER = os.getenv("MINIO_ROOT_USER")
MINIO_PASS = os.getenv("MINIO_ROOT_PASSWORD")
WINDOW = os.getenv("WINDOW")

default_args = {
    "owner": "data-platform",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


with DAG(
    dag_id="lakehouse_pipeline",
    default_args=default_args,
    schedule=f"*/{WINDOW} * * * *",   
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    is_paused_upon_creation=False,
    tags=["lakehouse", "bronze"]
) as dag:

    bronze = SparkSubmitOperator(
        task_id="bronze",
        application="/opt/spark-apps/spark_bronze.py",
        conn_id="spark_default",
        name="bronze-kafka-to-minio",
        verbose=True,
        execution_timeout=timedelta(minutes=10),
        retries=0,
        conf={
            # ===== Spark =====
            # "spark.master": "spark://spark-master:7077",
            "spark.sql.shuffle.partitions": "4",

            # ===== Ventana Airflow → Spark =====
            "spark.bronze.window.start": "{{ logical_date.int_timestamp * 1000 }}",
            "spark.bronze.window.end": "{{ (logical_date + macros.timedelta(minutes=" + str(WINDOW) + ")).int_timestamp * 1000 }}",


            # ===== Kafka =====
            "spark.kafka.bootstrap.servers": "kafka:9092",
            "spark.kafka.topic": "events",

            # ===== MinIO / S3A =====
            "spark.hadoop.fs.s3a.endpoint": "http://minio1:9000",
            "spark.hadoop.fs.s3a.path.style.access": "true",
            "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
            "spark.hadoop.fs.s3a.access.key": MINIO_USER,
            "spark.hadoop.fs.s3a.secret.key": MINIO_PASS,
            "spark.hadoop.fs.s3a.impl": "org.apache.hadoop.fs.s3a.S3AFileSystem",
            "spark.hadoop.fs.s3a.aws.credentials.provider": "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider",
            "spark.hadoop.mapreduce.fileoutputcommitter.algorithm.version": "2",
            "spark.hadoop.fs.s3a.committer.name": "directory",

            # ===== JARS OBLIGATORIOS =====
            "spark.jars.packages": ",".join([
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1",
                "org.apache.hadoop:hadoop-aws:3.3.2",
                "com.amazonaws:aws-java-sdk-bundle:1.12.262"
            ]),
        },
    )

    bronze
