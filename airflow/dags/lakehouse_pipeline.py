from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "data-platform",
    "start_date": datetime(2024, 1, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

with DAG(
    dag_id="lakehouse_pipeline",
    default_args=default_args,
    schedule="*/5 * * * *",   
    catchup=False,
    max_active_runs=1,
    max_active_tasks=1,
    is_paused_upon_creation=False,
    tags=["lakehouse", "bronze"]
) as dag:

    # bronze = SparkSubmitOperator(
    #     task_id="bronze",
    #     application="/opt/spark/examples/src/main/python/pi.py",
    #     conn_id="spark_default",
    #     name="arrow-spark",
    #     verbose=True,
    #     conf={
    #         "spark.master": "spark://spark-master:7077",
    #     },
    # )
    
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
            "spark.master": "spark://spark-master:7077",
            "spark.sql.shuffle.partitions": "4",

            # ===== Ventana Airflow → Spark =====
            "spark.bronze.window.start": "{{ logical_date.int_timestamp * 1000 }}",
            "spark.bronze.window.end": "{{ (logical_date + macros.timedelta(minutes=5)).int_timestamp * 1000 }}",


            # ===== Kafka =====
            "spark.kafka.bootstrap.servers": "kafka:9092",
            "spark.kafka.topic": "events",

            # ===== Structured Streaming =====
            "spark.sql.streaming.checkpointLocation": "s3a://bronze/_checkpoints",

            # ===== MinIO / S3A =====
            "spark.hadoop.fs.s3a.endpoint": "http://minio1:9000",
            "spark.hadoop.fs.s3a.path.style.access": "true",
            "spark.hadoop.fs.s3a.connection.ssl.enabled": "false",
            "spark.hadoop.fs.s3a.access.key": "minioadmin",
            "spark.hadoop.fs.s3a.secret.key": "minioadmin",

            # ===== JARS OBLIGATORIOS =====
            "spark.jars.packages": (
                "org.apache.hadoop:hadoop-aws:3.3.4,"
                "com.amazonaws:aws-java-sdk-bundle:1.12.262,"
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,"
                "org.apache.kafka:kafka-clients:3.5.1"
            ),
        },
    )

    bronze
