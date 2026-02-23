from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, from_unixtime,
    year, month, dayofmonth, hour
)
from pyspark.sql.types import (
    StructType, StructField,
    IntegerType, StringType, DoubleType
)
import sys


def main():
    spark = (
        SparkSession.builder
        .appName("KafkaToBronzeBatch")
        .config("spark.sql.catalogImplementation", "in-memory")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    # ===== Ventana temporal =====
    window_start = int(spark.conf.get("spark.bronze.window.start"))
    window_end = int(spark.conf.get("spark.bronze.window.end"))

    print(f"Processing window: {window_start} → {window_end}")

    # ===== Esquema =====
    schema = StructType([
        StructField("user_id", IntegerType()),
        StructField("product", StringType()),
        StructField("price", DoubleType()),
        StructField("timestamp", DoubleType()),  # epoch seconds
    ])

    # ===== Kafka BATCH =====
    df_kafka = (
        spark.read
        .format("kafka")
        .option(
            "kafka.bootstrap.servers",
            spark.conf.get("spark.kafka.bootstrap.servers")
        )
        .option(
            "subscribe",
            spark.conf.get("spark.kafka.topic")
        )
        .option("startingOffsets", "earliest")
        .option("endingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # ===== Parse JSON =====
    df = (
        df_kafka
        .selectExpr("CAST(value AS STRING)")
        .select(from_json(col("value"), schema).alias("data"))
        .select("data.*")
    )

    # ===== Timestamp =====
    df = df.withColumn(
        "event_time",
        from_unixtime(col("timestamp"))
    )

    # ===== Ventana Airflow =====
    df = df.filter(
        (col("timestamp") * 1000 >= window_start) &
        (col("timestamp") * 1000 < window_end)
    )

    # ===== Contar UNA vez =====
    rows = df.count()
    print(f"Rows in window: {rows}")

    # if rows == 0:
    #     print("No data for this window")
    #     spark.stop()
    #     sys.exit(0)

    # ===== Particiones =====
    df = (
        df
        .withColumn("year", year("event_time"))
        .withColumn("month", month("event_time"))
        .withColumn("day", dayofmonth("event_time"))
        .withColumn("hour", hour("event_time"))
    )

    # ===== Escritura =====
    (
        df.write
        .mode("append")
        .partitionBy("year", "month", "day", "hour")
        .parquet("s3a://bronze/eventos_batch")
    )

    spark.stop()
    sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Spark job failed:")
        print(e)
        sys.exit(1)
