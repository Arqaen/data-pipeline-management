from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, from_unixtime, year, month, dayofmonth, hour
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

spark = (
    SparkSession.builder
    .appName("KafkaToBronzeBatch")
    .getOrCreate()
)

schema = StructType([
    StructField("user_id", IntegerType()),
    StructField("product", StringType()),
    StructField("price", DoubleType()),
    StructField("timestamp", DoubleType()),  # epoch seconds
])

df_kafka = (
    spark.read
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "events")
    .option("startingOffsets", "earliest")
    .load()
)

df = (
    df_kafka
    .selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)

# normalizar a timestamp real
df = df.withColumn(
    "event_time",
    from_unixtime(col("timestamp"))
)

df = (
    df
    .withColumn("year", year("event_time"))
    .withColumn("month", month("event_time"))
    .withColumn("day", dayofmonth("event_time"))
    .withColumn("hour", hour("event_time"))
)

df_count = df.count()
print(f"Rows to write: {df_count}")

if df_count > 0:
    (
        df.write
        .mode("append")
        .partitionBy("year", "month", "day", "hour")
        .parquet("s3a://bronze/eventos_batch")
    )

spark.stop()
