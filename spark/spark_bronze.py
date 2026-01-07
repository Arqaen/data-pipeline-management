from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, when, from_unixtime, year, month, dayofmonth, hour
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, DoubleType

spark = SparkSession.builder.appName("KafkaToBronzeBatch").getOrCreate()

schema = StructType([
    StructField("user_id", IntegerType()),
    StructField("product", StringType()),
    StructField("price", DoubleType()),
    StructField("timestamp", DoubleType()),  # seconds o ms
])

start_ts = int(spark.conf.get("spark.bronze.window.start"))  # ms
end_ts = int(spark.conf.get("spark.bronze.window.end"))      # ms

df_kafka = (
    spark.read.format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "events")
    .option("startingOffsets", "earliest")
    .option("endingOffsets", "latest")
    .load()
)

df = (
    df_kafka.selectExpr("CAST(value AS STRING) AS value")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)

# Normaliza timestamp a ms
df = df.withColumn(
    "event_ts_ms",
    when(col("timestamp") < 1e12, (col("timestamp") * 1000)).otherwise(col("timestamp")).cast("long")
)

# Filtra por ventana
df = df.filter((col("event_ts_ms") >= start_ts) & (col("event_ts_ms") < end_ts))

if df.rdd.isEmpty():
    print("No data to write")
else:
    df = (
        df.withColumn("event_time", from_unixtime(col("event_ts_ms") / 1000))
        .withColumn("year", year("event_time"))
        .withColumn("month", month("event_time"))
        .withColumn("day", dayofmonth("event_time"))
        .withColumn("hour", hour("event_time"))
    )

    (
        df.write.mode("append")
        .partitionBy("year", "month", "day", "hour")
        .parquet("s3a://bronze/eventos_batch")
    )

spark.stop()
