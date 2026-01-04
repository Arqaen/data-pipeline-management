from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
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
    StructField("timestamp", DoubleType())
])

start_ts = int(spark.conf.get("spark.bronze.window.start"))
end_ts = int(spark.conf.get("spark.bronze.window.end"))

df_kafka = (
    spark.read
    .format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "events")
    .option("startingOffsetsByTimestamp", start_ts)
    .option("endingOffsetsByTimestamp", end_ts) 
    .load()
)

df_parsed = (
    df_kafka
    .selectExpr("CAST(value AS STRING)")
    .select(from_json(col("value"), schema).alias("data"))
    .select("data.*")
)

df_parsed.write.mode("overwrite").partitionBy("year", "month", "day", "hour").parquet("s3a://bronze/eventos_batch")
