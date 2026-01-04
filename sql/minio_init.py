import boto3
import os
from dotenv import load_dotenv

load_dotenv()

s3 = boto3.client(
    "s3",
    endpoint_url="http://minio:9000",
    aws_access_key_id=os.getenv("MINIO_ROOT_USER"),
    aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD"),
)

buckets = ["bronze", "silver", "gold"]

for bucket in buckets:
    object_lock = bucket == "bronze"
    try:
        s3.create_bucket(Bucket=bucket)
        s3.put_bucket_versioning(
            Bucket=bucket,
            VersioningConfiguration={"Status": "Enabled"},
            ObjectLockEnabledForBucket=object_lock,
        )
    except Exception as e:
        print(f"Error creating bucket {bucket}: {e}")
