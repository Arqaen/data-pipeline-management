FROM python:latest

WORKDIR /app

RUN pip install boto3

COPY ../sql/minio_init.py .

CMD ["python", "minio_init.py"]