FROM python:latest

WORKDIR /app

RUN pip install kafka-python

COPY ../kafka/kafka_producer.py .

CMD ["python", "kafka_producer.py"]
