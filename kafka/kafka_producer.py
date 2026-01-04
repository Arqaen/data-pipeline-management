import json, time, random
from kafka import KafkaProducer

producer = KafkaProducer(
    bootstrap_servers="kafka:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

while True:
    evento = {
        "user_id": random.randint(1, 100),
        "product": random.choice(["A", "B", "C"]),
        "price": round(random.uniform(10, 100), 2),
        "timestamp": time.time()
    }
    producer.send("events", evento)
    time.sleep(1)
