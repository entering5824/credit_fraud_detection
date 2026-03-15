"""
Demo: produce transaction events to Kafka for the fraud pipeline.
Run Kafka and then: python -m pipeline.kafka_producer_demo
"""

import json
import random
import time

try:
    from kafka import KafkaProducer
except ImportError:
    print("Install kafka-python: pip install kafka-python")
    raise

from pipeline.feature_store import put

BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"


def make_event(transaction_id: str, fraud_like: bool = False) -> dict:
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    if fraud_like:
        event = {f"V{i}": round(random.uniform(-10, 10), 4) for i in range(1, 29)}
        event["Amount"] = round(random.uniform(2000, 5000), 2)
    else:
        event = {f"V{i}": round(random.uniform(-2, 2), 4) for i in range(1, 29)}
        event["Amount"] = round(random.uniform(1, 1000), 2)
    event["transaction_id"] = transaction_id
    return event


def main():
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    for i in range(5):
        tid = f"tx-{int(time.time())}-{i}"
        event = make_event(tid, fraud_like=(i % 2 == 1))
        producer.send(TOPIC, value=event)
        put(tid, {k: v for k, v in event.items() if k != "transaction_id"})
        print(f"Sent {tid}")
        time.sleep(0.5)
    producer.flush()
    print("Done.")


if __name__ == "__main__":
    main()
