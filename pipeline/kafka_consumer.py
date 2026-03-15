"""
Consume transaction events from Kafka and write to feature store.
Run: python -m pipeline.kafka_consumer
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from kafka import KafkaConsumer
except ImportError:
    print("Install kafka-python: pip install kafka-python")
    sys.exit(1)

from pipeline.feature_store import put

BOOTSTRAP = "localhost:9092"
TOPIC = "transactions"


def main():
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    )
    for msg in consumer:
        event = msg.value
        tid = event.pop("transaction_id", None)
        if not tid:
            continue
        put(tid, event)
        print(f"Stored {tid}")


if __name__ == "__main__":
    main()
