"""
Kafka Consumer — reads transaction events and feeds them to FraudEventProcessor.

Configuration via environment variables:
    KAFKA_BOOTSTRAP_SERVERS  – default: localhost:9092
    KAFKA_TOPIC              – default: transactions
    KAFKA_GROUP_ID           – default: fraud-agent-group
    KAFKA_AUTO_OFFSET_RESET  – default: earliest
    FRAUD_ALERT_THRESHOLD    – default: 0.85

If kafka-python is not installed, the consumer gracefully prints a
helpful error and exits rather than crashing on import.

Usage
-----
    # In a separate process / container
    python -m pipeline.stream.kafka_consumer

    # Programmatic
    from pipeline.stream.kafka_consumer import FraudKafkaConsumer
    consumer = FraudKafkaConsumer()
    consumer.run()
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from typing import Any, Callable, Optional

from pipeline.stream.fraud_event_processor import FraudEventProcessor

logger = logging.getLogger(__name__)


class FraudKafkaConsumer:
    """
    Wraps kafka-python KafkaConsumer and feeds events to FraudEventProcessor.

    Parameters
    ----------
    bootstrap_servers : Kafka broker(s).
    topic             : Topic to subscribe to.
    group_id          : Consumer group.
    alert_callback    : Called with the investigation report on high-risk results.
    alert_threshold   : Fraud probability threshold for alerts.
    request_type      : "investigate" | "analyze" — investigation depth.
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        topic: str | None = None,
        group_id: str | None = None,
        auto_offset_reset: str | None = None,
        alert_callback: Optional[Callable[[dict], None]] = None,
        alert_threshold: float | None = None,
        request_type: str = "investigate",
    ) -> None:
        self._servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        self._topic   = topic             or os.getenv("KAFKA_TOPIC",              "transactions")
        self._group   = group_id          or os.getenv("KAFKA_GROUP_ID",           "fraud-agent-group")
        self._offset  = auto_offset_reset or os.getenv("KAFKA_AUTO_OFFSET_RESET",  "earliest")
        self._threshold = alert_threshold or float(os.getenv("FRAUD_ALERT_THRESHOLD", "0.85"))

        from src.agent_runtime.agent_loop import AgentLoop
        self._loop = AgentLoop(workers=4)

        self._processor = FraudEventProcessor(
            agent_loop=self._loop,
            alert_callback=alert_callback,
            alert_threshold=self._threshold,
            request_type=request_type,
        )

        self._running = False

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def run(self) -> None:
        """Start the agent loop and consume messages until stopped."""
        self._loop.start()
        self._running = True
        _register_signals(self.stop)

        logger.info(
            "FraudKafkaConsumer starting — topic=%s group=%s servers=%s",
            self._topic, self._group, self._servers,
        )

        try:
            from kafka import KafkaConsumer
        except ImportError:
            logger.error(
                "kafka-python is not installed. Install it with: pip install kafka-python"
            )
            sys.exit(1)

        consumer = KafkaConsumer(
            self._topic,
            bootstrap_servers=self._servers,
            group_id=self._group,
            auto_offset_reset=self._offset,
            value_deserializer=lambda v: json.loads(v.decode("utf-8")),
            enable_auto_commit=True,
        )

        try:
            for message in consumer:
                if not self._running:
                    break
                event: dict[str, Any] = message.value
                logger.debug(
                    "Received message partition=%d offset=%d",
                    message.partition, message.offset,
                )
                self._processor.process(event)
        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()
            self._loop.stop()
            logger.info(
                "Consumer stopped. Stats: %s",
                self._processor.stats,
            )

    def stop(self, *_: Any) -> None:
        logger.info("Shutdown signal received")
        self._running = False


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

def _register_signals(stop_fn: Callable) -> None:
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, lambda s, f: stop_fn())
        except (OSError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    FraudKafkaConsumer().run()
