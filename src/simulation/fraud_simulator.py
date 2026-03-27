"""
Fraud Simulator — generates synthetic transaction scenarios for agent testing.

Generates three categories:
  normal       – plausible legitimate transactions
  fraud        – realistic fraud transactions matching known patterns
  attack       – structured attack scenarios (card testing, velocity burst, ATO)

Usage
-----
    from src.simulation.fraud_simulator import FraudSimulator

    sim = FraudSimulator(seed=42)

    # Single transaction
    tx = sim.normal_transaction(user_id="u001")
    tx = sim.fraud_transaction(user_id="u001", pattern="velocity_fraud")

    # Full scenario (returns list of events)
    events = sim.attack_scenario("card_testing_attack", user_id="u001", merchant_id="m007")
    events = sim.attack_scenario("velocity_burst", user_id="u002")

    # Bulk dataset
    dataset = sim.generate_dataset(n_normal=1000, n_fraud=100)
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimTransaction:
    """Synthetic transaction ready to feed into the agent."""

    transaction_id: str
    user_id:        str
    merchant_id:    str
    amount:         float
    timestamp:      float
    label:          int            # 0 = legitimate, 1 = fraud
    pattern:        str            # "normal" | pattern name
    features:       dict = field(default_factory=dict)

    def to_event(self) -> dict:
        """Return a flat event dict (Schema A for FraudEventProcessor)."""
        return {
            "transaction_id": self.transaction_id,
            "user_id":        self.user_id,
            "merchant_id":    self.merchant_id,
            "timestamp":      self.timestamp,
            "label":          self.label,
            **self.features,
        }


class FraudSimulator:
    """
    Configurable fraud transaction simulator.

    Parameters
    ----------
    seed        : Random seed for reproducibility.
    n_components: Number of PCA components to simulate (default 28 = V1..V28).
    """

    def __init__(
        self,
        seed: int = 42,
        n_components: int = 28,
    ) -> None:
        self._rng = random.Random(seed)
        self._n = n_components
        self._user_pool  = [f"user_{i:04d}" for i in range(200)]
        self._merch_pool = [f"merch_{i:04d}" for i in range(100)]

    # ------------------------------------------------------------------ #
    # Single transaction generators
    # ------------------------------------------------------------------ #

    def normal_transaction(
        self,
        user_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> SimTransaction:
        """Generate a plausible legitimate transaction."""
        uid = user_id or self._rng.choice(self._user_pool)
        mid = merchant_id or self._rng.choice(self._merch_pool)
        ts  = timestamp or time.time()

        amount = round(self._rng.lognormvariate(3.5, 1.0), 2)  # median ~$33
        features = self._base_features(amount, ts)
        # Normal PCA components: small values near zero
        for i in range(1, self._n + 1):
            features[f"V{i}"] = round(self._rng.gauss(0, 0.5), 4)

        return SimTransaction(
            transaction_id=str(uuid.uuid4()),
            user_id=uid, merchant_id=mid,
            amount=amount, timestamp=ts,
            label=0, pattern="normal",
            features=features,
        )

    def fraud_transaction(
        self,
        user_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        pattern: str = "unknown",
        timestamp: Optional[float] = None,
    ) -> SimTransaction:
        """
        Generate a fraudulent transaction matching a known pattern.

        Supported patterns: velocity_fraud, account_takeover, testing_attack,
        large_anomalous_purchase, unknown.
        """
        uid = user_id or self._rng.choice(self._user_pool)
        mid = merchant_id or self._rng.choice(self._merch_pool)
        ts  = timestamp or time.time()

        if pattern == "testing_attack":
            amount = round(self._rng.uniform(0.5, 5.0), 2)
        elif pattern == "large_anomalous_purchase":
            amount = round(self._rng.uniform(1500, 8000), 2)
        elif pattern in {"velocity_fraud", "account_takeover"}:
            amount = round(self._rng.lognormvariate(5.0, 0.8), 2)
        else:
            amount = round(self._rng.lognormvariate(4.0, 1.2), 2)

        features = self._base_features(amount, ts)
        # Fraud PCA components: wider distribution, some extreme values
        for i in range(1, self._n + 1):
            features[f"V{i}"] = round(self._rng.gauss(0, 2.0), 4)
        # Shift key components to increase fraud probability
        features["V1"]  = round(self._rng.uniform(-5, -1), 4)
        features["V3"]  = round(self._rng.uniform(-4,  0), 4)
        features["V14"] = round(self._rng.uniform(-6, -2), 4)

        return SimTransaction(
            transaction_id=str(uuid.uuid4()),
            user_id=uid, merchant_id=mid,
            amount=amount, timestamp=ts,
            label=1, pattern=pattern,
            features=features,
        )

    # ------------------------------------------------------------------ #
    # Attack scenario builders
    # ------------------------------------------------------------------ #

    def attack_scenario(
        self,
        scenario: str,
        user_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        n_transactions: Optional[int] = None,
    ) -> list[SimTransaction]:
        """
        Build a full multi-transaction attack scenario.

        Supported scenarios
        -------------------
        card_testing_attack  – N low-value probes across distinct merchants.
        velocity_burst       – N rapid transactions at the same merchant.
        account_takeover     – Dormant account + sudden new-merchant spike.
        merchant_fraud_cluster – Multiple users hit the same rogue merchant.

        Returns a time-ordered list of SimTransaction objects.
        """
        uid = user_id or self._rng.choice(self._user_pool)
        now = time.time()

        if scenario == "card_testing_attack":
            count = n_transactions or self._rng.randint(5, 12)
            txs = []
            for i in range(count):
                ts  = now - (count - i) * self._rng.uniform(60, 180)
                mid = self._rng.choice(self._merch_pool)
                txs.append(self.fraud_transaction(uid, mid, "testing_attack", ts))
            return txs

        elif scenario == "velocity_burst":
            count = n_transactions or self._rng.randint(6, 15)
            mid = merchant_id or self._rng.choice(self._merch_pool)
            txs = []
            for i in range(count):
                ts = now - (count - i) * self._rng.uniform(30, 90)
                txs.append(self.fraud_transaction(uid, mid, "velocity_fraud", ts))
            return txs

        elif scenario == "account_takeover":
            # 1. Historical normal transactions (3–5 months ago)
            txs = []
            for i in range(self._rng.randint(3, 8)):
                ts = now - self._rng.uniform(90, 180) * 86400
                txs.append(self.normal_transaction(uid, timestamp=ts))
            # 2. Sudden new-merchant fraud transactions
            count = n_transactions or self._rng.randint(3, 6)
            for i in range(count):
                mid = self._rng.choice(self._merch_pool)
                ts  = now - (count - i) * self._rng.uniform(120, 300)
                txs.append(self.fraud_transaction(uid, mid, "account_takeover", ts))
            return sorted(txs, key=lambda t: t.timestamp)

        elif scenario == "merchant_fraud_cluster":
            mid = merchant_id or self._rng.choice(self._merch_pool)
            count = n_transactions or self._rng.randint(8, 20)
            txs = []
            for i in range(count):
                u = self._rng.choice(self._user_pool)
                ts = now - self._rng.uniform(0, 7 * 86400)
                txs.append(self.fraud_transaction(u, mid, "unknown", ts))
            return sorted(txs, key=lambda t: t.timestamp)

        else:
            raise ValueError(f"Unknown scenario: '{scenario}'. "
                             "Use: card_testing_attack, velocity_burst, account_takeover, "
                             "merchant_fraud_cluster")

    # ------------------------------------------------------------------ #
    # Bulk dataset
    # ------------------------------------------------------------------ #

    def generate_dataset(
        self,
        n_normal: int = 1000,
        n_fraud: int = 100,
        fraud_patterns: Optional[list[str]] = None,
    ) -> list[SimTransaction]:
        """
        Generate a mixed dataset.

        Parameters
        ----------
        n_normal       : Number of legitimate transactions.
        n_fraud        : Number of fraud transactions.
        fraud_patterns : Patterns to distribute fraud across (default: all).
        """
        patterns = fraud_patterns or [
            "velocity_fraud", "account_takeover", "testing_attack",
            "large_anomalous_purchase", "unknown",
        ]
        txs: list[SimTransaction] = []

        for _ in range(n_normal):
            txs.append(self.normal_transaction())

        per_pattern = max(1, n_fraud // len(patterns))
        remainder   = n_fraud - per_pattern * len(patterns)
        for i, p in enumerate(patterns):
            count = per_pattern + (1 if i < remainder else 0)
            for _ in range(count):
                txs.append(self.fraud_transaction(pattern=p))

        self._rng.shuffle(txs)
        return txs

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _base_features(self, amount: float, timestamp: float) -> dict:
        return {"Amount": amount, "Time": timestamp % 86400}
