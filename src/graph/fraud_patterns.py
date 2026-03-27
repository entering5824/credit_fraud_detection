"""
Graph-level fraud pattern detection.

Operates on a TransactionGraph and identifies multi-transaction patterns
that are invisible when analysing individual transactions in isolation.

Patterns
--------
velocity_burst       User fires many transactions in a short window.
card_testing_attack  Many low-value transactions spread across merchants.
merchant_cluster     A group of merchants shares an unusually high fraud rate.
dormant_account      Account suddenly active after long silence.
high_degree_anomaly  User suddenly transacts at many new merchants.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.graph.transaction_graph import TransactionGraph, get_transaction_graph


# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

_VELOCITY_BURST_THRESHOLD   = 5    # txns in 1 h
_CARD_TEST_AMOUNT_MAX       = 5.0  # USD — low-value probe
_CARD_TEST_COUNT_MIN        = 3    # at least 3 distinct merchants
_MERCHANT_FRAUD_RATE_HIGH   = 0.15 # 15%
_DORMANT_SILENCE_SECONDS    = 30 * 24 * 3600  # 30 days
_HIGH_DEGREE_NEW_MERCHANTS  = 5    # new merchants in 24 h


@dataclass
class PatternSignal:
    pattern:     str
    severity:    str          # low | medium | high | critical
    evidence:    dict[str, Any] = field(default_factory=dict)
    description: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern":     self.pattern,
            "severity":    self.severity,
            "evidence":    self.evidence,
            "description": self.description,
        }


class GraphFraudDetector:
    """
    Detects multi-transaction fraud patterns for a given user.

    Parameters
    ----------
    graph : TransactionGraph (default: module-level singleton)
    """

    def __init__(self, graph: TransactionGraph | None = None) -> None:
        self._graph = graph or get_transaction_graph()

    def detect(
        self,
        user_id: str,
        current_amount: float = 0.0,
        current_merchant_id: str | None = None,
    ) -> list[PatternSignal]:
        """
        Run all pattern detectors for *user_id*.
        Returns a list of PatternSignal objects, sorted by severity.
        """
        signals: list[PatternSignal] = []

        signals += self._detect_velocity_burst(user_id)
        signals += self._detect_card_testing(user_id, current_amount, current_merchant_id)
        signals += self._detect_dormant_account(user_id)
        signals += self._detect_high_degree_anomaly(user_id, current_merchant_id)
        if current_merchant_id:
            signals += self._detect_merchant_cluster(current_merchant_id)

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        return sorted(signals, key=lambda s: severity_order.get(s.severity, 9))

    # ------------------------------------------------------------------ #
    # Individual detectors
    # ------------------------------------------------------------------ #

    def _detect_velocity_burst(self, user_id: str) -> list[PatternSignal]:
        v1h = self._graph.transaction_velocity(user_id, 3600)
        if v1h < _VELOCITY_BURST_THRESHOLD:
            return []
        severity = "critical" if v1h >= 10 else "high"
        return [PatternSignal(
            pattern="velocity_burst",
            severity=severity,
            evidence={"velocity_1h": v1h, "threshold": _VELOCITY_BURST_THRESHOLD},
            description=f"User made {v1h} transactions in the last hour (threshold: {_VELOCITY_BURST_THRESHOLD}).",
        )]

    def _detect_card_testing(
        self, user_id: str, current_amount: float, current_merchant_id: str | None
    ) -> list[PatternSignal]:
        if current_amount > _CARD_TEST_AMOUNT_MAX:
            return []
        recent = self._graph.get_user_transactions(user_id, limit=20)
        low_value = [e for e in recent if e.amount <= _CARD_TEST_AMOUNT_MAX]
        distinct_merchants = len({e.merchant_id for e in low_value})
        if current_merchant_id:
            distinct_merchants += 1
        if distinct_merchants < _CARD_TEST_COUNT_MIN:
            return []
        return [PatternSignal(
            pattern="card_testing_attack",
            severity="critical",
            evidence={
                "low_value_tx_count":  len(low_value),
                "distinct_merchants":  distinct_merchants,
                "amount_threshold":    _CARD_TEST_AMOUNT_MAX,
            },
            description=(
                f"Pattern consistent with card-validity probing: "
                f"{len(low_value)} low-value transactions across {distinct_merchants} merchants."
            ),
        )]

    def _detect_dormant_account(self, user_id: str) -> list[PatternSignal]:
        transactions = self._graph.get_user_transactions(user_id, limit=200)
        if len(transactions) < 2:
            return []
        # Sort by timestamp ascending
        sorted_tx = sorted(transactions, key=lambda e: e.timestamp)
        # Gap between second-to-last and latest historical tx
        if len(sorted_tx) < 2:
            return []
        gap = sorted_tx[-1].timestamp - sorted_tx[-2].timestamp
        if gap < _DORMANT_SILENCE_SECONDS:
            return []
        days = round(gap / 86400)
        return [PatternSignal(
            pattern="dormant_account",
            severity="medium",
            evidence={"silence_days": days, "threshold_days": _DORMANT_SILENCE_SECONDS // 86400},
            description=(
                f"Account was dormant for {days} days before this transaction. "
                "May indicate account takeover."
            ),
        )]

    def _detect_high_degree_anomaly(
        self, user_id: str, current_merchant_id: str | None
    ) -> list[PatternSignal]:
        cutoff = time.time() - 86400
        recent = [
            e for e in self._graph.get_user_transactions(user_id, limit=200)
            if e.timestamp >= cutoff
        ]
        merchants_24h = {e.merchant_id for e in recent}
        if current_merchant_id:
            merchants_24h.add(current_merchant_id)
        count = len(merchants_24h)
        if count < _HIGH_DEGREE_NEW_MERCHANTS:
            return []
        return [PatternSignal(
            pattern="high_degree_anomaly",
            severity="high",
            evidence={"distinct_merchants_24h": count, "threshold": _HIGH_DEGREE_NEW_MERCHANTS},
            description=(
                f"User transacted at {count} distinct merchants in 24 h — "
                "unusually high fan-out may indicate compromised card."
            ),
        )]

    def _detect_merchant_cluster(self, merchant_id: str) -> list[PatternSignal]:
        rate = self._graph.merchant_fraud_rate(merchant_id)
        if rate < _MERCHANT_FRAUD_RATE_HIGH:
            return []
        severity = "critical" if rate >= 0.40 else "high"
        return [PatternSignal(
            pattern="merchant_cluster",
            severity=severity,
            evidence={"merchant_id": merchant_id, "fraud_rate": round(rate, 3)},
            description=(
                f"Merchant '{merchant_id}' has a historical fraud rate of {rate:.1%}, "
                "above the {:.0%} alert threshold.".format(_MERCHANT_FRAUD_RATE_HIGH)
            ),
        )]
