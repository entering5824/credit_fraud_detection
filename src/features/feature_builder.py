"""
Feature Builder — computes derived features for each feature family and
writes them into the Feature Store.

Feature families
----------------

1. transaction_features   (raw + minimal derived)
   Amount, Time, V1..V28, hour_of_day, is_weekend, time_since_last_tx

2. user_behavior_features (rolling aggregates over recent history)
   transactions_last_5m   count of txns in last 5 minutes
   transactions_last_1h   count of txns in last 1 hour
   transactions_last_24h  count of txns in last 24 hours
   avg_amount_7d          rolling average amount over 7 days
   spending_spike_ratio   current amount / avg_amount_7d
   user_velocity_score    composite velocity signal (0–1)

3. merchant_features      (aggregated per merchant_id)
   merchant_tx_count      total transactions at this merchant
   merchant_avg_amount    average transaction amount
   merchant_fraud_rate    historical fraud rate
   is_high_risk_merchant  flag: merchant_fraud_rate > 0.15

4. graph_features         (from TransactionGraph)
   distinct_merchants_24h  unique merchants in last 24 h
   velocity_1h             transactions in last 1 h (from graph)
   velocity_24h            transactions in last 24 h
   merchant_degree         total distinct merchants ever

Usage
-----
    from src.features.feature_builder import FeatureBuilder

    builder = FeatureBuilder()
    builder.build_and_store(
        tx_id="tx_001",
        raw_features={"Amount": 120.0, "V1": -0.3, ...},
        user_id="user_42",
        merchant_id="merch_07",
        timestamp=1712345678.0,
    )
    fv = builder.store.get_merged("tx_001", user_id="user_42", merchant_id="merch_07")
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Optional

from src.features.feature_store import FeatureStore, get_feature_store


class FeatureBuilder:
    """
    Computes and persists feature families into the Feature Store.

    Parameters
    ----------
    store : FeatureStore implementation (default: module-level singleton)
    """

    def __init__(self, store: Optional[FeatureStore] = None) -> None:
        self.store = store or get_feature_store()

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def build_and_store(
        self,
        tx_id: str,
        raw_features: dict[str, Any],
        user_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
        timestamp: Optional[float] = None,
        fraud_label: Optional[int] = None,
    ) -> None:
        """
        Compute all feature families for a transaction and persist them.
        """
        ts = timestamp or time.time()

        # 1. Transaction features
        tx_feat = self._build_transaction_features(raw_features, ts)
        self.store.put_transaction(tx_id, tx_feat)

        # 2. User behavioral features
        if user_id:
            user_feat = self._build_user_features(user_id, raw_features, ts)
            self.store.put_user_features(user_id, user_feat)

        # 3. Merchant features
        if merchant_id:
            merch_feat = self._build_merchant_features(
                merchant_id, raw_features.get("Amount", 0.0), fraud_label
            )
            self.store.put_merchant_features(merchant_id, merch_feat)

        # 4. Graph features
        if user_id:
            graph_feat = self._build_graph_features(user_id, merchant_id)
            self.store.put_graph_features(user_id, graph_feat)

    # ------------------------------------------------------------------ #
    # Family builders
    # ------------------------------------------------------------------ #

    def _build_transaction_features(
        self, raw: dict[str, Any], ts: float
    ) -> dict[str, Any]:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        feat = dict(raw)
        feat.setdefault("Amount", 0.0)
        feat.setdefault("Time", ts % 86400)
        feat["hour_of_day"]  = dt.hour
        feat["day_of_week"]  = dt.weekday()       # 0=Mon, 6=Sun
        feat["is_weekend"]   = int(dt.weekday() >= 5)
        feat["log_amount"]   = math.log1p(float(feat["Amount"]))
        return feat

    def _build_user_features(
        self, user_id: str, raw: dict[str, Any], ts: float
    ) -> dict[str, Any]:
        """Compute rolling behavioural aggregates from graph history."""
        from src.graph.transaction_graph import get_transaction_graph
        graph = get_transaction_graph()

        amount = float(raw.get("Amount", 0.0))

        tx_1h  = graph.transaction_velocity(user_id, 3600)
        tx_24h = graph.transaction_velocity(user_id, 86400)
        tx_5m  = graph.transaction_velocity(user_id, 300)
        avg_7d = graph.average_amount(user_id) or amount

        return {
            "transactions_last_5m":  tx_5m,
            "transactions_last_1h":  tx_1h,
            "transactions_last_24h": tx_24h,
            "avg_amount_7d":         round(avg_7d, 4),
            "spending_spike_ratio":  round(amount / max(avg_7d, 1.0), 4),
            "user_velocity_score":   round(min(tx_1h / 10.0, 1.0), 4),
            "unique_merchants_24h":  len(set(
                e.merchant_id
                for e in graph.get_user_transactions(user_id, limit=200)
                if e.timestamp >= ts - 86400
            )),
        }

    def _build_merchant_features(
        self,
        merchant_id: str,
        amount: float,
        fraud_label: Optional[int],
    ) -> dict[str, Any]:
        from src.graph.transaction_graph import get_transaction_graph
        graph = get_transaction_graph()

        fraud_rate = graph.merchant_fraud_rate(merchant_id)
        txs = graph.get_merchant_transactions(merchant_id, limit=500)
        tx_count = len(txs)
        avg_amt = sum(t.amount for t in txs) / max(tx_count, 1)

        return {
            "merchant_tx_count":       tx_count,
            "merchant_avg_amount":     round(avg_amt, 4),
            "merchant_fraud_rate":     round(fraud_rate, 4),
            "is_high_risk_merchant":   int(fraud_rate > 0.15),
        }

    def _build_graph_features(
        self, user_id: str, merchant_id: Optional[str]
    ) -> dict[str, Any]:
        from src.graph.transaction_graph import get_transaction_graph
        graph = get_transaction_graph()

        return {
            "distinct_merchants_24h": len(set(
                e.merchant_id
                for e in graph.get_user_transactions(user_id, limit=200)
                if e.timestamp >= time.time() - 86400
            )),
            "velocity_1h":   graph.transaction_velocity(user_id, 3600),
            "velocity_24h":  graph.transaction_velocity(user_id, 86400),
            "merchant_degree": graph.user_degree_centrality(user_id),
        }
