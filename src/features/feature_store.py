"""
Production Feature Store — stores and retrieves normalised feature vectors
for transactions, users, merchants, and graph signals.

Feature families
----------------
transaction  Raw + engineered transaction features (V1..V28, Amount, Time, …)
user         Behavioural rolling aggregates per user
merchant     Merchant statistics and fraud rate
graph        Graph-derived signals from TransactionGraph

The agent consumes a single merged FeatureVector; it never reads raw events
directly.  This separation enables:
  • Offline feature computation (batch jobs writing to Redis/DB)
  • Consistent train/serve feature parity
  • Easy feature versioning

Architecture
------------
FeatureStore (ABC)
    └── InMemoryFeatureStore    ← in-process (dev/test)
    └── (future) RedisFeatureStore, PostgresFeatureStore

Usage
-----
    from src.features.feature_store import get_feature_store

    store = get_feature_store()
    store.put_transaction("tx_001", {"Amount": 120.0, "V1": -0.3, ...})
    store.put_user_features("user_42", {"transactions_last_5m": 3, ...})

    fv = store.get_merged("tx_001", user_id="user_42", merchant_id="merch_7")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Feature vector dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureVector:
    """Merged feature vector ready for ML scoring."""

    transaction_id:      Optional[str]
    transaction_features: dict[str, Any] = field(default_factory=dict)
    user_features:       dict[str, Any]  = field(default_factory=dict)
    merchant_features:   dict[str, Any]  = field(default_factory=dict)
    graph_features:      dict[str, Any]  = field(default_factory=dict)

    def merged(self) -> dict[str, Any]:
        """Flat dict for model input — transaction features take priority."""
        out: dict[str, Any] = {}
        out.update(self.graph_features)
        out.update(self.merchant_features)
        out.update(self.user_features)
        out.update(self.transaction_features)  # highest priority
        return out

    def to_dict(self) -> dict:
        return {
            "transaction_id":       self.transaction_id,
            "transaction_features": self.transaction_features,
            "user_features":        self.user_features,
            "merchant_features":    self.merchant_features,
            "graph_features":       self.graph_features,
        }


# ---------------------------------------------------------------------------
# Abstract feature store
# ---------------------------------------------------------------------------

class FeatureStore(ABC):
    @abstractmethod
    def put_transaction(self, tx_id: str, features: dict[str, Any]) -> None: ...

    @abstractmethod
    def put_user_features(self, user_id: str, features: dict[str, Any]) -> None: ...

    @abstractmethod
    def put_merchant_features(self, merchant_id: str, features: dict[str, Any]) -> None: ...

    @abstractmethod
    def put_graph_features(self, user_id: str, features: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_transaction(self, tx_id: str) -> Optional[dict[str, Any]]: ...

    @abstractmethod
    def get_user_features(self, user_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def get_merchant_features(self, merchant_id: str) -> dict[str, Any]: ...

    @abstractmethod
    def get_graph_features(self, user_id: str) -> dict[str, Any]: ...

    def get_merged(
        self,
        tx_id: str,
        user_id: Optional[str] = None,
        merchant_id: Optional[str] = None,
    ) -> FeatureVector:
        tx_feat = self.get_transaction(tx_id) or {}
        user_feat   = self.get_user_features(user_id)       if user_id     else {}
        merch_feat  = self.get_merchant_features(merchant_id) if merchant_id else {}
        graph_feat  = self.get_graph_features(user_id)      if user_id     else {}
        return FeatureVector(
            transaction_id=tx_id,
            transaction_features=tx_feat,
            user_features=user_feat,
            merchant_features=merch_feat,
            graph_features=graph_feat,
        )


# ---------------------------------------------------------------------------
# In-memory implementation
# ---------------------------------------------------------------------------

class InMemoryFeatureStore(FeatureStore):
    """LRU-capped in-memory feature store."""

    def __init__(self, max_entries: int = 50_000) -> None:
        self._tx:       OrderedDict = OrderedDict()
        self._users:    OrderedDict = OrderedDict()
        self._merchants: OrderedDict = OrderedDict()
        self._graph:    OrderedDict = OrderedDict()
        self._max = max_entries

    def _put(self, store: OrderedDict, key: str, value: dict) -> None:
        if key in store:
            store.move_to_end(key)
        elif len(store) >= self._max:
            store.popitem(last=False)
        store[key] = value

    def put_transaction(self, tx_id: str, features: dict) -> None:
        self._put(self._tx, tx_id, features)

    def put_user_features(self, user_id: str, features: dict) -> None:
        existing = dict(self._users.get(user_id, {}))
        existing.update(features)
        self._put(self._users, user_id, existing)

    def put_merchant_features(self, merchant_id: str, features: dict) -> None:
        existing = dict(self._merchants.get(merchant_id, {}))
        existing.update(features)
        self._put(self._merchants, merchant_id, existing)

    def put_graph_features(self, user_id: str, features: dict) -> None:
        existing = dict(self._graph.get(user_id, {}))
        existing.update(features)
        self._put(self._graph, user_id, existing)

    def get_transaction(self, tx_id: str) -> Optional[dict]:
        return self._tx.get(tx_id)

    def get_user_features(self, user_id: str) -> dict:
        return dict(self._users.get(user_id, {}))

    def get_merchant_features(self, merchant_id: str) -> dict:
        return dict(self._merchants.get(merchant_id, {}))

    def get_graph_features(self, user_id: str) -> dict:
        return dict(self._graph.get(user_id, {}))


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_default_store: FeatureStore = InMemoryFeatureStore()


def get_feature_store() -> FeatureStore:
    return _default_store


def set_feature_store(store: FeatureStore) -> None:
    global _default_store
    _default_store = store
