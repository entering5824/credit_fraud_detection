"""
Transaction Graph — builds and queries a bipartite user-merchant graph.

Nodes: users (u:<user_id>) and merchants (m:<merchant_id>)
Edges: transactions  {amount, timestamp, fraud_flag}

The graph is stored in-memory as adjacency dicts.  In production, swap
for a NetworkX-backed or Neo4j-backed implementation.

Key queries
-----------
  get_user_merchants(user_id)    → merchants the user transacted with
  get_merchant_users(merchant_id)→ users who transacted at the merchant
  transaction_velocity(user_id, window_seconds)  → # txns in window
  user_degree_centrality(user_id)                → distinct merchant count
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TxEdge:
    """Directed edge: user → merchant."""
    user_id:     str
    merchant_id: str
    amount:      float
    timestamp:   float = field(default_factory=time.time)
    fraud_flag:  bool  = False
    tx_id:       Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "user_id":     self.user_id,
            "merchant_id": self.merchant_id,
            "amount":      self.amount,
            "timestamp":   self.timestamp,
            "fraud_flag":  self.fraud_flag,
            "tx_id":       self.tx_id,
        }


class TransactionGraph:
    """
    In-memory bipartite user-merchant transaction graph.

    Parameters
    ----------
    max_edges : hard cap on stored edges (oldest evicted first)
    """

    def __init__(self, max_edges: int = 100_000) -> None:
        # user_id → list of TxEdge
        self._user_edges: dict[str, list[TxEdge]] = defaultdict(list)
        # merchant_id → list of TxEdge
        self._merchant_edges: dict[str, list[TxEdge]] = defaultdict(list)
        # All edges in insertion order
        self._all_edges: list[TxEdge] = []
        self._max = max_edges

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def add_transaction(
        self,
        user_id: str,
        merchant_id: str,
        amount: float,
        timestamp: Optional[float] = None,
        fraud_flag: bool = False,
        tx_id: Optional[str] = None,
    ) -> TxEdge:
        edge = TxEdge(
            user_id=user_id,
            merchant_id=merchant_id,
            amount=amount,
            timestamp=timestamp or time.time(),
            fraud_flag=fraud_flag,
            tx_id=tx_id,
        )
        # Evict oldest if at capacity
        if len(self._all_edges) >= self._max:
            old = self._all_edges.pop(0)
            if old in self._user_edges.get(old.user_id, []):
                self._user_edges[old.user_id].remove(old)
            if old in self._merchant_edges.get(old.merchant_id, []):
                self._merchant_edges[old.merchant_id].remove(old)

        self._all_edges.append(edge)
        self._user_edges[user_id].append(edge)
        self._merchant_edges[merchant_id].append(edge)
        return edge

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #

    def get_user_transactions(
        self, user_id: str, limit: int = 100
    ) -> list[TxEdge]:
        return list(reversed(self._user_edges.get(user_id, [])))[:limit]

    def get_merchant_transactions(
        self, merchant_id: str, limit: int = 100
    ) -> list[TxEdge]:
        return list(reversed(self._merchant_edges.get(merchant_id, [])))[:limit]

    def get_user_merchants(self, user_id: str) -> list[str]:
        """Distinct merchants the user transacted with."""
        return list({e.merchant_id for e in self._user_edges.get(user_id, [])})

    def get_merchant_users(self, merchant_id: str) -> list[str]:
        """Distinct users who transacted at this merchant."""
        return list({e.user_id for e in self._merchant_edges.get(merchant_id, [])})

    def transaction_velocity(
        self, user_id: str, window_seconds: float = 3600.0
    ) -> int:
        """Number of transactions by *user_id* in the last *window_seconds*."""
        cutoff = time.time() - window_seconds
        return sum(1 for e in self._user_edges.get(user_id, []) if e.timestamp >= cutoff)

    def user_degree_centrality(self, user_id: str) -> int:
        """Number of distinct merchants = user's degree in the bipartite graph."""
        return len(self.get_user_merchants(user_id))

    def average_amount(self, user_id: str) -> float:
        edges = self._user_edges.get(user_id, [])
        if not edges:
            return 0.0
        return sum(e.amount for e in edges) / len(edges)

    def merchant_fraud_rate(self, merchant_id: str) -> float:
        edges = self._merchant_edges.get(merchant_id, [])
        if not edges:
            return 0.0
        return sum(1 for e in edges if e.fraud_flag) / len(edges)

    def summary(self, user_id: str) -> dict:
        edges = self._user_edges.get(user_id, [])
        return {
            "user_id":            user_id,
            "total_transactions": len(edges),
            "distinct_merchants": self.user_degree_centrality(user_id),
            "velocity_1h":        self.transaction_velocity(user_id, 3600),
            "velocity_24h":       self.transaction_velocity(user_id, 86400),
            "avg_amount":         round(self.average_amount(user_id), 2),
        }


# Module-level singleton
_default_graph = TransactionGraph()


def get_transaction_graph() -> TransactionGraph:
    return _default_graph
