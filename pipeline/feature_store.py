"""
Simple in-memory feature store for fraud pipeline.
Key: transaction_id, Value: dict of features (V1..V28, Amount).
For production, replace with Redis or dedicated feature store.
"""

from typing import Dict, Any, Optional

_store: Dict[str, Dict[str, Any]] = {}


def put(transaction_id: str, features: Dict[str, Any]) -> None:
    _store[transaction_id] = dict(features)


def get(transaction_id: str) -> Optional[Dict[str, Any]]:
    return _store.get(transaction_id)


def delete(transaction_id: str) -> bool:
    if transaction_id in _store:
        del _store[transaction_id]
        return True
    return False


def clear() -> None:
    _store.clear()
