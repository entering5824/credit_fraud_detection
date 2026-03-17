from __future__ import annotations

from typing import Any, Dict, Optional

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

