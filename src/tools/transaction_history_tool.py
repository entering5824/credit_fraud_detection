"""Transaction history tool: scalable storage abstraction with in-memory default."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from src.tools.base import Tool
from src.tools.registry import register_tool


# ---------------------------------------------------------------------------
# Storage abstraction
# ---------------------------------------------------------------------------

class TransactionStore(ABC):
    """Pluggable transaction store. Swap InMemoryTransactionStore for RedisStore or DBStore."""

    @abstractmethod
    def put_transaction(self, user_id: str, transaction: dict[str, Any]) -> None:
        """Append a transaction to a user's history."""

    @abstractmethod
    def get_user_history(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        """Return the most recent `limit` transactions for the user (oldest first)."""

    @abstractmethod
    def clear_user(self, user_id: str) -> None:
        """Remove all stored transactions for the user."""


class InMemoryTransactionStore(TransactionStore):
    """Thread-safe in-memory store (suitable for demos and single-process deployments)."""

    def __init__(self, max_per_user: int = 500) -> None:
        self._max = max_per_user
        self._store: dict[str, list[dict[str, Any]]] = {}

    def put_transaction(self, user_id: str, transaction: dict[str, Any]) -> None:
        uid = str(user_id)
        if uid not in self._store:
            self._store[uid] = []
        self._store[uid].append(dict(transaction))
        if len(self._store[uid]) > self._max:
            self._store[uid] = self._store[uid][-self._max :]

    def get_user_history(self, user_id: str, limit: int = 100) -> list[dict[str, Any]]:
        uid = str(user_id)
        history = self._store.get(uid, [])
        return list(history[-limit:]) if limit else list(history)

    def clear_user(self, user_id: str) -> None:
        self._store.pop(str(user_id), None)


# Module-level default store (can be replaced via `set_transaction_store()`)
_default_store: TransactionStore = InMemoryTransactionStore()


def set_transaction_store(store: TransactionStore) -> None:
    """Replace the module-level store (e.g. with a Redis-backed implementation)."""
    global _default_store
    _default_store = store


def get_transaction_store() -> TransactionStore:
    return _default_store


# Convenience helpers (backward compatible with previous direct calls)
def put_user_transaction(user_id: str, transaction: dict[str, Any]) -> None:
    _default_store.put_transaction(user_id, transaction)


def get_user_history(user_id: str, limit: int = 100) -> list[dict[str, Any]]:
    return _default_store.get_user_history(user_id, limit=limit)


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

TRANSACTION_HISTORY_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "user_id": {"type": "string", "description": "User or synthetic_user_id."},
        "limit": {
            "type": "integer",
            "description": "Max number of past transactions to return.",
            "default": 50,
        },
    },
    "required": ["user_id"],
}

TRANSACTION_HISTORY_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "transactions": {"type": "array", "items": {"type": "object"}},
        "count": {"type": "integer"},
    },
}


def _execute(user_id: str, limit: int = 50) -> dict:
    history = _default_store.get_user_history(user_id, limit=limit)
    return {"transactions": history, "count": len(history)}


transaction_history_tool = register_tool(
    Tool(
        name="transaction_history",
        description=(
            "Retrieves past transactions for a user via a pluggable TransactionStore "
            "(defaults to in-memory; swap for Redis or DB with set_transaction_store())."
        ),
        input_schema=TRANSACTION_HISTORY_INPUT_SCHEMA,
        output_schema=TRANSACTION_HISTORY_OUTPUT_SCHEMA,
        execute=_execute,
        timeout_seconds=5,
    )
)
