"""Unit tests: TransactionStore abstraction and InMemoryTransactionStore."""

from __future__ import annotations

import pytest
from src.tools.transaction_history_tool import (
    InMemoryTransactionStore,
    TransactionStore,
    set_transaction_store,
    get_transaction_store,
    put_user_transaction,
    get_user_history,
)


TX1 = {"Amount": 50.0, "Time": 100.0, "V1": 0.1}
TX2 = {"Amount": 999.0, "Time": 200.0, "V1": -3.2}
TX3 = {"Amount": 5.0, "Time": 300.0, "V1": 0.0}


class TestInMemoryTransactionStore:
    def setup_method(self):
        self.store = InMemoryTransactionStore(max_per_user=3)

    def test_empty_history(self):
        assert self.store.get_user_history("user_x") == []

    def test_put_and_get(self):
        self.store.put_transaction("u1", TX1)
        history = self.store.get_user_history("u1")
        assert len(history) == 1
        assert history[0]["Amount"] == 50.0

    def test_multiple_transactions_ordered(self):
        self.store.put_transaction("u1", TX1)
        self.store.put_transaction("u1", TX2)
        history = self.store.get_user_history("u1")
        assert len(history) == 2
        assert history[-1]["Amount"] == 999.0

    def test_limit_respected(self):
        self.store.put_transaction("u1", TX1)
        self.store.put_transaction("u1", TX2)
        history = self.store.get_user_history("u1", limit=1)
        assert len(history) == 1
        assert history[0]["Amount"] == TX2["Amount"]

    def test_max_per_user_evicts_oldest(self):
        for i in range(5):
            self.store.put_transaction("u1", {"Amount": float(i)})
        history = self.store.get_user_history("u1")
        # max_per_user=3 so only last 3 should survive
        assert len(history) == 3
        assert history[0]["Amount"] == 2.0

    def test_clear_user(self):
        self.store.put_transaction("u1", TX1)
        self.store.clear_user("u1")
        assert self.store.get_user_history("u1") == []

    def test_isolation_between_users(self):
        self.store.put_transaction("u1", TX1)
        self.store.put_transaction("u2", TX2)
        assert len(self.store.get_user_history("u1")) == 1
        assert len(self.store.get_user_history("u2")) == 1

    def test_is_abstract_base(self):
        assert issubclass(InMemoryTransactionStore, TransactionStore)


class TestStoreSwap:
    """Verify that set_transaction_store() replaces the module-level default."""

    def test_swap_and_use_custom_store(self):
        original = get_transaction_store()
        new_store = InMemoryTransactionStore()
        set_transaction_store(new_store)

        put_user_transaction("swap_user", TX3)
        history = get_user_history("swap_user")
        assert len(history) == 1
        assert history[0]["Amount"] == 5.0

        # Restore
        set_transaction_store(original)
