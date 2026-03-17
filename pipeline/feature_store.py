"""
Legacy shim (kept for backward compatibility).

New canonical feature store lives in `src.feature_store.in_memory`.
"""

from src.feature_store.in_memory import put, get, delete, clear  # noqa: F401
