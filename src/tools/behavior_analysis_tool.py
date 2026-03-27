"""Behavior analysis tool: behavioral signals and anomaly flags."""

from __future__ import annotations

import pandas as pd

from src.features.feature_engineering import build_features
from src.schemas.feature_schema import BASE_FEATURES
from src.tools.base import Tool
from src.tools.registry import register_tool

BEHAVIOR_ANALYSIS_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "object",
            "description": "Transaction features (V1..V28, Amount; Time optional for velocity).",
        },
        "recent_transactions": {
            "type": "array",
            "description": "Optional prior transaction dicts for rolling-window context.",
        },
    },
    "required": ["features"],
}

BEHAVIOR_ANALYSIS_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "user_avg_amount": {"type": ["number", "null"]},
        "spending_spike_ratio": {"type": ["number", "null"]},
        "merchant_frequency_score": {"type": ["number", "null"]},
        "transaction_velocity_1h": {"type": ["number", "null"]},
        "transaction_velocity_24h": {"type": ["number", "null"]},
        "is_new_merchant_for_user": {"type": ["number", "null"]},
        "signals": {"type": "object"},
        "flags": {"type": "array", "items": {"type": "string"}},
    },
}

BEHAVIOR_FIELDS = [
    "user_avg_amount",
    "user_med_amount",
    "spending_spike_ratio",
    "merchant_frequency_score",
    "is_new_merchant_for_user",
    "transactions_last_1h",
    "transactions_last_24h",
    "transaction_velocity_1h",
    "transaction_velocity_24h",
    "time_since_last_transaction",
]


def _features_to_df(features: dict) -> pd.DataFrame:
    row = {k: float(features.get(k, 0.0)) for k in BASE_FEATURES}
    row["Time"] = float(features.get("Time", 0.0))
    return pd.DataFrame([row])


def _build_signals_and_flags(row: pd.Series) -> tuple[dict, list[str]]:
    signals: dict = {}
    flags: list[str] = []
    for f in BEHAVIOR_FIELDS:
        if f in row.index and pd.notna(row.get(f)):
            signals[f] = float(row[f])
    if row.get("spending_spike_ratio", 1.0) > 2.0:
        flags.append("spending spike detected")
    if row.get("is_new_merchant_for_user") == 1:
        flags.append("new merchant interaction")
    if row.get("transaction_velocity_1h", 0) >= 5 or row.get("transactions_last_1h", 0) >= 5:
        flags.append("high transaction velocity")
    if 0 < row.get("time_since_last_transaction", 86400) < 60:
        flags.append("very short time since last transaction")
    return signals, flags


def _execute(features: dict, recent_transactions: list[dict] | None = None) -> dict:
    context_frames: list[pd.DataFrame] = []
    if recent_transactions:
        for t in recent_transactions:
            context_frames.append(_features_to_df(t))

    current_df = _features_to_df(features)
    if context_frames:
        full_df = pd.concat([*context_frames, current_df], ignore_index=True)
    else:
        full_df = current_df

    built = build_features(full_df)
    row = built.iloc[-1]
    signals, flags = _build_signals_and_flags(row)
    return {
        "user_avg_amount": signals.get("user_avg_amount"),
        "spending_spike_ratio": signals.get("spending_spike_ratio"),
        "merchant_frequency_score": signals.get("merchant_frequency_score"),
        "transaction_velocity_1h": signals.get("transaction_velocity_1h"),
        "transaction_velocity_24h": signals.get("transaction_velocity_24h"),
        "is_new_merchant_for_user": signals.get("is_new_merchant_for_user"),
        "signals": signals,
        "flags": flags,
    }


behavior_analysis_tool = register_tool(
    Tool(
        name="behavior_analysis",
        description=(
            "Computes behavioral features (spending_spike_ratio, merchant_frequency_score, "
            "transaction_velocity) and returns anomaly flags such as spending spike, "
            "new merchant interaction, and high velocity."
        ),
        input_schema=BEHAVIOR_ANALYSIS_INPUT_SCHEMA,
        output_schema=BEHAVIOR_ANALYSIS_OUTPUT_SCHEMA,
        execute=_execute,
        timeout_seconds=15,
    )
)
