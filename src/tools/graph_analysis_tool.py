"""
Graph Analysis Tool — detects multi-transaction fraud patterns using
the user-merchant transaction graph.

Input
-----
  user_id      : str    – required
  amount       : float  – current transaction amount (for card-testing detector)
  merchant_id  : str    – current merchant (optional)

Output
------
  patterns       : list of detected PatternSignal dicts
  pattern_count  : int
  graph_summary  : dict (velocity_1h, velocity_24h, avg_amount, …)
  top_pattern    : str  – highest severity pattern name, or "none"
  severity       : str  – highest severity level detected
"""

from __future__ import annotations

from src.graph.fraud_patterns import GraphFraudDetector
from src.graph.transaction_graph import get_transaction_graph
from src.tools.base import Tool
from src.tools.registry import register_tool

_SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}


def _execute(
    user_id: str,
    amount: float = 0.0,
    merchant_id: str | None = None,
    **_: object,
) -> dict:
    graph = get_transaction_graph()
    detector = GraphFraudDetector(graph=graph)

    signals = detector.detect(
        user_id=user_id,
        current_amount=amount,
        current_merchant_id=merchant_id,
    )

    patterns_dicts = [s.to_dict() for s in signals]
    top = patterns_dicts[0] if patterns_dicts else None

    return {
        "patterns":      patterns_dicts,
        "pattern_count": len(patterns_dicts),
        "top_pattern":   top["pattern"]  if top else "none",
        "severity":      top["severity"] if top else "none",
        "graph_summary": graph.summary(user_id),
    }


graph_analysis_tool = register_tool(Tool(
    name="graph_analysis",
    description=(
        "Detects multi-transaction fraud patterns (velocity burst, card testing, "
        "dormant account, merchant cluster) from the transaction graph."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "user_id":     {"type": "string"},
            "amount":      {"type": "number"},
            "merchant_id": {"type": "string"},
        },
        "required": ["user_id"],
    },
    output_schema={
        "type": "object",
        "properties": {
            "patterns":      {"type": "array"},
            "pattern_count": {"type": "integer"},
            "top_pattern":   {"type": "string"},
            "severity":      {"type": "string"},
            "graph_summary": {"type": "object"},
        },
    },
    execute=_execute,
    timeout_seconds=5,
))
