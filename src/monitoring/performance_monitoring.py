from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.core.paths import get_paths


@dataclass(frozen=True)
class PerfRecord:
    ts: float
    endpoint: str
    latency_ms: float
    fraud_probability: float
    prediction: int


def append_perf_record(path: Path, record: PerfRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record.__dict__) + "\n")


def summarize_perf_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"error": f"log not found: {path}"}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"rows": 0}
    df = pd.DataFrame(rows)
    return {
        "rows": int(len(df)),
        "latency_ms_p50": float(df["latency_ms"].quantile(0.50)),
        "latency_ms_p95": float(df["latency_ms"].quantile(0.95)),
        "latency_ms_p99": float(df["latency_ms"].quantile(0.99)),
        "mean_fraud_probability": float(df["fraud_probability"].mean()),
        "alert_rate": float(df["prediction"].mean()),
        "score_p99": float(df["fraud_probability"].quantile(0.99)),
    }


def main() -> None:
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Summarize API performance log (JSONL).")
    parser.add_argument(
        "--log",
        type=str,
        default=str(paths.results_monitoring_dir / "api_perf.jsonl"),
    )
    args = parser.parse_args()
    rep = summarize_perf_log(Path(args.log))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()

