from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.core.paths import get_paths


@dataclass(frozen=True)
class SimConfig:
    n_rows: int = 5000
    fraud_rate: float = 0.01
    seed: int = 42


def generate_demo_creditcard_like(cfg: SimConfig = SimConfig()) -> pd.DataFrame:
    """
    Generate a creditcard.csv-like dataset:
      Time, V1..V28, Amount, Class

    This is for demos / portfolio runs (NOT to claim real performance).
    """
    rng = np.random.default_rng(cfg.seed)
    n = int(cfg.n_rows)
    y = (rng.random(n) < float(cfg.fraud_rate)).astype(int)

    # Time as seconds since start (monotonic)
    inter_arrival = rng.exponential(scale=30.0, size=n)  # avg 30s between txs
    time = np.cumsum(inter_arrival).astype(float)

    # V features: normal, with fraud shifted to create signal
    X = rng.normal(0.0, 1.0, size=(n, 28)).astype(float)
    # fraud pattern: heavier tails + shifts on a few components
    fraud_idx = np.where(y == 1)[0]
    if fraud_idx.size:
        X[fraud_idx, :5] += rng.normal(2.0, 0.8, size=(fraud_idx.size, 5))
        X[fraud_idx, 10:15] += rng.normal(-2.0, 0.8, size=(fraud_idx.size, 5))
        X[fraud_idx] += rng.normal(0.0, 0.5, size=X[fraud_idx].shape)

    # Amount: log-normal; fraud tends to be higher
    amount = rng.lognormal(mean=3.5, sigma=0.7, size=n)
    if fraud_idx.size:
        amount[fraud_idx] *= rng.uniform(2.0, 6.0, size=fraud_idx.size)

    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = amount
    df["Class"] = y
    return df


def main() -> None:
    paths = get_paths()
    parser = argparse.ArgumentParser(description="Generate demo creditcard-like CSV.")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--fraud-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=str(paths.data_dir / "raw" / "demo_creditcard.csv"))
    args = parser.parse_args()

    df = generate_demo_creditcard_like(
        SimConfig(n_rows=int(args.rows), fraud_rate=float(args.fraud_rate), seed=int(args.seed))
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("Saved:", str(out))


if __name__ == "__main__":
    main()

