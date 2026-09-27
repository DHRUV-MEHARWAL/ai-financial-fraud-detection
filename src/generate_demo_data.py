from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

TRANSACTION_TYPES = ["Card", "Online", "ATM", "Bank Transfer", "UPI"]


def generate(n: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2026-01-01 09:00")
    timestamps = start + pd.to_timedelta(rng.integers(0, 120 * 24 * 60, n), unit="m")
    timestamps = pd.Series(timestamps).sort_values().reset_index(drop=True)

    amount = rng.lognormal(mean=6.0, sigma=0.85, size=n)
    previous = np.maximum(amount * rng.uniform(0.35, 1.2, n), 100)
    transactions_today = rng.poisson(4, n) + 1
    new_device = rng.binomial(1, 0.08, n)
    new_location = rng.binomial(1, 0.10, n)
    tx_type = rng.choice(TRANSACTION_TYPES, size=n, p=[0.22, 0.28, 0.10, 0.15, 0.25])

    hour = timestamps.dt.hour + timestamps.dt.minute / 60
    night = (hour < 6) | (hour >= 23)
    ratio = amount / np.maximum(previous, 1)

    # Synthetic fraud rules create realistic, learnable patterns for the demo.
    risk_signal = (
        (amount > 30000).astype(float) * 2.2
        + (ratio > 5).astype(float) * 1.8
        + (night).astype(float) * 1.1
        + new_device * 1.4
        + new_location * 1.1
        + (transactions_today > 10).astype(float) * 1.3
        + np.isin(tx_type, ["Online", "UPI"]).astype(float) * 0.3
    )
    probability = 1 / (1 + np.exp(-(risk_signal - 4.0)))
    y = (rng.random(n) < probability).astype(int)

    df = pd.DataFrame({
        "Amount": amount.round(2),
        "PreviousAmount": previous.round(2),
        "TransactionsToday": transactions_today,
        "NewDevice": new_device,
        "NewLocation": new_location,
        "TransactionType": tx_type,
        "TransactionDate": timestamps.dt.strftime("%Y-%m-%d"),
        "TransactionTime": timestamps.dt.strftime("%H:%M"),
        "Class": y,
    })
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=10000)
    parser.add_argument("--output", default="data/demo_transactions.csv")
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    generate(args.rows).to_csv(out, index=False)
    print(f"Saved {args.rows} demo rows to {out}")
