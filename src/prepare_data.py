from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Human-readable transaction fields used by the redesigned FinTech prototype.
RAW_FEATURES = [
    "Amount",
    "PreviousAmount",
    "TransactionsToday",
    "NewDevice",
    "NewLocation",
    "TransactionTypeCode",
    "HourSin",
    "HourCos",
    "DayOfWeekSin",
    "DayOfWeekCos",
    "IsNight",
    "AmountVsPrevious",
]
TARGET = "Class"

TRANSACTION_TYPES = {
    "Card": 0,
    "Online": 1,
    "ATM": 2,
    "Bank Transfer": 3,
    "UPI": 4,
}


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Amount", "PreviousAmount", "TransactionsToday", "NewDevice",
        "NewLocation", "TransactionType", "TransactionDate", "TransactionTime",
        TARGET,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    date = pd.to_datetime(out["TransactionDate"], errors="coerce")
    time = pd.to_datetime(out["TransactionTime"], format="%H:%M", errors="coerce")
    if date.isna().any() or time.isna().any():
        raise ValueError("TransactionDate must be YYYY-MM-DD and TransactionTime must be HH:MM (24-hour format).")

    hour = time.dt.hour + time.dt.minute / 60.0
    weekday = date.dt.dayofweek.astype(float)
    out["TransactionTypeCode"] = out["TransactionType"].map(TRANSACTION_TYPES).fillna(-1).astype(float)
    out["HourSin"] = np.sin(2 * np.pi * hour / 24.0)
    out["HourCos"] = np.cos(2 * np.pi * hour / 24.0)
    out["DayOfWeekSin"] = np.sin(2 * np.pi * weekday / 7.0)
    out["DayOfWeekCos"] = np.cos(2 * np.pi * weekday / 7.0)
    out["IsNight"] = ((hour < 6) | (hour >= 23)).astype(float)
    out["AmountVsPrevious"] = out["Amount"].astype(float) / np.maximum(out["PreviousAmount"].astype(float), 1.0)
    out["NewDevice"] = out["NewDevice"].astype(int)
    out["NewLocation"] = out["NewLocation"].astype(int)
    out["Amount"] = out["Amount"].astype(float)
    out["PreviousAmount"] = out["PreviousAmount"].astype(float)
    out["TransactionsToday"] = out["TransactionsToday"].astype(float)
    out = out.sort_values(["TransactionDate", "TransactionTime"]).reset_index(drop=True)
    return out


def load_dataset(path: str, max_rows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = engineer_features(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=RAW_FEATURES + [TARGET])
    df[TARGET] = df[TARGET].astype(int)
    if max_rows is not None and len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    return df


def fit_scaler_on_training_part(df: pd.DataFrame, train_fraction: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    split = int(len(df) * train_fraction)
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    scaler = StandardScaler()
    train[RAW_FEATURES] = scaler.fit_transform(train[RAW_FEATURES])
    test[RAW_FEATURES] = scaler.transform(test[RAW_FEATURES])
    return train, test, scaler


def save_scaler(scaler: StandardScaler, output_path: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_path)


def make_sequences(df: pd.DataFrame, sequence_length: int = 10):
    X = df[RAW_FEATURES].to_numpy(dtype=np.float32)
    y = df[TARGET].to_numpy(dtype=np.int32)
    sequences = []
    labels = []
    for end in range(sequence_length - 1, len(df)):
        start = end - sequence_length + 1
        sequences.append(X[start : end + 1])
        labels.append(y[end])
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int32)
