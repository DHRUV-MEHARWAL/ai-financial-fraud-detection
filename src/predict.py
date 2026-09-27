from __future__ import annotations

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from prepare_data import RAW_FEATURES, TRANSACTION_TYPES, engineer_features


class FraudDetector:
    def __init__(self):
        self.scaler = joblib.load("models/scaler.joblib")
        self.ae = tf.keras.models.load_model("models/autoencoder.keras")
        self.lstm = tf.keras.models.load_model("models/lstm.keras")
        self.ae_threshold = json.loads(Path("models/autoencoder_meta.json").read_text())["threshold"]
        self.sequence_length = json.loads(Path("models/lstm_meta.json").read_text())["sequence_length"]

    def _prepare(self, transactions: pd.DataFrame) -> pd.DataFrame:
        engineered = engineer_features(transactions)
        x = engineered[RAW_FEATURES].astype(float).copy()
        engineered[RAW_FEATURES] = self.scaler.transform(x)
        return engineered

    def predict(self, transactions: pd.DataFrame) -> dict:
        prepared = self._prepare(transactions)
        xs = prepared[RAW_FEATURES].to_numpy(dtype=np.float32)
        recon = self.ae.predict(xs, verbose=0)
        ae_error = np.mean(np.square(xs - recon), axis=1)
        ae_score = float(np.clip(ae_error[-1] / max(self.ae_threshold * 2, 1e-8), 0, 1))

        seq = xs[-self.sequence_length :]
        if len(seq) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(seq), xs.shape[1]), dtype=np.float32)
            seq = np.vstack([padding, seq])
        lstm_score = float(self.lstm.predict(seq[None, ...], verbose=0).ravel()[0])

        risk = 0.5 * ae_score + 0.5 * lstm_score
        if risk >= 0.70:
            prediction = "FRAUD"
            level = "HIGH"
        elif risk >= 0.45:
            prediction = "REVIEW"
            level = "MEDIUM"
        else:
            prediction = "LEGITIMATE"
            level = "LOW"

        return {
            "autoencoder_anomaly_score": ae_score,
            "lstm_fraud_probability": lstm_score,
            "combined_risk_score": risk,
            "prediction": prediction,
            "risk_level": level,
        }
