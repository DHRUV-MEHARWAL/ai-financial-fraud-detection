from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, roc_auc_score, precision_recall_curve

from prepare_data import load_dataset, fit_scaler_on_training_part, make_sequences, RAW_FEATURES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo_transactions.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    df = load_dataset(args.data, args.max_rows)
    train_df, test_df, _ = fit_scaler_on_training_part(df)

    # LSTM test scores
    lstm = tf.keras.models.load_model("models/lstm.keras")
    lstm_meta = json.loads(Path("models/lstm_meta.json").read_text())
    X_test, y_test = make_sequences(test_df, int(lstm_meta["sequence_length"]))
    lstm_scores = lstm.predict(X_test, verbose=0).ravel()

    # Autoencoder test scores
    ae = tf.keras.models.load_model("models/autoencoder.keras")
    x = test_df[RAW_FEATURES].to_numpy(dtype=np.float32)
    recon = ae.predict(x, verbose=0)
    ae_scores = np.mean(np.square(x - recon), axis=1)
    ae_threshold = json.loads(Path("models/autoencoder_meta.json").read_text())["threshold"]

    # Align scores at the end of each sequence.
    ae_scores = ae_scores[int(lstm_meta["sequence_length"]) - 1 :]
    combined = 0.5 * lstm_scores + 0.5 * np.clip(ae_scores / max(ae_threshold * 2, 1e-8), 0, 1)
    pred = (combined >= 0.5).astype(int)

    report = classification_report(y_test, pred, digits=4, zero_division=0)
    print(report)
    print(f"PR-AUC: {average_precision_score(y_test, combined):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, combined):.4f}")

    cm = confusion_matrix(y_test, pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title("Combined model confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], ["Legitimate", "Fraud"])
    ax.set_yticks([0, 1], ["Legitimate", "Fraud"])
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, int(v), ha="center", va="center")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    precision, recall, _ = precision_recall_curve(y_test, combined)
    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision)
    plt.title("Precision-Recall curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(out_dir / "precision_recall.png", dpi=150)
    plt.close()

    (out_dir / "classification_report.txt").write_text(report)
    print("Saved evaluation artifacts in outputs/.")


if __name__ == "__main__":
    main()
