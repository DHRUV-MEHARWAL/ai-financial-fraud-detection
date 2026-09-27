from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from sklearn.utils.class_weight import compute_class_weight

from prepare_data import load_dataset, fit_scaler_on_training_part, save_scaler, make_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo_transactions.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--sequence-length", type=int, default=10)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    df = load_dataset(args.data, args.max_rows)
    train_df, test_df, scaler = fit_scaler_on_training_part(df)
    # Keep one scaler file for the app; train_autoencoder.py and train_lstm.py produce the same one.
    save_scaler(scaler, "models/scaler.joblib")

    # Scale data frame is already scaled by fit_scaler_on_training_part.
    X_train, y_train = make_sequences(train_df, args.sequence_length)
    X_test, y_test = make_sequences(test_df, args.sequence_length)

    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(64),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")])

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(classes, weights)}

    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    history = model.fit(X_train, y_train, validation_split=0.15, epochs=args.epochs, batch_size=256, class_weight=class_weight, shuffle=False, callbacks=callbacks, verbose=1)
    model.save("models/lstm.keras")

    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="train loss")
    plt.plot(history.history["val_loss"], label="validation loss")
    plt.title("LSTM training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/lstm_loss.png", dpi=150)
    plt.close()

    Path("models/lstm_meta.json").write_text(json.dumps({"sequence_length": args.sequence_length}, indent=2))
    print("Saved LSTM model.")


if __name__ == "__main__":
    main()
