from __future__ import annotations

import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, Model

from prepare_data import load_dataset, fit_scaler_on_training_part, save_scaler, RAW_FEATURES


def build_autoencoder(input_dim: int) -> Model:
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(16, activation="relu")(x)
    latent = layers.Dense(8, activation="relu", name="latent")(x)
    x = layers.Dense(16, activation="relu")(latent)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)
    model = Model(inputs, outputs, name="fraud_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/demo_transactions.csv")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)

    df = load_dataset(args.data, args.max_rows)
    train, test, scaler = fit_scaler_on_training_part(df)
    save_scaler(scaler, "models/scaler.joblib")

    # Train on legitimate examples from the training portion.
    normal = train[train["Class"] == 0][RAW_FEATURES].to_numpy(dtype=np.float32)
    x_test = test[RAW_FEATURES].to_numpy(dtype=np.float32)

    model = build_autoencoder(len(RAW_FEATURES))
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)]
    history = model.fit(normal, normal, validation_split=0.15, epochs=args.epochs, batch_size=256, shuffle=True, callbacks=callbacks, verbose=1)
    model.save("models/autoencoder.keras")

    train_pred = model.predict(normal, verbose=0)
    test_pred = model.predict(x_test, verbose=0)
    train_err = np.mean(np.square(normal - train_pred), axis=1)
    test_err = np.mean(np.square(x_test - test_pred), axis=1)
    threshold = float(np.percentile(train_err, 99.5))

    meta = {"threshold": threshold, "input_features": RAW_FEATURES}
    Path("models/autoencoder_meta.json").write_text(json.dumps(meta, indent=2))

    plt.figure(figsize=(8, 4))
    plt.hist(train_err, bins=80, alpha=0.75, label="Normal training")
    plt.axvline(threshold, linestyle="--", label="99.5% threshold")
    plt.title("Autoencoder reconstruction error")
    plt.xlabel("Mean squared reconstruction error")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/autoencoder_error.png", dpi=150)
    plt.close()

    print(f"Saved Autoencoder and threshold={threshold:.6f}")


if __name__ == "__main__":
    main()
