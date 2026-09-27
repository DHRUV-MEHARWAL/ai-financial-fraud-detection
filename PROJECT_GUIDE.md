# How to proceed — beginner path

## Stage 1: Understand the idea

Learn these words before coding the models:

- transaction
- fraud label
- feature
- anomaly
- sequence
- autoencoder
- encoder / decoder
- reconstruction error
- LSTM
- time step
- precision / recall / F1

## Stage 2: Run the included demo

1. Create a Python environment.
2. Install `requirements.txt`.
3. Run the two training scripts.
4. Run `evaluate.py`.
5. Open the Streamlit app.

## Stage 3: Move to the real dataset

Download `creditcard.csv` and place it in `data/`.
Run with `--max-rows 100000` first. Once everything works, increase the row count.

## Stage 4: Understand the Autoencoder

Your explanation in the viva:

> The Autoencoder learns a compact representation of normal transactions. It reconstructs the input. When an unusual transaction is passed through the trained network, the reconstruction error tends to be higher, so a threshold can be used for anomaly detection.

## Stage 5: Understand the LSTM

Your explanation:

> LSTM is used to learn dependencies across ordered transactions. Instead of looking only at one transaction, the model receives a short sequence of transactions and predicts whether the latest transaction is suspicious.

## Stage 6: Combine the models

This starter uses:

`risk = 0.5 * autoencoder_score + 0.5 * lstm_probability`

This is a simple educational fusion rule. For the report, say that the weights and threshold are hyperparameters that should be validated on a validation set.

## Stage 7: Improve the project

Good future improvements:

- build sequences by customer/card ID when such an identifier is available
- use time gaps and velocity features
- tune the decision threshold for high recall
- compare Autoencoder-only, LSTM-only and combined results
- add a baseline model such as Logistic Regression or Random Forest
- add model explainability
- save experiments and metrics

## Viva-ready limitation

The anonymized credit-card dataset does not provide customer/card identifiers, so this project's LSTM uses chronological transaction windows rather than true per-customer behavioral sequences. This is a methodological limitation, not something to hide.
