# AI-Based Financial Fraud Detection System

**Domain:** FinTech  
**Deep Learning Models:** Autoencoder + LSTM

This is a beginner-friendly college project starter. It contains a complete pipeline for:

1. Loading a real credit-card fraud dataset or the included demo dataset
2. Exploratory data analysis
3. Preprocessing and scaling
4. Autoencoder-based anomaly detection
5. LSTM-based sequential fraud detection
6. Combining anomaly and sequence scores into a final risk score
7. Evaluation with precision, recall, F1, PR-AUC and a confusion matrix
8. A Streamlit demo dashboard

## Dataset used

The recommended real dataset is the **Credit Card Fraud Detection** dataset (284,807 transactions; 492 frauds) commonly distributed via Kaggle. Its feature set is `Time`, `V1`-`V28`, `Amount`, and `Class`; the V-features are PCA-transformed/anonymized. Because the real CSV is large and platform terms may apply, it is not redistributed in this starter. See `DATASET_SETUP.md`.

A small synthetic `demo_transactions.csv` is included so the complete project can be run immediately.

## Project structure

```text
ai_financial_fraud_detection/
├── data/
│   ├── demo_transactions.csv
│   └── README_DATA.txt
├── models/
├── outputs/
├── src/
│   ├── generate_demo_data.py
│   ├── prepare_data.py
│   ├── train_autoencoder.py
│   ├── train_lstm.py
│   ├── evaluate.py
│   ├── predict.py
│   └── app.py
├── notebooks/
│   └── 01_eda_notes.md
├── requirements.txt
├── DATASET_SETUP.md
├── PROJECT_GUIDE.md
├── REPORT_OUTLINE.md
├── VIVA_QUESTIONS.md
└── README.md
```

## Recommended environment

Python 3.10 or 3.11 is recommended for a smooth TensorFlow setup.

### Install

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Run immediately using the included demo data

```bash
python src/train_autoencoder.py --data data/demo_transactions.csv
python src/train_lstm.py --data data/demo_transactions.csv
python src/evaluate.py --data data/demo_transactions.csv
streamlit run src/app.py
```

The demo data is intentionally small so you can learn the pipeline quickly. **Do not report demo-data results as results on the real dataset.**

## Run with the real dataset

After downloading `creditcard.csv`, put it here:

```text
data/creditcard.csv
```

Then:

```bash
python src/train_autoencoder.py --data data/creditcard.csv --max-rows 100000
python src/train_lstm.py --data data/creditcard.csv --max-rows 100000
python src/evaluate.py --data data/creditcard.csv --max-rows 100000
streamlit run src/app.py
```

Later, remove `--max-rows` if your computer can handle the full dataset.

## Important academic note

The real Kaggle/ULB dataset does not contain a customer/account identifier. The LSTM in this starter therefore treats **time-ordered consecutive transactions** as a sequence. In your report, explicitly state this assumption as a limitation. A production system would normally build sequences per card/customer/account and would use richer behavioral features.
