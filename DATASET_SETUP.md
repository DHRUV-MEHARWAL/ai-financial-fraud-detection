# Dataset Setup

## Recommended real dataset

Use the public **Credit Card Fraud Detection** dataset commonly known as `creditcard.csv`.

Known dataset facts:
- 284,807 transactions
- 492 fraud cases
- 284,315 legitimate cases
- Columns: `Time`, `V1` ... `V28`, `Amount`, `Class`
- `Class=1` means fraud and `Class=0` means legitimate

The dataset is widely described in public repositories and originates from a collaboration involving Worldline and the Machine Learning Group at ULB. One public description reports the 284,807/492 split and explains the PCA-transformed V1-V28 features. See the project sources linked in your final answer.

## Download

A simple route is to download the dataset from Kaggle and extract `creditcard.csv` into this project's `data/` directory. Do not commit the large CSV to GitHub.

Expected path:

```text
data/creditcard.csv
```

## Why the starter includes a synthetic CSV

The real dataset is large and redistribution is not assumed here. The included demo dataset lets you run and understand the project immediately without waiting for a separate download.
