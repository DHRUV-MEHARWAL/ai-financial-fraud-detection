# Frontend redesign

The old Streamlit app exposed `V1`-`V28` because those were columns in the original anonymized credit-card dataset.
That is not a suitable user interface.

The redesigned app in `src/app.py` now asks for:

- Transaction amount
- Previous transaction amount
- Transaction date
- Transaction time in normal 24-hour `HH:MM` format
- Transaction type
- Number of transactions today
- New/unfamiliar device
- New/unusual location

Behind the scenes, `src/prepare_data.py` converts these human-readable inputs into model features such as:

- hour/day cyclic features
- night flag
- amount-vs-previous ratio
- transaction type code
- device/location signals

The Autoencoder and LSTM are retrained on the redesigned synthetic dataset.

## Rebuild after these changes

From the project root:

```powershell
python src/generate_demo_data.py --rows 10000 --output data/demo_transactions.csv
python src/train_autoencoder.py --data data/demo_transactions.csv
python src/train_lstm.py --data data/demo_transactions.csv
python src/evaluate.py --data data/demo_transactions.csv
streamlit run src/app.py
```

Do not reuse model files trained on the old `V1`-`V28` schema. Retrain both models after replacing the dataset and code.
