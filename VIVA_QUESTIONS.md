# Viva questions

### What is financial fraud detection?
It is the process of identifying transactions or behavior that may be unauthorized or suspicious.

### Why is fraud detection difficult?
Fraud is rare, patterns change, and false positives can inconvenience genuine customers.

### Why use an Autoencoder?
It can learn the representation of normal data and flag high reconstruction error as anomalous.

### What is reconstruction error?
It is a numerical measure of the difference between the original input and the Autoencoder's reconstruction.

### Why use LSTM?
LSTM is designed for sequential data and can learn patterns across ordered observations.

### What is class imbalance?
It occurs when one class, such as legitimate transactions, is much more common than the minority fraud class.

### Why not use accuracy alone?
A highly imbalanced dataset can produce high accuracy even when a model fails to detect fraud.

### What is precision?
Among transactions predicted as fraud, precision is the fraction that are actually fraud.

### What is recall?
Among actual fraud transactions, recall is the fraction correctly detected.

### What is F1-score?
The harmonic mean of precision and recall.

### What is a limitation of your LSTM setup?
The selected anonymized dataset lacks a customer/card identifier, so chronological windows are used rather than customer-specific histories.
