# Report outline

## 1. Title
AI-Based Financial Fraud Detection System

## 2. Abstract
One paragraph describing the FinTech problem, class imbalance, Autoencoder anomaly detection, LSTM sequence modeling, and final risk scoring.

## 3. Problem statement
Traditional rule-based systems can miss new fraud patterns and may create false positives. The project investigates Deep Learning methods for suspicious transaction detection.

## 4. Objectives
- Detect abnormal financial transactions.
- Learn normal transaction representations with an Autoencoder.
- Learn temporal transaction patterns with LSTM.
- Combine both signals into a risk score.
- Evaluate with fraud-focused metrics.

## 5. Dataset
Describe source, size, columns, anonymization and class imbalance. Use the exact statistics of the dataset you actually trained on.

## 6. Preprocessing
Missing-value handling, feature scaling, chronological split, sequence construction.

## 7. Methodology
Autoencoder architecture, anomaly threshold, LSTM architecture, class weighting, combined score.

## 8. Results
Report Precision, Recall, F1, PR-AUC, ROC-AUC and confusion matrix separately for each model and for the combined system.

## 9. Limitations
No customer ID in the anonymized dataset; LSTM sequences are chronological rather than card-specific. The prototype is not a real banking production system.

## 10. Future scope
Real-time streaming, customer-specific sequences, graph neural networks, explainability, adaptive thresholds, and stronger fraud-cost modeling.

## 11. Conclusion
Summarize what the system learned and what the evaluation showed.
