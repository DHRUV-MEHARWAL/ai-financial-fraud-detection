# EDA checklist

Use Jupyter or VS Code and run these first:

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('../data/demo_transactions.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df['Class'].value_counts())
print(df.describe())
```

Then plot:

```python
sns.countplot(x='Class', data=df)
plt.show()

sns.histplot(data=df, x='Amount', hue='Class', bins=50, element='step')
plt.show()
```

For the real dataset, explain why the fraud class is highly imbalanced and why precision-recall metrics are more informative than raw accuracy.
