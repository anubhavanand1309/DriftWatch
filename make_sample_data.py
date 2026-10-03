"""Creates two sample CSV files in the sample_data folder.

old_data.csv = data the model was trained on
new_data.csv = fresh data where some columns have drifted
"""

import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
N = 2000

old = pd.DataFrame(
    {
        "age": rng.normal(30, 6, N).round(0),
        "income": rng.normal(40000, 8000, N).round(0),
        "credit_score": rng.normal(650, 50, N).round(0),
        "city": rng.choice(["Chennai", "Delhi", "Mumbai"], size=N, p=[0.4, 0.3, 0.3]),
        "plan": rng.choice(["Basic", "Premium"], size=N, p=[0.7, 0.3]),
    }
)

new = pd.DataFrame(
    {
        "age": rng.normal(38, 6, N).round(0),  # drifted (older users)
        "income": rng.normal(46000, 8000, N).round(0),  # drifted (higher income)
        "credit_score": rng.normal(650, 50, N).round(0),  # NOT drifted
        "city": rng.choice(["Chennai", "Delhi", "Mumbai"], size=N, p=[0.15, 0.25, 0.6]),  # drifted
        "plan": rng.choice(["Basic", "Premium"], size=N, p=[0.7, 0.3]),  # NOT drifted
    }
)

os.makedirs("sample_data", exist_ok=True)
old.to_csv("sample_data/old_data.csv", index=False)
new.to_csv("sample_data/new_data.csv", index=False)
print("Created sample_data/old_data.csv and sample_data/new_data.csv")
