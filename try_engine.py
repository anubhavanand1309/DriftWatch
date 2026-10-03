"""Quick check: run the drift engine on the sample CSVs and print the result."""

import pandas as pd

import drift_engine as de

old = pd.read_csv("sample_data/old_data.csv")
new = pd.read_csv("sample_data/new_data.csv")

result = de.analyze(old, new)

pd.set_option("display.width", 120)
print(result.round(4).to_string(index=False))
print()
print(de.overall_verdict(result))
