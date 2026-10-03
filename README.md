# DriftWatch
Desktop app that detects data drift between training and new datasets using PSI, KS-test and chi-square, with color-coded alerts and a retrain verdict. Built with Python, Tkinter and scikit-learn.

A desktop dashboard that checks whether your new data has drifted away from the data your ML model was trained on.

## Features
- Per-column drift detection: PSI and KS-test for numeric columns, chi-square for categorical columns
- Color-coded status: Stable, Warning, Drift
- Overall verdict: stable, monitor closely, or retrain recommended

## Tech stack
Python, Tkinter, pandas, SciPy, scikit-learn, matplotlib, pytest

## Run the tests
pip install -r requirements.txt
pytest

## License
MIT
