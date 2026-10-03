"""DriftWatch drift engine.

This file contains ONLY statistics. It has no Tkinter code, so it can be
tested on its own and reused by the GUI later.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Numeric columns
# ---------------------------------------------------------------------------
def psi(old, new, bins=10):
    """Population Stability Index between two numeric samples.

    Rule of thumb: < 0.1 stable, 0.1 to 0.25 warning, > 0.25 drift.
    """
    old = pd.Series(old).dropna().to_numpy(dtype=float)
    new = pd.Series(new).dropna().to_numpy(dtype=float)
    if len(old) == 0 or len(new) == 0:
        raise ValueError("Both columns need at least one non-missing value.")

    # Bin edges come from the OLD data (equal-sized groups of old values).
    edges = np.unique(np.quantile(old, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:  # old column has one repeated value
        return 0.0 if np.allclose(new, old[0]) else 1.0
    edges[0], edges[-1] = -np.inf, np.inf

    old_share = np.histogram(old, bins=edges)[0] / len(old)
    new_share = np.histogram(new, bins=edges)[0] / len(new)

    # Avoid division by zero / log(0) for empty bins.
    old_share = np.clip(old_share, 1e-4, None)
    new_share = np.clip(new_share, 1e-4, None)

    return float(np.sum((new_share - old_share) * np.log(new_share / old_share)))


def ks_drift(old, new):
    """Kolmogorov-Smirnov test. Returns (statistic, p_value).

    A small p-value (below 0.05) means the two samples likely come
    from different distributions.
    """
    old = pd.Series(old).dropna().to_numpy(dtype=float)
    new = pd.Series(new).dropna().to_numpy(dtype=float)
    result = stats.ks_2samp(old, new)
    return float(result.statistic), float(result.pvalue)


# ---------------------------------------------------------------------------
# Categorical columns
# ---------------------------------------------------------------------------
def chi_square_drift(old, new):
    """Chi-square test on category counts. Returns (statistic, p_value)."""
    old = pd.Series(old).dropna().astype(str)
    new = pd.Series(new).dropna().astype(str)

    categories = sorted(set(old) | set(new))
    if len(categories) < 2:
        return 0.0, 1.0

    old_counts = old.value_counts().reindex(categories, fill_value=0)
    new_counts = new.value_counts().reindex(categories, fill_value=0)

    table = np.array([old_counts.to_numpy(), new_counts.to_numpy()])
    chi2, p_value, _, _ = stats.chi2_contingency(table)
    return float(chi2), float(p_value)


# ---------------------------------------------------------------------------
# Status labels and the full analysis
# ---------------------------------------------------------------------------
def numeric_status(psi_value):
    if psi_value < 0.1:
        return "Stable"
    if psi_value < 0.25:
        return "Warning"
    return "Drift"


def categorical_status(p_value):
    if p_value > 0.05:
        return "Stable"
    if p_value > 0.01:
        return "Warning"
    return "Drift"


def analyze(df_old, df_new):
    """Compare every column that exists in both DataFrames.

    Returns a DataFrame with: Column, Type, Method, Score, P-Value, Status.
    """
    common = [c for c in df_old.columns if c in df_new.columns]
    if not common:
        raise ValueError("The two datasets have no column names in common.")

    rows = []
    for col in common:
        both_numeric = pd.api.types.is_numeric_dtype(
            df_old[col]
        ) and pd.api.types.is_numeric_dtype(df_new[col])

        if both_numeric:
            score = psi(df_old[col], df_new[col])
            _, p_value = ks_drift(df_old[col], df_new[col])
            rows.append([col, "Numeric", "PSI + KS", score, p_value, numeric_status(score)])
        else:
            score, p_value = chi_square_drift(df_old[col], df_new[col])
            rows.append([col, "Categorical", "Chi-square", score, p_value, categorical_status(p_value)])

    return pd.DataFrame(
        rows, columns=["Column", "Type", "Method", "Score", "P-Value", "Status"]
    )


def overall_verdict(result):
    """Turn the per-column table into one sentence for the user."""
    total = len(result)
    drifted = int((result["Status"] == "Drift").sum())
    warned = int((result["Status"] == "Warning").sum())

    if drifted / total >= 0.3:
        return f"Retrain recommended: {drifted} of {total} columns show drift."
    if drifted > 0 or warned > 0:
        return f"Monitor closely: {drifted} drift, {warned} warning out of {total} columns."
    return f"Stable: no drift found in {total} columns."
