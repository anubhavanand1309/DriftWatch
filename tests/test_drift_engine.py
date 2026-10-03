import numpy as np
import pandas as pd

import drift_engine as de

rng = np.random.default_rng(42)


def test_psi_same_data_is_near_zero():
    a = rng.normal(50, 10, 2000)
    b = rng.normal(50, 10, 2000)
    assert de.psi(a, b) < 0.1


def test_psi_shifted_data_is_high():
    a = rng.normal(50, 10, 2000)
    b = rng.normal(70, 10, 2000)
    assert de.psi(a, b) > 0.25


def test_ks_detects_shift():
    a = rng.normal(0, 1, 2000)
    b = rng.normal(1, 1, 2000)
    _, p = de.ks_drift(a, b)
    assert p < 0.01


def test_ks_same_data_not_flagged():
    a = rng.normal(0, 1, 2000)
    b = rng.normal(0, 1, 2000)
    _, p = de.ks_drift(a, b)
    assert p > 0.05


def test_chi_square_same_and_changed():
    old = rng.choice(["A", "B", "C"], size=2000, p=[0.5, 0.3, 0.2])
    same = rng.choice(["A", "B", "C"], size=2000, p=[0.5, 0.3, 0.2])
    changed = rng.choice(["A", "B", "C"], size=2000, p=[0.2, 0.3, 0.5])
    assert de.chi_square_drift(old, same)[1] > 0.05
    assert de.chi_square_drift(old, changed)[1] < 0.01


def test_chi_square_handles_new_category():
    old = ["A"] * 100 + ["B"] * 100
    new = ["A"] * 100 + ["B"] * 50 + ["C"] * 50
    _, p = de.chi_square_drift(old, new)
    assert p < 0.05


def test_analyze_and_verdict():
    old = pd.DataFrame(
        {
            "age": rng.normal(30, 5, 2000),
            "score": rng.normal(60, 10, 2000),
            "city": rng.choice(["X", "Y"], size=2000, p=[0.5, 0.5]),
        }
    )
    new = pd.DataFrame(
        {
            "age": rng.normal(45, 5, 2000),  # shifted
            "score": rng.normal(60, 10, 2000),  # same
            "city": rng.choice(["X", "Y"], size=2000, p=[0.8, 0.2]),  # changed
        }
    )
    result = de.analyze(old, new)
    status = dict(zip(result["Column"], result["Status"]))
    assert status["age"] == "Drift"
    assert status["score"] == "Stable"
    assert status["city"] == "Drift"
    assert de.overall_verdict(result).startswith("Retrain recommended")


def test_analyze_no_common_columns_raises():
    a = pd.DataFrame({"x": [1, 2, 3]})
    b = pd.DataFrame({"y": [1, 2, 3]})
    try:
        de.analyze(a, b)
        assert False, "should have raised ValueError"
    except ValueError:
        pass
