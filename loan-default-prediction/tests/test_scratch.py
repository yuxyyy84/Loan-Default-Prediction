"""Sanity checks: our NumPy model must behave like scikit-learn's.

Run:  python tests/test_scratch.py     (or:  pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.logistic_scratch import LogisticRegressionScratch, sigmoid, log_loss  # noqa: E402


def test_sigmoid_basics():
    assert sigmoid(0) == 0.5
    assert sigmoid(100) > 0.999 and sigmoid(-100) < 0.001


def test_log_loss_perfect_vs_bad():
    y = np.array([1, 0, 1, 0])
    assert log_loss(y, np.array([0.99, 0.01, 0.99, 0.01])) < 0.05
    assert log_loss(y, np.array([0.01, 0.99, 0.01, 0.99])) > 4


def test_loss_decreases():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 3))
    y = (X[:, 0] - X[:, 1] + rng.normal(0, 0.5, 500) > 0).astype(int)
    m = LogisticRegressionScratch(lr=0.1, n_iter=300).fit(X, y)
    assert m.loss_history_[-1] < m.loss_history_[0]


def test_matches_sklearn():
    rng = np.random.default_rng(1)
    n = 2000
    X = rng.normal(size=(n, 4))
    true_w = np.array([1.5, -2.0, 0.0, 0.7])
    y = rng.binomial(1, sigmoid(X @ true_w - 0.5))

    ours = LogisticRegressionScratch(lr=0.2, n_iter=5000, l2=1.0 / n).fit(X, y)
    ref = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)   # C=1  <->  l2 = 1/n

    assert np.allclose(ours.w_, ref.coef_[0], atol=1e-3)
    assert abs(ours.b_ - ref.intercept_[0]) < 1e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
