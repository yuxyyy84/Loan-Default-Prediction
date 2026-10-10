"""Logistic Regression written from scratch with NumPy only.

Read this file top to bottom: it is the whole algorithm in ~80 lines.

Model:      p = sigmoid(X @ w + b)                 (probability of default)
Loss:       mean log-loss + (l2 / 2) * ||w||^2      (bias b is not penalised)
Optimiser:  full-batch gradient descent
"""
import numpy as np


def sigmoid(z):
    """Squash any real number into (0, 1). Clipped for numerical safety."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def log_loss(y, p, eps=1e-12):
    """Average negative log-likelihood (a.k.a. binary cross-entropy)."""
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iter=3000, l2=0.0, tol=1e-7):
        self.lr = lr            # learning rate (step size)
        self.n_iter = n_iter    # max number of gradient steps
        self.l2 = l2            # L2 (ridge) regularisation strength
        self.tol = tol          # stop early when the gradient is ~0 (we are at the bottom)
        self.w_ = None
        self.b_ = 0.0
        self.loss_history_ = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        self.w_ = np.zeros(d)
        self.b_ = 0.0
        self.loss_history_ = []

        for _ in range(self.n_iter):
            p = sigmoid(X @ self.w_ + self.b_)          # 1) predict probabilities
            error = p - y                                # 2) how wrong are we?
            grad_w = X.T @ error / n + self.l2 * self.w_ # 3) gradient of the loss
            grad_b = error.mean()
            self.w_ -= self.lr * grad_w                  # 4) step downhill
            self.b_ -= self.lr * grad_b

            loss = log_loss(y, p) + 0.5 * self.l2 * float(self.w_ @ self.w_)
            self.loss_history_.append(loss)
            if np.linalg.norm(np.append(grad_w, grad_b)) < self.tol:   # flat -> converged
                break
        return self

    def predict_proba(self, X):
        """Probability of class 1 (default) for every row."""
        return sigmoid(np.asarray(X, dtype=float) @ self.w_ + self.b_)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
