# Logistic Regression: the 6 concepts, in plain words

Each concept points to the place in this project where you can see it working.

---

## 1. Probability vs. class prediction (the sigmoid)

A linear model gives any number from -infinity to +infinity. A probability must lie
between 0 and 1. The **sigmoid** function squeezes any number into that range:

```
p = 1 / (1 + e^(-z))        where  z = w1*x1 + w2*x2 + ... + b
```

- z = 0 gives p = 0.5, large positive z gives p near 1, large negative z gives p near 0.
- The model outputs a **probability** (e.g. 0.23 = 23% chance of default). Turning it into a
  yes/no **class** needs a **threshold** (default 0.5, but you can change it, see concept 5).

*See:* `sigmoid()` in `src/logistic_scratch.py`.

---

## 2. Log-loss and gradient descent (how it learns)

**Why not mean squared error?** With a sigmoid, squared error gives a bumpy loss surface.
**Log-loss** (binary cross-entropy) is the loss that comes from maximum likelihood and is convex,
so there is one bottom to find:

```
loss = -mean( y*log(p) + (1-y)*log(1-p) )
```

It punishes confident wrong answers heavily: saying 1% when the truth is "default" costs a lot.

**Gradient descent** repeats four steps: predict, measure the error `(p - y)`, compute the
gradient `X^T (p - y) / n`, and move the weights a small step (the *learning rate*) downhill.

*See:* `LogisticRegressionScratch.fit()` and `reports/loss_curve.png`.
Tip: features are standardised (mean 0, spread 1) so all weights learn at a similar speed.

---

## 3. Regularisation (L1 vs L2)

Adding a penalty on the size of the weights stops the model from over-fitting or from giving
huge weights to correlated features.

| | Penalty | Effect |
|---|---|---|
| **L2 (ridge)** | sum of w² | shrinks all weights smoothly, rarely exactly zero |
| **L1 (lasso)** | sum of abs(w) | can push weights to **exactly zero**, so it acts as feature selection |

In scikit-learn the strength is `C`, the **inverse** of regularisation: small C means strong penalty.

*See:* `reports/l1_vs_l2.png`. On the real data here (120,000 training rows), `C = 0.02` is too weak
to zero anything. A smaller `C` (for example `0.002`) sets `DebtRatio` and `MonthlyIncome` to exactly
zero. Lots of data makes the same penalty matter less.

---

## 4. Coefficients and odds ratios (how to read the model)

- **Odds** = p / (1 - p). If p = 0.2, the odds are 0.25 (1 default for every 4 repayments).
- Logistic regression is *linear in the log-odds*. So a coefficient `w` means: +1 unit of that
  feature adds `w` to the log-odds, i.e. **multiplies the odds by e^w** (the *odds ratio*).
- Odds ratio above 1 raises risk, below 1 lowers it, and exactly 1 means no effect.
- Numeric features here are standardised, so the odds ratio is **per +1 standard deviation**.
  If your data has text columns, their one-hot features are compared with a **reference category**
  (the first level, which is dropped).
- Odds ratios are **not** probability changes, and they show association, not causation.

*See:* `reports/odds_ratios.png` and `reports/odds_ratios.csv`.

---

## 5. Thresholds, precision and recall

|  | Predicted default | Predicted repay |
|---|---|---|
| **Actually defaults** | True positive | **False negative** (missed a defaulter) |
| **Actually repays** | **False positive** (rejected a good customer) | True negative |

- **Precision** = of those we flagged, how many really default.
- **Recall** = of all real defaulters, how many we caught.
- **Lowering the threshold** catches more defaulters (recall up) but flags more good customers
  (precision down). There is no free lunch.
- The right threshold depends on **business cost**. Here we assume missing a defaulter costs
  5x more than rejecting a good customer (`COST_FN`, `COST_FP` in `src/train.py`; change them and
  re-run). The threshold is chosen on **training data only**, then checked on the test set.

*See:* `reports/threshold_tradeoff.png` and `reports/confusion_matrices.png`.

---

## 6. Why accuracy can mislead (class imbalance)

If 93% of customers repay, a "model" that always says *repay* is 93% accurate and catches
**zero** defaulters. In this project the default 0.5 threshold gives 93.7% accuracy, barely above
that lazy baseline (93.3%), while recall is only 16.4%.

Better questions to ask:
- **ROC-AUC**: how well does the model rank risky customers above safe ones (0.5 = random, 1 = perfect)?
- **PR-AUC**: like ROC-AUC but more honest when the positive class is rare.
- **Recall and precision** at the threshold you actually plan to use.

*See:* the printed results table when you run `python src/train.py`.

---

### Quick self-test

1. Why does `p = 0.5` correspond to `z = 0`?
2. A coefficient is 0.4 on a standardised feature. By what factor do the odds change per +1 SD? (e^0.4 = 1.49)
3. You lower the threshold from 0.5 to 0.15. Which goes up, and which goes down?
4. Why must the scaler be fitted on the training data only?
