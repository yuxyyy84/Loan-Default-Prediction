"""Loan default prediction with Logistic Regression - end-to-end pipeline.

Steps
  1. Load data, stratified train/test split
  2. Preprocess (impute, scale, one-hot) - fitted on TRAIN only
  3. Train our scratch model and scikit-learn's, and check they agree
  4. Evaluate: accuracy is misleading on imbalanced data, so use ROC-AUC, PR-AUC, recall...
  5. Read the model: coefficients -> odds ratios
  6. Choose the decision threshold from business costs (using train data only)
  7. L1 vs L2 regularisation
  8. Save charts, tables and the final model

Usage:  python src/train.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.logistic_scratch import LogisticRegressionScratch  # noqa: E402

DATA_PATH = ROOT / "data" / "loan_data.csv"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
TARGET = "default"
SEED = 42

# Business assumption (change it!): missing a defaulter costs 5x more than
# rejecting a good customer.  Used only to choose the decision threshold.
COST_FN, COST_FP = 5.0, 1.0


# ----------------------------------------------------------------------------
def make_preprocessor(num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), num_cols))
    if cat_cols:   # some datasets (e.g. Give Me Some Credit) have no text columns
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))]),
                      cat_cols))
    return ColumnTransformer(parts)


def feature_names(prep, num_cols, cat_cols):
    names = list(num_cols)
    if cat_cols:
        names += list(prep.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_cols))
    return names


def score(y, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def expected_cost(y, proba, t):
    pred = proba >= t
    fn = int(((~pred) & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    return COST_FN * fn + COST_FP * fp


def save(fig, name):
    fig.tight_layout()
    fig.savefig(REPORTS / name, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    REPORTS.mkdir(exist_ok=True)
    MODELS.mkdir(exist_ok=True)

    # 1) Data -----------------------------------------------------------------
    if not DATA_PATH.exists():
        sys.exit("data/loan_data.csv not found. Put your CSV there (it needs a 0/1 column named 'default').")
    df = pd.read_csv(DATA_PATH)
    X, y = df.drop(columns=TARGET), df[TARGET].values
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    num_cols = [c for c in X.columns if c not in cat_cols]
    print(f"Rows: {len(df):,} | default rate: {y.mean():.1%} | "
          f"missing cells: {int(X.isna().sum().sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)

    # 2) Preprocess (fit on train only -> no leakage) -------------------------
    prep = make_preprocessor(num_cols, cat_cols)
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)
    names = feature_names(prep, num_cols, cat_cols)

    # 3) Scratch model vs scikit-learn ---------------------------------------
    n = len(y_train)
    scratch = LogisticRegressionScratch(lr=0.5, n_iter=100000, l2=1.0 / n, tol=1e-10).fit(Xtr, y_train)
    # C=1 <-> l2 = 1/n.  tol=1e-8: sklearn's default tol (1e-4) stops after ~12 steps, a bit early.
    sk = LogisticRegression(C=1.0, max_iter=5000, tol=1e-8).fit(Xtr, y_train)

    p_scratch = scratch.predict_proba(Xte)
    p_sk = sk.predict_proba(Xte)[:, 1]
    max_coef_gap = float(np.abs(scratch.w_ - sk.coef_[0]).max())
    max_proba_gap = float(np.abs(p_scratch - p_sk).max())
    print(f"\nScratch vs sklearn  -> max |coef gap| = {max_coef_gap:.5f}, "
          f"max |probability gap| = {max_proba_gap:.5f} "
          f"({len(scratch.loss_history_)} gradient steps)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(scratch.loss_history_, color="#3b7ddd")
    ax.set_xscale("log")
    ax.set_xlabel("Gradient descent step (log scale)")
    ax.set_ylabel("Training loss (log-loss + L2)")
    ax.set_title("Scratch model: loss goes down as it learns")
    save(fig, "loss_curve.png")

    # 4) Evaluation at the default 0.5 threshold ------------------------------
    majority_acc = max(y_test.mean(), 1 - y_test.mean())
    res_scratch, res_sk = score(y_test, p_scratch), score(y_test, p_sk)
    table = pd.DataFrame({"Scratch (NumPy)": res_scratch, "scikit-learn": res_sk}).drop("threshold").round(4)
    print(f"\nTest-set results at threshold 0.50  "
          f"(a model that always says 'no default' scores accuracy {majority_acc:.3f}):\n{table}\n")

    fpr, tpr, _ = roc_curve(y_test, p_sk)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {res_sk['roc_auc']:.3f}", color="#3b7ddd")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title("ROC curve (test set)")
    ax.legend()
    save(fig, "roc_curve.png")

    prec, rec, _ = precision_recall_curve(y_test, p_sk)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rec, prec, color="#e07b39", label=f"PR-AUC = {res_sk['pr_auc']:.3f}")
    ax.axhline(y_test.mean(), color="k", ls="--", alpha=0.4, label="Random guess")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)")
    ax.legend()
    save(fig, "pr_curve.png")

    # 5) Read the model: coefficients -> odds ratios --------------------------
    # Numeric features are standardised, so exp(coef) = odds multiplier for +1 SD.
    # One-hot features: exp(coef) = odds multiplier versus the reference category.
    odds = pd.DataFrame({"feature": names, "coefficient": sk.coef_[0]})
    odds["odds_ratio"] = np.exp(odds["coefficient"])
    odds = odds.sort_values("odds_ratio", ascending=False).reset_index(drop=True)
    odds.round(4).to_csv(REPORTS / "odds_ratios.csv", index=False)
    print("Odds ratios (sklearn model):\n", odds.round(3).to_string(index=False), "\n")

    o = odds.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(o["feature"], o["odds_ratio"] - 1, left=1,
            color=np.where(o["odds_ratio"] > 1, "#d9534f", "#2e8b57"))
    ax.axvline(1, color="k", lw=1)
    ax.set_xlabel("Odds ratio  (>1 raises default risk, <1 lowers it)")
    ax.set_title("What drives default?\n(+1 SD for numbers, vs reference level for categories)", fontsize=10)
    save(fig, "odds_ratios.png")

    # 6) Choose the threshold from business costs (TRAIN data only) -----------
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    pipe = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)),
                     ("model", LogisticRegression(C=1.0, max_iter=5000, tol=1e-8))])
    oof = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    grid = np.arange(0.02, 0.90, 0.01)
    costs = [expected_cost(y_train, oof, t) for t in grid]
    best_t = float(grid[int(np.argmin(costs))])

    tuned = score(y_test, p_sk, best_t)
    base_cost = expected_cost(y_test, p_sk, 0.5)
    tuned_cost = expected_cost(y_test, p_sk, best_t)
    print(f"Cost-based threshold (FN={COST_FN:g}x, FP={COST_FP:g}x): {best_t:.2f}")
    print(f"  threshold 0.50 -> recall {res_sk['recall']:.3f}, precision {res_sk['precision']:.3f}, cost {base_cost:.0f}")
    print(f"  threshold {best_t:.2f} -> recall {tuned['recall']:.3f}, precision {tuned['precision']:.3f}, cost {tuned_cost:.0f}\n")

    ts = np.arange(0.02, 0.90, 0.01)
    pr = [precision_score(y_test, p_sk >= t, zero_division=0) for t in ts]
    rc = [recall_score(y_test, p_sk >= t) for t in ts]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(ts, pr, label="Precision", color="#e07b39")
    ax.plot(ts, rc, label="Recall", color="#3b7ddd")
    ax.axvline(0.5, color="gray", ls=":", label="Default 0.50")
    ax.axvline(best_t, color="green", ls="--", label=f"Chosen {best_t:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_title("Threshold trade-off (test set, for illustration)")
    ax.legend()
    save(fig, "threshold_tradeoff.png")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, t, title in [(axes[0], 0.5, "Threshold 0.50"), (axes[1], best_t, f"Threshold {best_t:.2f}")]:
        cm = confusion_matrix(y_test, (p_sk >= t).astype(int))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1], ["Pred: repay", "Pred: default"])
        ax.set_yticks([0, 1], ["Actual: repay", "Actual: default"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_title(title)
    save(fig, "confusion_matrices.png")

    # 7) L1 vs L2 regularisation ---------------------------------------------
    C_STRONG = 0.02
    with warnings.catch_warnings():   # 'penalty=' is being renamed in newest scikit-learn
        warnings.simplefilter("ignore")
        l2m = LogisticRegression(penalty="l2", C=C_STRONG, max_iter=2000).fit(Xtr, y_train)
        l1m = LogisticRegression(penalty="l1", C=C_STRONG, solver="liblinear", max_iter=2000).fit(Xtr, y_train)
    reg = pd.DataFrame({"feature": names, "plain": sk.coef_[0],
                        "L2 (C=0.02)": l2m.coef_[0], "L1 (C=0.02)": l1m.coef_[0]})
    reg.round(4).to_csv(REPORTS / "l1_vs_l2_coefficients.csv", index=False)
    n_zero = int((l1m.coef_[0] == 0).sum())
    zeroed = [f for f, c in zip(names, l1m.coef_[0]) if c == 0]
    print(f"L1 (C={C_STRONG}) set {n_zero} of {len(names)} coefficients to exactly 0: {zeroed}\n")

    r = reg.set_index("feature").iloc[::-1]
    ypos = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.barh(ypos - 0.27, r["plain"], 0.27, label="Plain (C=1)", color="#bbbbbb")
    ax.barh(ypos, r["L2 (C=0.02)"], 0.27, label="L2, strong", color="#3b7ddd")
    ax.barh(ypos + 0.27, r["L1 (C=0.02)"], 0.27, label="L1, strong", color="#e07b39")
    ax.set_yticks(ypos, r.index)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Coefficient")
    ax.set_title("L2 shrinks coefficients; L1 can set them exactly to zero", fontsize=10)
    ax.legend(loc="lower right")
    save(fig, "l1_vs_l2.png")

    # 8) Save everything ------------------------------------------------------
    final = Pipeline([("prep", prep), ("model", sk)])
    joblib.dump({"pipeline": final, "threshold": best_t,
                 "num_cols": num_cols, "cat_cols": cat_cols}, MODELS / "loan_model.joblib")

    summary = {
        "rows": int(len(df)), "default_rate": float(y.mean()),
        "majority_class_accuracy_test": float(majority_acc),
        "scratch_vs_sklearn": {"max_coef_gap": max_coef_gap, "max_probability_gap": max_proba_gap},
        "test_threshold_0.50": {"scratch": res_scratch, "sklearn": res_sk},
        "cost_assumption": {"false_negative": COST_FN, "false_positive": COST_FP},
        "test_tuned_threshold": tuned,
        "business_cost_test": {"threshold_0.50": base_cost, "tuned": tuned_cost},
        "l1_zeroed_features": zeroed,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"Saved charts/tables to {REPORTS} and the model to {MODELS / 'loan_model.joblib'}")


if __name__ == "__main__":
    main()
