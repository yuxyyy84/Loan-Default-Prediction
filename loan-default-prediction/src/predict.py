"""Score loan applicants with the saved model.

Usage:
  python src/predict.py                      # scores the first 5 rows of data/loan_data.csv
  python src/predict.py path/to/new.csv      # scores your own file (same columns as training)

Run  python src/train.py  first.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "loan_model.joblib"
DEFAULT_CSV = ROOT / "data" / "loan_data.csv"


def main():
    if not MODEL_PATH.exists():
        raise SystemExit("Model not found. Run:  python src/train.py")
    art = joblib.load(MODEL_PATH)
    features = art["num_cols"] + art["cat_cols"]

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    df = pd.read_csv(path)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns the model needs: {missing}")
    if path == DEFAULT_CSV:
        df = df.head(5)

    proba = art["pipeline"].predict_proba(df[features])[:, 1]
    for i, p in enumerate(proba, 1):
        decision = "REVIEW / HIGH RISK" if p >= art["threshold"] else "APPROVE (low risk)"
        extra = f"  (actual: {'default' if df['default'].iloc[i-1] == 1 else 'repaid'})" if "default" in df else ""
        print(f"Applicant {i}: default probability = {p:.1%}  ->  {decision}{extra}")
    print(f"(decision threshold = {art['threshold']:.2f})")


if __name__ == "__main__":
    main()
