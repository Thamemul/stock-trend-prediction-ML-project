"""
train_model.py -- Stock Price Trend Prediction Training Pipeline

Trains five classifiers on the stock dataset, compares performance,
and persists the best model + scaler + metrics for the Flask app.

Models:
  1. Random Forest Classifier
  2. Gradient Boosting Classifier
  3. XGBoost Classifier
  4. Voting Ensemble (soft)
  5. Stacking Ensemble
"""

import json
import logging
import warnings
import os
import sys

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from xgboost import XGBClassifier

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("training.log", mode="w", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────
DATASET_PATH = os.environ.get("DATASET_PATH", "dataset.csv")
MODEL_PATH = os.environ.get("MODEL_PATH", "model.pkl")
SCALER_PATH = os.environ.get("SCALER_PATH", "scaler.pkl")
METRICS_PATH = os.environ.get("METRICS_PATH", "metrics.json")
TEST_SIZE = 0.2
RANDOM_STATE = 42

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_50", "RSI", "MACD", "Prev_Close",
]
TARGET_COL = "Trend"


def load_dataset(path: str) -> pd.DataFrame:
    """Load the stock dataset from CSV."""
    logger.info("=" * 60)
    logger.info("  Stock Price Trend Prediction - Training Pipeline")
    logger.info("=" * 60)

    df = pd.read_csv(path)
    logger.info(f"[DATA] Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"       Columns: {list(df.columns)}")
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing numeric values with column medians."""
    for col in FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing = df[FEATURE_COLS].isnull().sum().sum()
    logger.info(f"[FIX] Missing values found: {missing}")

    for col in FEATURE_COLS:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"       Filled '{col}' with median ({median_val:.2f})")

    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute technical indicators if the columns are missing.
    If they already exist (from the CSV), this is a no-op.
    """
    if "SMA_10" not in df.columns:
        df["SMA_10"] = df["Close"].rolling(window=10, min_periods=1).mean()
    if "SMA_50" not in df.columns:
        df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
    if "RSI" not in df.columns:
        delta = df["Close"].diff()
        gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df["RSI"] = 100 - (100 / (1 + rs))
    if "MACD" not in df.columns:
        df["MACD"] = (
            df["Close"].ewm(span=12, adjust=False).mean()
            - df["Close"].ewm(span=26, adjust=False).mean()
        )
    if "Prev_Close" not in df.columns:
        df["Prev_Close"] = df["Close"].shift(1).fillna(df["Close"].iloc[0])
    return df


def encode_target(df: pd.DataFrame) -> tuple:
    """Label-encode the Trend column (DOWN=0, UP=1)."""
    le = LabelEncoder()
    df[TARGET_COL] = le.fit_transform(df[TARGET_COL])
    logger.info(
        f"[LABEL] Target classes: "
        f"{dict(zip(le.classes_, le.transform(le.classes_)))}"
    )
    return df, le


def scale_features(X_train, X_test):
    """Fit StandardScaler on train data and transform both splits."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def build_models():
    """
    Return a dict of name -> untrained estimator.
    Includes individual models + voting + stacking ensembles.
    """
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=RANDOM_STATE
    )
    gb = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1, random_state=RANDOM_STATE
    )
    xgb = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        verbosity=0,
    )

    # Voting Ensemble (soft = probability averaging)
    voting = VotingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("xgb", xgb)],
        voting="soft",
    )

    # Stacking Ensemble with Logistic Regression as meta-learner
    stacking = StackingClassifier(
        estimators=[("rf", rf), ("gb", gb), ("xgb", xgb)],
        final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        cv=5,
        stack_method="predict_proba",
    )

    return {
        "Random Forest": rf,
        "Gradient Boosting": gb,
        "XGBoost": xgb,
        "Voting Ensemble": voting,
        "Stacking Ensemble": stacking,
    }


def evaluate_model(model, X_test, y_test, label_encoder):
    """Compute all evaluation metrics for a trained model."""
    y_pred = model.predict(X_test)

    # Probability estimates for ROC AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_proba)
    else:
        roc = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(
        y_test, y_pred,
        target_names=label_encoder.classes_,
        output_dict=True,
    )

    return {
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "roc_auc": round(roc * 100, 2) if roc is not None else "N/A",
        "confusion_matrix": cm,
        "classification_report": {
            k: (
                {mk: round(mv, 4) for mk, mv in v.items()}
                if isinstance(v, dict)
                else round(v, 4)
            )
            for k, v in report.items()
        },
    }


def train_and_evaluate(models, X_train, X_test, y_train, y_test, le):
    """Train each model, evaluate, and track the best."""
    results = {}
    best_acc = 0
    best_name = ""
    best_model = None

    logger.info("")
    logger.info("-" * 60)
    logger.info("  Model Training & Evaluation")
    logger.info("-" * 60)

    for name, model in models.items():
        logger.info(f"  Training: {name} ...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, le)
        results[name] = metrics

        logger.info(f"  [OK] {name}")
        logger.info(f"       Accuracy  : {metrics['accuracy']}%")
        logger.info(f"       Precision : {metrics['precision']}%")
        logger.info(f"       Recall    : {metrics['recall']}%")
        logger.info(f"       F1 Score  : {metrics['f1_score']}%")
        logger.info(f"       ROC AUC   : {metrics['roc_auc']}%")
        logger.info(f"       Confusion : {metrics['confusion_matrix']}")
        logger.info("")

        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]
            best_name = name
            best_model = model

    return results, best_name, best_acc, best_model


def save_artifacts(best_model, scaler, le, best_name, best_acc, results):
    """Persist model, scaler, and metrics to disk."""
    # Save model
    joblib.dump(best_model, MODEL_PATH)
    logger.info(f"[BEST] Best Model : {best_name} ({best_acc}%)")
    logger.info(f"[SAVE] Model      : {MODEL_PATH}")

    # Save scaler
    joblib.dump(scaler, SCALER_PATH)
    logger.info(f"[SAVE] Scaler     : {SCALER_PATH}")

    # Save label encoder alongside for inference
    joblib.dump(le, "label_encoder.pkl")
    logger.info("[SAVE] Encoder    : label_encoder.pkl")

    # Save metrics JSON
    metrics_output = {
        "best_model": best_name,
        "best_accuracy": best_acc,
        "feature_columns": FEATURE_COLS,
        "models": results,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)
    logger.info(f"[SAVE] Metrics    : {METRICS_PATH}")


# ── Main entry point ──────────────────────────────────────────
def main():
    """Execute the full training pipeline."""
    # 1. Load
    df = load_dataset(DATASET_PATH)

    # 2. Feature engineering (recompute if missing)
    df = feature_engineering(df)

    # 3. Handle missing values
    df = handle_missing_values(df)

    # 4. Encode target
    df, le = encode_target(df)

    # 5. Split
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    # 6. Scale
    X_train_s, X_test_s, scaler = scale_features(X_train, X_test)

    # 7. Build models
    models = build_models()

    # 8. Train & evaluate
    results, best_name, best_acc, best_model = train_and_evaluate(
        models, X_train_s, X_test_s, y_train, y_test, le
    )

    # 9. Save everything
    save_artifacts(best_model, scaler, le, best_name, best_acc, results)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
