"""Run end-to-end fraud detection pipeline on LendingClub accepted dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from src.preprocessor import FeatureEngineer, DataPreprocessor
from src.trainer import ModelTrainer
from src.evaluator import ModelEvaluator

RAW_ACCEPTED_PATH = Path("data/raw/accepted_2007_to_2018Q4.csv")
PROCESSED_PATH = Path("data/processed/loan_applications_processed.csv")
REPORTS_DIR = Path("models")
EVAL_JSON_PATH = REPORTS_DIR / "evaluation_results.json"


FRAUD_STATUSES = {
    "Charged Off",
    "Default",
    "Late (31-120 days)",
    "Late (16-30 days)",
    "Does not meet the credit policy. Status:Charged Off",
}


def parse_emp_length(value: str | float | int | None) -> float | None:
    """Parse employment length to numeric years.

    Args:
        value: Raw employment length string (e.g., "10+ years", "< 1 year")

    Returns:
        Parsed years as float or None if unknown
    """
    if value is None or pd.isna(value):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip().lower()
    if value in {"n/a", "na", "none"}:
        return None
    if value.startswith("<"):
        return 0.5
    if value.startswith("10+"):
        return 10.0

    digits = "".join(ch for ch in value if ch.isdigit())
    return float(digits) if digits else None


def load_and_process_raw(
    raw_path: Path,
    output_path: Path,
    sample_size: int | None = 200000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Load LendingClub accepted dataset and create processed dataset.

    Args:
        raw_path: Path to accepted dataset CSV
        output_path: Path to save processed dataset
        sample_size: Optional sample size for speed and memory efficiency
        random_state: Random seed

    Returns:
        Processed DataFrame
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {raw_path}")

    usecols = [
        "loan_amnt",
        "annual_inc",
        "fico_range_low",
        "fico_range_high",
        "emp_length",
        "dti",
        "loan_status",
    ]

    df = pd.read_csv(raw_path, usecols=usecols, low_memory=False)

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)

    # Parse fields
    df["employment_years"] = df["emp_length"].apply(parse_emp_length)
    df["credit_score"] = (df["fico_range_low"] + df["fico_range_high"]) / 2
    df["fraud_label"] = df["loan_status"].apply(lambda x: 1 if x in FRAUD_STATUSES else 0)

    processed = df.rename(
        columns={
            "annual_inc": "income",
            "loan_amnt": "loan_amount",
        }
    )

    processed = processed[[
        "income",
        "loan_amount",
        "credit_score",
        "employment_years",
        "dti",
        "fraud_label",
    ]]

    # Drop rows with missing required values
    processed = processed.dropna(subset=["income", "loan_amount", "credit_score", "employment_years"])

    # Save processed dataset
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(output_path, index=False)

    return processed


def train_and_evaluate(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Any]]:
    """Train models and evaluate on hold-out test set.

    Args:
        df: Processed dataset

    Returns:
        Tuple of evaluation metrics and run metadata
    """
    X = df.drop("fraud_label", axis=1)
    y = df["fraud_label"]

    # Feature engineering
    X_engineered = FeatureEngineer.engineer_features(X)

    # Preprocess (scale/encode)
    preprocessor = DataPreprocessor()
    X_processed = preprocessor.fit_transform(X_engineered)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train.values, y_train.values)

    evaluator = ModelEvaluator()
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        y_pred = model.predict(X_test.values)
        y_pred_proba = model.predict_proba(X_test.values)[:, 1]
        results[name] = evaluator.evaluate(y_test.values, y_pred, y_pred_proba)

    metadata = {
        "total_samples": int(len(df)),
        "fraud_rate": float(df["fraud_label"].mean()),
        "features_used": list(X_processed.columns),
    }

    return results, metadata


def save_evaluation(results: Dict[str, Dict[str, float]], metadata: Dict[str, Any]) -> None:
    """Save evaluation results to JSON.

    Args:
        results: Evaluation metrics
        metadata: Run metadata
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": metadata,
        "results": results,
    }
    with EVAL_JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    """Run the complete pipeline."""
    print("Processing raw dataset...")
    processed = load_and_process_raw(RAW_ACCEPTED_PATH, PROCESSED_PATH)
    print(f"Processed dataset saved to: {PROCESSED_PATH}")

    print("Training and evaluating models...")
    results, metadata = train_and_evaluate(processed)

    save_evaluation(results, metadata)
    print(f"Evaluation results saved to: {EVAL_JSON_PATH}")


if __name__ == "__main__":
    main()
