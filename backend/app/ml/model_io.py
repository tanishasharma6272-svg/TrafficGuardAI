import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

# Ensure backend root is in sys.path when executed directly
_backend_root = Path(__file__).resolve().parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.ml.baseline_model import BaselineRiskModel
from app.ml.evaluation import compare_models, evaluate_model, format_comparison_table
from app.ml.training_data import load_training_dataset, split_train_val_test_chronological
from app.ml.xgboost_model import XGBoostRiskModel
from app.services.feature_engineering import MODEL_FEATURE_NAMES, TrafficFeatureVector, to_numerical_feature_dict

# Default storage directory for development model artifacts
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
DEFAULT_XGB_FILENAME = "xgboost_risk_model.json"
DEFAULT_BASELINE_FILENAME = "baseline_model.joblib"
DEFAULT_METADATA_FILENAME = "metadata.json"

# In-memory cached model instance for low-latency inference
_CACHED_MODEL: Optional[Any] = None
_CACHED_METADATA: Optional[Dict[str, Any]] = None


def extract_features_and_target(
    records: List[Dict[str, Any]],
    feature_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract standard 18-column feature matrix and target vector from records.

    Args:
        records: List of dataset dictionary rows.
        feature_names: Optional explicit list of feature names. Defaults to MODEL_FEATURE_NAMES.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[str]]: (X matrix, y vector, feature_names)
    """
    feats = feature_names or list(MODEL_FEATURE_NAMES)
    X_rows: List[List[float]] = []
    y_vals: List[float] = []

    for r in records:
        X_rows.append([float(r[name]) for name in feats])
        y_vals.append(float(r["risk_score"]))

    return np.asarray(X_rows, dtype=np.float64), np.asarray(y_vals, dtype=np.float64), feats


def save_model_artifacts(
    model: Any,
    metadata: Dict[str, Any],
    model_dir: Optional[Path] = None,
    baseline_model: Optional[BaselineRiskModel] = None,
    xgb_model: Optional[XGBoostRiskModel] = None,
) -> Dict[str, Path]:
    """Save trained model artifacts and comprehensive metadata JSON to disk.

    Args:
        model: Selected primary model instance (e.g. BaselineRiskModel or XGBoostRiskModel).
        metadata: Metadata summary dictionary.
        model_dir: Target directory. Defaults to backend/data/models/.
        baseline_model: Optional baseline model instance to serialize.
        xgb_model: Optional XGBoost model instance to serialize.

    Returns:
        Dict[str, Path]: Map of artifact names to written file paths.
    """
    out_dir = model_dir or DEFAULT_MODEL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: Dict[str, Path] = {}

    # 1. Save primary / selected model
    if isinstance(model, XGBoostRiskModel):
        xgb_path = out_dir / DEFAULT_XGB_FILENAME
        model.save(xgb_path)
        saved_paths["primary_model"] = xgb_path
    elif isinstance(model, BaselineRiskModel):
        base_path = out_dir / DEFAULT_BASELINE_FILENAME
        model.save(base_path)
        saved_paths["primary_model"] = base_path

    # 2. Save baseline model if provided as secondary
    if baseline_model is not None and isinstance(baseline_model, BaselineRiskModel):
        base_path = out_dir / DEFAULT_BASELINE_FILENAME
        baseline_model.save(base_path)
        saved_paths["baseline_model"] = base_path

    # 3. Save XGBoost model if provided as secondary
    if xgb_model is not None and isinstance(xgb_model, XGBoostRiskModel):
        xgb_path = out_dir / DEFAULT_XGB_FILENAME
        xgb_model.save(xgb_path)
        saved_paths["xgb_model"] = xgb_path

    # 4. Save comprehensive metadata
    meta_path = out_dir / DEFAULT_METADATA_FILENAME
    with open(meta_path, mode="w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    saved_paths["metadata"] = meta_path

    return saved_paths


def load_model_artifacts(
    model_dir: Optional[Path] = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load the trained primary model and associated metadata dynamically from disk.

    Args:
        model_dir: Directory containing saved model artifacts.

    Returns:
        Tuple[Any, Dict[str, Any]]: (Loaded model instance, Metadata dict).
    """
    global _CACHED_MODEL, _CACHED_METADATA

    in_dir = model_dir or DEFAULT_MODEL_DIR
    meta_path = in_dir / DEFAULT_METADATA_FILENAME

    if not meta_path.exists():
        raise FileNotFoundError(
            f"Model metadata not found at '{meta_path}'. Run training pipeline first."
        )

    with open(meta_path, mode="r", encoding="utf-8") as f:
        metadata = json.load(f)

    selected_type = metadata.get("selected_model_type", "BaselineRidge")

    if selected_type == "XGBoostRegressor":
        model_path = in_dir / DEFAULT_XGB_FILENAME
        model = XGBoostRiskModel.load(model_path)
    else:
        model_path = in_dir / DEFAULT_BASELINE_FILENAME
        model = BaselineRiskModel.load(model_path)

    _CACHED_MODEL = model
    _CACHED_METADATA = metadata
    return model, metadata


def predict_risk(
    feature_input: Union[TrafficFeatureVector, Dict[str, Any], List[float], np.ndarray],
    model: Optional[Any] = None,
) -> float:
    """Pure risk prediction function accepting feature vector or dict and returning score in [0, 100].

    Args:
        feature_input: TrafficFeatureVector, feature dict, or numerical 18-element vector.
        model: Optional pre-loaded model instance. If None, loads cached or disk model.

    Returns:
        float: Bounded predicted risk score in [0.0, 100.0].
    """
    global _CACHED_MODEL

    if model is None:
        if _CACHED_MODEL is None:
            _CACHED_MODEL, _ = load_model_artifacts()
        model = _CACHED_MODEL

    # Extract 18 numerical features in exact order
    if isinstance(feature_input, TrafficFeatureVector):
        feat_dict = to_numerical_feature_dict(feature_input, include_police=False)
        row = [feat_dict[name] for name in MODEL_FEATURE_NAMES]
    elif isinstance(feature_input, dict):
        row = [float(feature_input[name]) for name in MODEL_FEATURE_NAMES]
    elif isinstance(feature_input, (list, tuple, np.ndarray)):
        row = [float(v) for v in feature_input]
        if len(row) != len(MODEL_FEATURE_NAMES):
            raise ValueError(
                f"Feature vector length mismatch: expected {len(MODEL_FEATURE_NAMES)}, got {len(row)}"
            )
    else:
        raise TypeError(f"Unsupported feature_input type: {type(feature_input)}")

    matrix = np.asarray([row], dtype=np.float64)
    preds = model.predict(matrix)
    score = float(preds[0])
    return round(min(max(score, 0.0), 100.0), 2)


def train_and_evaluate_pipeline(
    dataset_csv_path: Optional[Path] = None,
    save_artifacts: bool = True,
    model_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute end-to-end training, evaluation, comparison, selection, and persistence pipeline.

    Workflow:
    1. Load synthetic development dataset.
    2. Chronologically split (70% train, 15% val, 15% test).
    3. Extract 18 standardized numerical features (excluding police_officers and ID categoricals).
    4. Fit Baseline Ridge model on Train set; evaluate on Validation set.
    5. Fit XGBoost model on Train set with early stopping on Validation set.
    6. Compare validation metrics and select superior model (lowest RMSE, highest R2).
    7. Evaluate selected model on held-out Test set.
    8. Extract feature importances and serialize model artifacts + metadata.

    Returns:
        Dict[str, Any]: Full pipeline execution results dictionary.
    """
    # 1. Load dataset and split chronologically
    _, all_records = load_training_dataset(dataset_csv_path)
    train_records, val_records, test_records = split_train_val_test_chronological(
        all_records, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
    )

    # 2. Extract 18 numerical feature matrices
    X_train, y_train, feature_names = extract_features_and_target(train_records)
    X_val, y_val, _ = extract_features_and_target(val_records)
    X_test, y_test, _ = extract_features_and_target(test_records)

    # 3. Train Baseline Model (Ridge + StandardScaler)
    baseline = BaselineRiskModel(alpha=1.0, random_state=42, feature_names=feature_names)
    baseline.fit(X_train, y_train)
    baseline_train_metrics = evaluate_model(baseline, X_train, y_train, split_name="Train")
    baseline_val_metrics = evaluate_model(baseline, X_val, y_val, split_name="Validation")

    # 4. Train XGBoost Model (with early stopping on validation fold)
    xgb_model = XGBoostRiskModel(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        early_stopping_rounds=20,
        feature_names=feature_names,
    )
    xgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    xgb_train_metrics = evaluate_model(xgb_model, X_train, y_train, split_name="Train")
    xgb_val_metrics = evaluate_model(xgb_model, X_val, y_val, split_name="Validation")

    # 5. Compare models & select winner based on Validation performance
    comparison = compare_models(baseline_val_metrics, xgb_val_metrics)
    selected_name = comparison["selected_model"]
    selected_model = baseline if selected_name == "BaselineRidge" else xgb_model

    # 6. Evaluate selected model on unseen Test set
    test_metrics = evaluate_model(selected_model, X_test, y_test, split_name="Test")

    # 7. Extract feature importances
    if hasattr(selected_model, "get_feature_importances"):
        feature_importances = selected_model.get_feature_importances()
    elif isinstance(xgb_model, XGBoostRiskModel) and xgb_model.is_fitted:
        feature_importances = xgb_model.get_feature_importances()
    else:
        feature_importances = {}

    # 8. Build comprehensive metadata dictionary
    metadata = {
        "model_name": f"TrafficGuard AI {selected_name} Risk Predictor",
        "training_data_mode": "SYNTHETIC_DEVELOPMENT",
        "disclaimer": "DEVELOPMENT ONLY. Synthetic training data. Must be replaced with real-world sensor/accident data for production.",
        "training_timestamp": datetime.now(timezone.utc).isoformat(),
        "random_seed": 42,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "target_variable": "risk_score",
        "target_bounds": [0.0, 100.0],
        "split_sizes": {
            "train_samples": len(X_train),
            "validation_samples": len(X_val),
            "test_samples": len(X_test),
            "total_samples": len(all_records),
        },
        "selected_model_type": selected_name,
        "selection_rationale": comparison["selection_rationale"],
        "metrics": {
            "baseline_train": baseline_train_metrics,
            "baseline_val": baseline_val_metrics,
            "xgboost_train": xgb_train_metrics,
            "xgboost_val": xgb_val_metrics,
            "selected_test": test_metrics,
        },
        "feature_importances": feature_importances,
        "xgboost_hyperparameters": {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 5,
            "min_child_weight": 3,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "best_iteration": xgb_model.best_iteration,
        },
    }

    # 9. Save artifacts to disk (save selected model and keep alternative available)
    saved_paths: Dict[str, Path] = {}
    if save_artifacts:
        saved_paths = save_model_artifacts(
            model=selected_model,
            metadata=metadata,
            model_dir=model_dir,
            baseline_model=baseline,
            xgb_model=xgb_model,
        )

    return {
        "baseline_model": baseline,
        "xgboost_model": xgb_model,
        "selected_model": selected_model,
        "comparison": comparison,
        "test_metrics": test_metrics,
        "metadata": metadata,
        "saved_paths": {k: str(v) for k, v in saved_paths.items()},
        "comparison_table": format_comparison_table(
            baseline_val_metrics, xgb_val_metrics, test_metrics
        ),
    }


if __name__ == "__main__":
    print("Executing end-to-end ML training and evaluation pipeline...")
    results = train_and_evaluate_pipeline(save_artifacts=True)
    print("\n--- Model Comparison Summary ---")
    print(results["comparison_table"])
    print(f"\nWinner: {results['comparison']['selected_model']}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.4f}, Test R2: {results['test_metrics']['r2']:.4f}")
    print("\nTop 5 Important Features:")
    for feat, imp in list(results["metadata"]["feature_importances"].items())[:5]:
        print(f"  - {feat}: {imp:.4f}")
