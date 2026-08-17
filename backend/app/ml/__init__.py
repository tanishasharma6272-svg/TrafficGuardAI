"""Machine learning, synthetic training data, and risk inference package for TrafficGuard AI."""

from app.ml.baseline_model import BaselineRiskModel
from app.ml.evaluation import (
    compare_models,
    compute_metrics,
    evaluate_model,
    format_comparison_table,
)
from app.ml.generate_training_data import (
    RANDOM_SEED,
    generate_synthetic_dataset,
    save_dataset_to_disk,
)
from app.ml.model_io import (
    extract_features_and_target,
    load_model_artifacts,
    predict_risk,
    save_model_artifacts,
    train_and_evaluate_pipeline,
)
from app.ml.training_data import (
    SyntheticTrainingRecord,
    load_training_dataset,
    split_train_val_test_chronological,
    verify_dataset_integrity,
)
from app.ml.shap_explainer import FEATURE_HUMAN_LABELS, MLShapExplainer
from app.ml.xgboost_model import XGBoostRiskModel

__all__ = [
    "SyntheticTrainingRecord",
    "load_training_dataset",
    "split_train_val_test_chronological",
    "verify_dataset_integrity",
    "generate_synthetic_dataset",
    "save_dataset_to_disk",
    "RANDOM_SEED",
    "BaselineRiskModel",
    "XGBoostRiskModel",
    "compute_metrics",
    "evaluate_model",
    "compare_models",
    "format_comparison_table",
    "save_model_artifacts",
    "load_model_artifacts",
    "predict_risk",
    "train_and_evaluate_pipeline",
    "extract_features_and_target",
    "MLShapExplainer",
    "FEATURE_HUMAN_LABELS",
]
