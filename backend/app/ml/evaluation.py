"""Evaluation metrics, model comparison, and benchmarking utilities for TrafficGuard AI."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    split_name: str = "Validation",
) -> Dict[str, Any]:
    """Compute regression evaluation metrics (MAE, RMSE, R2, Max Error).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.
        split_name: Name of evaluation fold (e.g. 'Train', 'Validation', 'Test').

    Returns:
        Dict[str, Any]: Dictionary of computed metric scores and split metadata.
    """
    y_t = np.asarray(y_true, dtype=np.float64)
    y_p = np.asarray(y_pred, dtype=np.float64)

    mae = mean_absolute_error(y_t, y_p)
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    r2 = r2_score(y_t, y_p)
    max_err = float(np.max(np.abs(y_t - y_p)))
    mean_residual = float(np.mean(y_p - y_t))

    return {
        "split": split_name,
        "sample_count": len(y_t),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "max_error": round(max_err, 4),
        "mean_residual": round(mean_residual, 4),
    }


def evaluate_model(
    model: Any,
    X: Union[List[List[float]], np.ndarray],
    y: Union[List[float], np.ndarray],
    split_name: str = "Validation",
) -> Dict[str, Any]:
    """Generate predictions using model and compute evaluation metrics.

    Args:
        model: Fitted model instance supporting .predict(X).
        X: Feature matrix.
        y: True risk scores.
        split_name: Evaluation split label.

    Returns:
        Dict[str, Any]: Metric dictionary.
    """
    y_pred = model.predict(X)
    return compute_metrics(y_true=y, y_pred=y_pred, split_name=split_name)


def compare_models(
    baseline_metrics: Dict[str, Any],
    xgb_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """Compare baseline vs XGBoost validation performance and select the superior model.

    Decision Criterion:
        1. Lower Validation RMSE (primary metric).
        2. Higher Validation R2 (secondary metric).

    Args:
        baseline_metrics: Validation metrics dict for baseline model.
        xgb_metrics: Validation metrics dict for XGBoost model.

    Returns:
        Dict[str, Any]: Comparison summary and winner selection.
    """
    base_rmse = float(baseline_metrics["rmse"])
    xgb_rmse = float(xgb_metrics["rmse"])
    base_r2 = float(baseline_metrics["r2"])
    xgb_r2 = float(xgb_metrics["r2"])

    rmse_improvement = base_rmse - xgb_rmse
    rmse_pct_improvement = (rmse_improvement / base_rmse) * 100.0 if base_rmse > 0 else 0.0

    # Decision rule: lowest validation RMSE (primary), highest validation R2 (secondary)
    if base_rmse < xgb_rmse:
        winner = "BaselineRidge"
    elif xgb_rmse < base_rmse:
        winner = "XGBoostRegressor"
    else:
        # Tie breaker on secondary metric: highest validation R2
        winner = "BaselineRidge" if base_r2 >= xgb_r2 else "XGBoostRegressor"

    selected_rmse = base_rmse if winner == "BaselineRidge" else xgb_rmse
    alt_rmse = xgb_rmse if winner == "BaselineRidge" else base_rmse

    return {
        "baseline_rmse": base_rmse,
        "xgb_rmse": xgb_rmse,
        "baseline_mae": float(baseline_metrics["mae"]),
        "xgb_mae": float(xgb_metrics["mae"]),
        "baseline_r2": base_r2,
        "xgb_r2": xgb_r2,
        "rmse_improvement": round(rmse_improvement, 4),
        "rmse_pct_improvement": round(rmse_pct_improvement, 2),
        "selected_model": winner,
        "selection_rationale": (
            f"{winner} achieved superior validation performance with RMSE of "
            f"{selected_rmse:.4f} vs {alt_rmse:.4f}."
        ),
    }


def format_comparison_table(
    baseline_metrics: Dict[str, Any],
    xgb_metrics: Dict[str, Any],
    test_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Format formatted text summary table of evaluation metrics."""
    b_split = str(baseline_metrics.get("split", "Val"))
    b_mae = float(baseline_metrics["mae"])
    b_rmse = float(baseline_metrics["rmse"])
    b_r2 = float(baseline_metrics["r2"])
    b_max = float(baseline_metrics["max_error"])

    x_split = str(xgb_metrics.get("split", "Val"))
    x_mae = float(xgb_metrics["mae"])
    x_rmse = float(xgb_metrics["rmse"])
    x_r2 = float(xgb_metrics["r2"])
    x_max = float(xgb_metrics["max_error"])

    lines = [
        "| Model | Split | MAE | RMSE | R2 Score | Max Error |",
        "|---|---|---|---|---|---|",
        f"| Baseline (Ridge) | {b_split} | {b_mae:.4f} | {b_rmse:.4f} | {b_r2:.4f} | {b_max:.4f} |",
        f"| XGBoost Regressor | {x_split} | {x_mae:.4f} | {x_rmse:.4f} | {x_r2:.4f} | {x_max:.4f} |",
    ]
    if test_metrics is not None:
        t_split = str(test_metrics.get("split", "Test"))
        t_mae = float(test_metrics["mae"])
        t_rmse = float(test_metrics["rmse"])
        t_r2 = float(test_metrics["r2"])
        t_max = float(test_metrics["max_error"])
        lines.append(
            f"| Selected Model | {t_split} | {t_mae:.4f} | {t_rmse:.4f} | {t_r2:.4f} | {t_max:.4f} |"
        )
    return "\n".join(lines)
