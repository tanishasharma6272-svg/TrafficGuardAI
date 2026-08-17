"""SHAP explainer engine supporting both regularized linear and gradient boosted tree models."""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import shap

# Ensure backend root is in sys.path when executed directly
_backend_root = Path(__file__).resolve().parent.parent.parent
if str(_backend_root) not in sys.path:
    sys.path.insert(0, str(_backend_root))

from app.services.feature_engineering import MODEL_FEATURE_NAMES

# Descriptive human-readable labels for dashboard presentation
FEATURE_HUMAN_LABELS: Dict[str, str] = {
    "traffic_speed": "Observed Traffic Speed (km/h)",
    "free_flow_speed": "Statutory Design Speed (km/h)",
    "congestion_ratio": "Kinematic Speed Loss Ratio",
    "speed_deficit": "Velocity Drop Below Design Limit (km/h)",
    "speed_ratio": "Speed Realization Fraction",
    "traffic_volume": "Vehicular Traffic Volume (veh/day)",
    "volume_capacity_ratio": "Roadway Capacity Saturation Ratio",
    "incident_frequency": "Monthly Incident & Breakdown Rate",
    "incident_index": "Normalized Incident Propensity Index",
    "accident_history": "Historical Annual Collision Rate",
    "accident_severity": "Normalized Collision Hazard Index",
    "road_factor": "Infrastructure & Geometric Friction Index",
    "population_factor": "Pedestrian & Urban Activity Density",
    "traffic_pressure_composite": "Volume-Pedestrian Friction Composite",
    "hour_of_day": "Diurnal Time of Day (Hour)",
    "day_of_week": "Day of Week",
    "is_weekend": "Weekend Traffic Pattern Indicator",
    "is_peak_hour": "Rush Hour Period Indicator",
}


def _safe_extract_scalar(value: Any) -> float:
    """Safely convert a scalar, NumPy scalar, or array-like value to a plain Python float."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return float(value.item())
        except (ValueError, TypeError):
            pass
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) > 0:
            first_item = value[0]
            if first_item is not None:
                if hasattr(first_item, "item") and callable(getattr(first_item, "item")):
                    try:
                        return float(first_item.item())
                    except (ValueError, TypeError):
                        pass
                if isinstance(first_item, (int, float)):
                    return float(first_item)
    try:
        return float(value)
    except (ValueError, TypeError):
        return 0.0


def _extract_shap_array(shap_res: Any) -> np.ndarray:
    """Extract a 1D NumPy array of SHAP values from a SHAP Explanation or ndarray result.

    Raises:
        RuntimeError: If the SHAP result is None, missing, or empty.
    """
    if shap_res is None:
        raise RuntimeError("SHAP explainer returned None unexpectedly.")

    # If it is a SHAP Explanation object with a .values attribute
    if hasattr(shap_res, "values"):
        vals = getattr(shap_res, "values")
        if vals is None:
            raise RuntimeError("SHAP Explanation object has None values.")
        arr = np.asarray(vals)
    else:
        arr = np.asarray(shap_res)

    if arr.ndim == 2:
        if arr.shape[0] == 0:
            raise RuntimeError("SHAP explainer returned an empty 2D array.")
        return np.asarray(arr[0], dtype=np.float64)
    elif arr.ndim == 1:
        return np.asarray(arr, dtype=np.float64)
    else:
        reshaped = arr.reshape(-1, arr.shape[-1])
        if reshaped.shape[0] == 0:
            raise RuntimeError("SHAP explainer returned an empty multi-dimensional array.")
        return np.asarray(reshaped[0], dtype=np.float64)


class MLShapExplainer:
    """Model-agnostic SHAP attribution explainer for traffic risk predictions."""

    def __init__(
        self,
        model: Any,
        model_type: str,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
    ) -> None:
        self.model: Any = model
        self.model_type: str = model_type
        self.feature_names: List[str] = feature_names or list(MODEL_FEATURE_NAMES)
        self.background_data: Optional[np.ndarray] = background_data
        self.explainer: Optional[Any] = None
        self.explainer_type: str = "Unknown"
        self.scaler: Optional[Any] = None
        self.regressor: Optional[Any] = None
        self.base_value: float = 0.0
        self._initialize_explainer()

    def _get_default_background(self) -> np.ndarray:
        """Load background dataset from training partition for masker initialization."""
        if self.background_data is not None:
            return self.background_data

        try:
            from app.ml.training_data import load_training_dataset, split_train_val_test_chronological
            from app.ml.model_io import extract_features_and_target

            _, all_records = load_training_dataset()
            train_records, _, _ = split_train_val_test_chronological(all_records)
            X_train, _, _ = extract_features_and_target(train_records, self.feature_names)
            sample_size = min(100, len(X_train))
            self.background_data = X_train[:sample_size]
            return self.background_data
        except Exception:
            return np.zeros((1, len(self.feature_names)), dtype=np.float64)

    def _initialize_explainer(self) -> None:
        """Initialize the mathematically appropriate SHAP explainer for the active model."""
        model = self.model
        if model is None:
            raise RuntimeError("Cannot initialize SHAP explainer with uninitialized model.")

        if self.model_type == "XGBoostRegressor":
            xgb_inner = getattr(model, "model", model)
            explainer_inst = shap.TreeExplainer(xgb_inner)
            self.explainer = explainer_inst
            self.explainer_type = "TreeExplainer"
            self.base_value = _safe_extract_scalar(explainer_inst.expected_value)
        else:
            # BaselineRidge or linear pipeline
            bg = self._get_default_background()
            if hasattr(model, "pipeline"):
                pipeline = model.pipeline
                scaler = pipeline.named_steps["scaler"] if hasattr(pipeline, "named_steps") else None
                regressor = pipeline.named_steps["regressor"] if hasattr(pipeline, "named_steps") else pipeline
                bg_scaled = scaler.transform(bg) if scaler is not None else bg
                self.scaler = scaler
                self.regressor = regressor
                explainer_inst = shap.LinearExplainer(regressor, bg_scaled)
                self.explainer = explainer_inst
                self.explainer_type = "LinearExplainer"
                self.base_value = _safe_extract_scalar(explainer_inst.expected_value)
            else:
                self.scaler = None
                self.regressor = model
                explainer_inst = shap.LinearExplainer(model, bg)
                self.explainer = explainer_inst
                self.explainer_type = "LinearExplainer"
                self.base_value = _safe_extract_scalar(explainer_inst.expected_value)

    def explain_sample(
        self,
        feature_row: Union[List[float], np.ndarray],
    ) -> Tuple[float, float, List[Dict[str, Any]]]:
        """Compute exact SHAP attributions for a single observation.

        Args:
            feature_row: 18-element numerical feature vector.

        Returns:
            Tuple[float, float, List[Dict[str, Any]]]:
                - base_value: Model expectation E[f(X)]
                - raw_prediction: Unclamped additive prediction explained by SHAP
                - attributions: List of 18 feature attributions sorted by |shap_value| descending.
        """
        explainer = self.explainer
        if explainer is None:
            raise RuntimeError("SHAP explainer was not properly initialized.")

        arr = np.asarray(feature_row, dtype=np.float64).reshape(1, -1)

        if self.model_type == "XGBoostRegressor":
            shap_res = explainer(arr)
            shap_values = _extract_shap_array(shap_res)
            raw_prediction = self.base_value + float(np.sum(shap_values))
        else:
            scaler = self.scaler
            arr_scaled = scaler.transform(arr) if scaler is not None else arr
            shap_res = explainer(arr_scaled)
            shap_values = _extract_shap_array(shap_res)
            raw_prediction = self.base_value + float(np.sum(shap_values))

        attributions: List[Dict[str, Any]] = []
        for name, val, phi in zip(self.feature_names, arr[0], shap_values):
            phi_float = round(_safe_extract_scalar(phi), 4)
            val_float = round(_safe_extract_scalar(val), 4)
            attributions.append({
                "feature_name": name,
                "feature_value": val_float,
                "shap_value": phi_float,
                "direction": "positive" if phi_float >= 0.0 else "negative",
                "human_label": FEATURE_HUMAN_LABELS.get(name, name.replace("_", " ").title()),
            })

        # Sort strictly by absolute SHAP magnitude descending
        attributions.sort(key=lambda item: abs(item["shap_value"]), reverse=True)

        return self.base_value, raw_prediction, attributions
