"""XGBoost regression risk prediction model for TrafficGuard AI."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import xgboost as xgb

from app.services.feature_engineering import MODEL_FEATURE_NAMES


class XGBoostRiskModel:
    """Gradient boosted decision tree regression model for traffic risk estimation."""

    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_child_weight: int = 3,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        early_stopping_rounds: int = 20,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_names = feature_names or list(MODEL_FEATURE_NAMES)
        self.model_type = "XGBoostRegressor"

        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=self.random_state,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.is_fitted = False
        self.best_iteration: Optional[int] = None

    def fit(
        self,
        X_train: Union[List[List[float]], np.ndarray],
        y_train: Union[List[float], np.ndarray],
        X_val: Optional[Union[List[List[float]], np.ndarray]] = None,
        y_val: Optional[Union[List[float], np.ndarray]] = None,
    ) -> "XGBoostRiskModel":
        """Fit the XGBoost regressor with early stopping on validation data.

        Args:
            X_train: Training feature matrix (N x 18).
            y_train: Training target risk scores (N).
            X_val: Optional validation feature matrix for early stopping.
            y_val: Optional validation target scores.

        Returns:
            XGBoostRiskModel: Fitted instance.
        """
        X_tr = np.asarray(X_train, dtype=np.float64)
        y_tr = np.asarray(y_train, dtype=np.float64)

        if X_tr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: Expected {len(self.feature_names)} features, got {X_tr.shape[1]}"
            )

        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None
        if X_val is not None and y_val is not None:
            X_v = np.asarray(X_val, dtype=np.float64)
            y_v = np.asarray(y_val, dtype=np.float64)
            eval_set = [(X_tr, y_tr), (X_v, y_v)]
        else:
            # If no validation set provided, temporarily disable early stopping
            self.model.set_params(early_stopping_rounds=None)

        self.model.fit(
            X_tr,
            y_tr,
            eval_set=eval_set,
            verbose=False,
        )
        self.is_fitted = True
        self.best_iteration = getattr(self.model, "best_iteration", None)
        return self

    def predict(self, X: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """Predict bounded risk scores in [0.0, 100.0].

        Args:
            X: Feature matrix (N x 18).

        Returns:
            np.ndarray: Predicted risk scores clipped strictly to [0.0, 100.0].
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() or load() before predict().")

        X_arr = np.asarray(X, dtype=np.float64)
        raw_preds = self.model.predict(X_arr)
        return np.clip(raw_preds, 0.0, 100.0)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance map sorted by importance score descending."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")

        importances = self.model.feature_importances_
        return {
            name: float(round(imp, 6))
            for name, imp in sorted(
                zip(self.feature_names, importances),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    def save(self, filepath: Union[str, Path]) -> Path:
        """Save trained XGBoost model artifact (JSON format) and hyperparameters."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "XGBoostRiskModel":
        """Load trained XGBoost model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"XGBoost model artifact not found at {path}")

        instance = cls()
        instance.model.load_model(str(path))
        instance.is_fitted = True
        return instance
