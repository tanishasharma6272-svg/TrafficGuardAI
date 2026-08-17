"""Scikit-learn regularized linear regression baseline model for TrafficGuard AI."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.services.feature_engineering import MODEL_FEATURE_NAMES


class BaselineRiskModel:
    """Standardized regularized linear regression baseline (Ridge + StandardScaler)."""

    def __init__(
        self,
        alpha: float = 1.0,
        random_state: int = 42,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.alpha = alpha
        self.random_state = random_state
        self.feature_names = feature_names or list(MODEL_FEATURE_NAMES)
        self.model_type = "BaselineRidge"

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=self.alpha, random_state=self.random_state)),
        ])
        self.is_fitted = False

    def fit(
        self,
        X_train: Union[List[List[float]], np.ndarray],
        y_train: Union[List[float], np.ndarray],
    ) -> "BaselineRiskModel":
        """Fit the regularized baseline model on training data.

        Args:
            X_train: Numerical training matrix (N x 18).
            y_train: Target risk scores (N).

        Returns:
            BaselineRiskModel: Fitted instance.
        """
        X_arr = np.asarray(X_train, dtype=np.float64)
        y_arr = np.asarray(y_train, dtype=np.float64)

        if X_arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature count mismatch: Expected {len(self.feature_names)} features, got {X_arr.shape[1]}"
            )

        self.pipeline.fit(X_arr, y_arr)
        self.is_fitted = True
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
        raw_preds = self.pipeline.predict(X_arr)
        return np.clip(raw_preds, 0.0, 100.0)

    def get_feature_importances(self) -> Dict[str, float]:
        """Return feature importance map based on normalized absolute standardized coefficients."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted.")
        regressor = self.pipeline.named_steps["regressor"]
        coefs = np.abs(regressor.coef_)
        total = np.sum(coefs)
        norm_coefs = coefs / total if total > 0 else coefs
        return {
            name: float(round(imp, 6))
            for name, imp in sorted(
                zip(self.feature_names, norm_coefs),
                key=lambda item: item[1],
                reverse=True,
            )
        }

    def save(self, filepath: Union[str, Path]) -> Path:
        """Serialize model artifact to disk via joblib."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "alpha": self.alpha,
            "random_state": self.random_state,
            "pipeline": self.pipeline,
            "is_fitted": self.is_fitted,
        }, path)
        return path

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaselineRiskModel":
        """Load serialized baseline model artifact from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Baseline model artifact not found at {path}")

        data = joblib.load(path)
        instance = cls(
            alpha=data.get("alpha", 1.0),
            random_state=data.get("random_state", 42),
            feature_names=data.get("feature_names"),
        )
        instance.pipeline = data["pipeline"]
        instance.is_fitted = data.get("is_fitted", True)
        return instance
