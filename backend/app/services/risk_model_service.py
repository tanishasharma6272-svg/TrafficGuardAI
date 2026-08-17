"""Model-agnostic ML risk inference service for TrafficGuard AI.

Loads the selected trained model artifact once from disk and provides pure,
bounded risk score predictions and standardized risk level categorizations.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.db.models import Location as DBLocation
from app.models.ml_risk import MLRiskDetail, MLRiskSummary
from app.services.feature_engineering import extract_features, to_numerical_feature_dict
from app.services.risk_thresholds import classify_risk_score
from app.services.traffic_normalizer import normalize_record

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "metadata.json"
DEFAULT_XGB_PATH = DEFAULT_MODEL_DIR / "xgboost_risk_model.json"
DEFAULT_BASELINE_PATH = DEFAULT_MODEL_DIR / "baseline_model.joblib"


class RiskModelService:
    """Model-agnostic singleton inference service for traffic risk prediction."""

    def __init__(self, model_dir: Optional[Path] = None) -> None:
        self.model_dir = model_dir or DEFAULT_MODEL_DIR
        self.metadata_path = self.model_dir / "metadata.json"
        self.model: Optional[Any] = None
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.model_type: str = "Unknown"
        self.training_data_mode: str = "SYNTHETIC_DEVELOPMENT"
        self._load_model_artifact()

    def _load_model_artifact(self) -> None:
        """Load selected model artifact and metadata dynamically from disk.

        Raises:
            RuntimeError: If model metadata or artifact files are missing or unreadable.
        """
        if not self.metadata_path.exists():
            raise RuntimeError(
                f"ML model metadata not found at '{self.metadata_path}'. "
                "The ML training pipeline must be executed before starting inference service."
            )

        try:
            with open(self.metadata_path, mode="r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to parse ML model metadata JSON: {e}") from e

        self.model_type = self.metadata.get("selected_model_type", "Unknown")
        self.training_data_mode = self.metadata.get(
            "training_data_mode", "SYNTHETIC_DEVELOPMENT"
        )
        self.feature_names = self.metadata.get("feature_names", [])

        if not self.feature_names:
            raise RuntimeError("ML model metadata does not specify 'feature_names'.")

        # Dynamically load the selected model artifact
        if self.model_type == "XGBoostRegressor":
            xgb_path = self.model_dir / "xgboost_risk_model.json"
            if not xgb_path.exists():
                raise RuntimeError(f"XGBoost model artifact not found at '{xgb_path}'.")
            from app.ml.xgboost_model import XGBoostRiskModel
            self.model = XGBoostRiskModel.load(xgb_path)
        else:
            # Baseline Ridge / Scikit-learn model
            base_path = self.model_dir / "baseline_model.joblib"
            if not base_path.exists():
                raise RuntimeError(f"Baseline model artifact not found at '{base_path}'.")
            from app.ml.baseline_model import BaselineRiskModel
            self.model = BaselineRiskModel.load(base_path)

    def _derive_contributing_factors(
        self,
        traffic_speed: float,
        free_flow_speed: float,
        congestion_ratio: float,
        speed_deficit: float,
        incident_frequency: float,
        accident_history: float,
        road_factor: float,
        traffic_pressure_composite: float,
    ) -> List[str]:
        """Derive readable kinematic traffic hazard indicators.

        NOTE: These are deterministic heuristic physical breakdowns and are
        EXPLICITLY NOT SHAP feature attributions.
        """
        factors: List[str] = []

        if congestion_ratio >= 0.40 or speed_deficit >= 15.0:
            pct_loss = round(congestion_ratio * 100.0, 1)
            factors.append(
                f"Kinematic Congestion Delay: {pct_loss}% speed reduction ({speed_deficit:.1f} km/h deficit)"
            )

        if incident_frequency >= 3.0:
            factors.append(
                f"Elevated Incident Propensity: {incident_frequency:.1f} breakdowns/month"
            )

        if accident_history >= 3.0:
            factors.append(
                f"Historical Collision Hazard: {accident_history:.1f} severe collisions/year"
            )

        if road_factor >= 0.50:
            factors.append(
                f"Infrastructure Road Friction: {road_factor:.2f} geometric hazard rating"
            )

        if traffic_pressure_composite >= 0.50:
            factors.append(
                f"Traffic & Demographic Pressure: {traffic_pressure_composite:.2f} volume-pedestrian index"
            )

        if not factors:
            factors.append("Nominal baseline traffic conditions across monitored corridor")

        return factors

    def predict_location(self, db_loc: DBLocation) -> MLRiskDetail:
        """Execute full ML inference pipeline on a single PostgreSQL Location record.

        Args:
            db_loc: SQLAlchemy Location model instance from PostgreSQL.

        Returns:
            MLRiskDetail: Comprehensive predicted risk detail.
        """
        # 1. Normalize record using strict validation rules (preserving exact source values)
        normalized = normalize_record({
            "location_id": db_loc.id,
            "name": db_loc.name,
            "latitude": db_loc.latitude,
            "longitude": db_loc.longitude,
            "coordinate_source": db_loc.coordinate_source,
            "traffic_speed": db_loc.traffic_speed,
            "free_flow_speed": db_loc.free_flow_speed,
            "traffic_volume": db_loc.traffic_volume,
            "incident_frequency": db_loc.incident_frequency,
            "accident_history": db_loc.accident_history,
            "road_factor": db_loc.road_factor,
            "population_factor": db_loc.population_factor,
            "police_officers": db_loc.police_officers,
            "data_mode": "DEMO",
        })

        # 2. Extract feature vector through standard feature engineering pipeline
        fv = extract_features(normalized)

        # 3. Extract exact 18 numerical predictors matching training schema
        feat_dict = to_numerical_feature_dict(fv, include_police=False)
        feature_row = [feat_dict[name] for name in self.feature_names]

        # 4. Predict risk score using loaded model (clamped to [0.0, 100.0])
        model = self.model
        if model is None:
            raise RuntimeError(
                "ML risk model is not loaded. Ensure the training pipeline has executed and generated model artifacts."
            )
        matrix = np.asarray([feature_row], dtype=np.float64)
        raw_pred = float(model.predict(matrix)[0])
        bounded_score = round(min(max(raw_pred, 0.0), 100.0), 2)

        # 5. Classify risk level using shared canonical classification policy
        risk_lvl = classify_risk_score(bounded_score)

        # 6. Derive heuristic kinematic contributing factors (non-SHAP)
        factors = self._derive_contributing_factors(
            traffic_speed=fv.traffic_speed,
            free_flow_speed=fv.free_flow_speed,
            congestion_ratio=fv.congestion_ratio,
            speed_deficit=fv.speed_deficit,
            incident_frequency=fv.incident_frequency,
            accident_history=fv.accident_history,
            road_factor=fv.road_factor,
            traffic_pressure_composite=fv.traffic_pressure_composite,
        )

        return MLRiskDetail(
            id=db_loc.id,
            name=db_loc.name,
            latitude=db_loc.latitude,
            longitude=db_loc.longitude,
            coordinate_source=db_loc.coordinate_source,
            traffic_speed=db_loc.traffic_speed,
            free_flow_speed=db_loc.free_flow_speed,
            traffic_volume=db_loc.traffic_volume,
            incident_frequency=db_loc.incident_frequency,
            accident_history=db_loc.accident_history,
            road_factor=db_loc.road_factor,
            population_factor=db_loc.population_factor,
            police_officers=db_loc.police_officers,
            congestion_ratio=fv.congestion_ratio,
            speed_deficit=fv.speed_deficit,
            traffic_pressure_composite=fv.traffic_pressure_composite,
            risk_score=bounded_score,
            risk_level=risk_lvl,
            model_type=self.model_type,
            training_data_mode=self.training_data_mode,
            contributing_factors=factors,
            factor_attribution_method="DERIVED_HEURISTIC_INDICATORS (NOT SHAP)",
            model_metadata={
                "model_type": self.model_type,
                "training_data_mode": self.training_data_mode,
                "feature_count": len(self.feature_names),
                "shap_available": False,
                "shap_notice": "SHAP feature attribution is unavailable in this endpoint and will be provided via dedicated SHAP explainability service.",
            },
        )

    def predict_all_locations(self, db_locations: List[DBLocation]) -> List[MLRiskSummary]:
        """Execute ML inference across a list of PostgreSQL Location records.

        Args:
            db_locations: List of SQLAlchemy Location model instances.

        Returns:
            List[MLRiskSummary]: List of summarized risk assessments.
        """
        summaries: List[MLRiskSummary] = []
        for loc in db_locations:
            detail = self.predict_location(loc)
            summaries.append(
                MLRiskSummary(
                    id=detail.id,
                    name=detail.name,
                    latitude=detail.latitude,
                    longitude=detail.longitude,
                    coordinate_source=detail.coordinate_source,
                    risk_score=detail.risk_score,
                    risk_level=detail.risk_level,
                    police_officers=detail.police_officers,
                    model_type=detail.model_type,
                    training_data_mode=detail.training_data_mode,
                )
            )
        return summaries


# Global cached singleton instance
_GLOBAL_RISK_SERVICE: Optional[RiskModelService] = None


def get_risk_model_service(force_reload: bool = False) -> RiskModelService:
    """Retrieve the singleton RiskModelService instance, initializing once on first call."""
    global _GLOBAL_RISK_SERVICE
    if _GLOBAL_RISK_SERVICE is None or force_reload:
        _GLOBAL_RISK_SERVICE = RiskModelService()
    return _GLOBAL_RISK_SERVICE
