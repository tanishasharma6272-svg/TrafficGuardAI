"""Model-agnostic ML risk inference service for TrafficGuard AI.

Loads the selected trained model artifact once from disk and provides pure,
bounded risk score predictions and standardized risk level categorizations.
Integrates directly with the configured TrafficProvider (TomTom or Demo).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from sqlalchemy.orm import Session

from app.db.models import Location as DBLocation
from app.models.ml_risk import MLRiskDetail, MLRiskSummary
from app.providers import ProviderFetchError, ProviderStatus, RawTrafficRecord, TrafficProvider, get_traffic_provider
from app.services.feature_engineering import extract_features, to_numerical_feature_dict
from app.services.risk_thresholds import classify_risk_score
from app.services.traffic_normalizer import normalize_record

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "models"
DEFAULT_METADATA_PATH = DEFAULT_MODEL_DIR / "metadata.json"


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
        congestion_ratio: float,
        speed_deficit: float,
        incident_frequency: float,
        accident_history: float,
        road_factor: float,
        traffic_pressure_composite: float,
    ) -> List[str]:
        """Derive readable kinematic traffic hazard indicators (non-SHAP heuristic factors)."""
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

    def predict_raw_record(
        self,
        raw_record: RawTrafficRecord,
        provider_status: Optional[ProviderStatus] = None,
    ) -> MLRiskDetail:
        """Execute full ML inference pipeline on an authoritative RawTrafficRecord.

        Args:
            raw_record: The authoritative RawTrafficRecord from the active TrafficProvider.
            provider_status: Optional typed operational health status of the provider.

        Returns:
            MLRiskDetail: Comprehensive predicted risk detail.
        """
        # 1. Normalize record using strict validation rules
        normalized = normalize_record(raw_record)

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
            congestion_ratio=fv.congestion_ratio,
            speed_deficit=fv.speed_deficit,
            incident_frequency=fv.incident_frequency,
            accident_history=fv.accident_history,
            road_factor=fv.road_factor,
            traffic_pressure_composite=fv.traffic_pressure_composite,
        )

        inc_count = raw_record.raw_metadata.get("current_incident_count")
        acc_count = raw_record.raw_metadata.get("current_accident_count")
        jam_count = raw_record.raw_metadata.get("current_jam_count")
        closure_count = raw_record.raw_metadata.get("current_road_closure_count")
        roadworks_count = raw_record.raw_metadata.get("current_roadworks_count")
        breakdown_count = raw_record.raw_metadata.get("current_broken_down_vehicle_count")
        inc_provider = raw_record.raw_metadata.get("incident_provider")
        inc_state = raw_record.raw_metadata.get("incident_provider_state")
        inc_timestamp = raw_record.raw_metadata.get("incident_snapshot_timestamp")

        metadata_dict: Dict[str, Any] = {
            "model_type": self.model_type,
            "training_data_mode": self.training_data_mode,
            "feature_count": len(self.feature_names),
            "provider_mode": raw_record.provider_mode,
            "shap_available": False,
            "shap_notice": "SHAP feature attribution is available via dedicated /api/ml/explain endpoint.",
        }

        if inc_provider is not None:
            metadata_dict["incident_provider"] = inc_provider
        if inc_state is not None:
            metadata_dict["incident_provider_state"] = inc_state
        if inc_timestamp is not None:
            metadata_dict["incident_snapshot_timestamp"] = inc_timestamp
        if inc_count is not None:
            metadata_dict["current_incident_count"] = inc_count
        if acc_count is not None:
            metadata_dict["current_accident_count"] = acc_count
        if jam_count is not None:
            metadata_dict["current_jam_count"] = jam_count
        if closure_count is not None:
            metadata_dict["current_road_closure_count"] = closure_count
        if roadworks_count is not None:
            metadata_dict["current_roadworks_count"] = roadworks_count
        if breakdown_count is not None:
            metadata_dict["current_broken_down_vehicle_count"] = breakdown_count

        if provider_status is not None:
            metadata_dict["traffic_provider"] = provider_status.provider
            metadata_dict["traffic_provider_state"] = provider_status.aggregate_state

        if "provenance" in raw_record.raw_metadata:
            metadata_dict["provenance"] = raw_record.raw_metadata["provenance"]

        return MLRiskDetail(
            id=raw_record.location_id,
            name=raw_record.name,
            latitude=raw_record.latitude,
            longitude=raw_record.longitude,
            coordinate_source=raw_record.coordinate_source,
            traffic_speed=raw_record.traffic_speed,
            free_flow_speed=raw_record.free_flow_speed,
            traffic_volume=raw_record.traffic_volume,
            incident_frequency=raw_record.incident_frequency,
            accident_history=raw_record.accident_history,
            road_factor=raw_record.road_factor,
            population_factor=raw_record.population_factor,
            police_officers=raw_record.police_officers,
            congestion_ratio=fv.congestion_ratio,
            speed_deficit=fv.speed_deficit,
            traffic_pressure_composite=fv.traffic_pressure_composite,
            risk_score=bounded_score,
            risk_level=risk_lvl,
            model_type=self.model_type,
            training_data_mode=self.training_data_mode,
            current_incident_count=inc_count,
            current_accident_count=acc_count,
            current_jam_count=jam_count,
            current_road_closure_count=closure_count,
            current_roadworks_count=roadworks_count,
            current_broken_down_vehicle_count=breakdown_count,
            incident_provider=inc_provider,
            incident_provider_state=inc_state,
            incident_snapshot_timestamp=inc_timestamp,
            contributing_factors=factors,
            factor_attribution_method="DERIVED_HEURISTIC_INDICATORS (NOT SHAP)",
            model_metadata=metadata_dict,
        )

    def predict_location(
        self,
        db_loc: DBLocation,
        db: Optional[Session] = None,
        provider: Optional[TrafficProvider] = None,
    ) -> MLRiskDetail:
        """Execute ML inference for a single location using the configured TrafficProvider.

        Args:
            db_loc: SQLAlchemy Location model instance from PostgreSQL.
            db: Optional active database session.
            provider: Optional explicit TrafficProvider instance.

        Returns:
            MLRiskDetail: Comprehensive predicted risk detail.

        Raises:
            ProviderFetchError: If the authoritative location record could not be obtained.
        """
        active_provider = provider or get_traffic_provider(db=db)
        raw_record = active_provider.get_location_traffic_record(db_loc.id)

        if raw_record is None:
            raise ProviderFetchError(
                f"Traffic provider failed to obtain telemetry for location '{db_loc.name}' (ID: {db_loc.id}).",
                location_id=db_loc.id,
            )

        status = active_provider.get_provider_status()
        return self.predict_raw_record(raw_record, provider_status=status)

    def predict_all_locations(
        self,
        db_locations: Optional[List[DBLocation]] = None,
        db: Optional[Session] = None,
        provider: Optional[TrafficProvider] = None,
    ) -> List[MLRiskSummary]:
        """Execute ML inference across monitored locations via the configured TrafficProvider.

        Args:
            db_locations: Optional list of SQLAlchemy Location instances (ignored if provider fetches batch).
            db: Optional active database session.
            provider: Optional explicit TrafficProvider instance.

        Returns:
            List[MLRiskSummary]: List of summarized risk assessments.
        """
        active_provider = provider or get_traffic_provider(db=db)
        raw_records = active_provider.get_traffic_records()
        status = active_provider.get_provider_status()

        summaries: List[MLRiskSummary] = []
        for raw in raw_records:
            detail = self.predict_raw_record(raw, provider_status=status)
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
