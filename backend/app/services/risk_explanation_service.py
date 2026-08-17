"""Service coordinating PostgreSQL location ingestion and SHAP explainability generation."""

from typing import Any, Dict, List, Optional
from app.db.models import Location as DBLocation
from app.ml.shap_explainer import MLShapExplainer
from app.models.ml_explanation import FeatureAttribution, MLRiskExplanation
from app.services.feature_engineering import extract_features, to_numerical_feature_dict
from app.services.risk_model_service import get_risk_model_service
from app.services.risk_thresholds import classify_risk_score
from app.services.traffic_normalizer import normalize_record


class RiskExplanationService:
    """Service providing SHAP feature attributions for monitored PostgreSQL locations."""

    def __init__(self) -> None:
        self.model_service = get_risk_model_service()
        self.explainer = MLShapExplainer(
            model=self.model_service.model,
            model_type=self.model_service.model_type,
            feature_names=self.model_service.feature_names,
        )

    def explain_location(self, db_loc: DBLocation) -> MLRiskExplanation:
        """Generate complete SHAP explanation for a single PostgreSQL Location record.

        Args:
            db_loc: SQLAlchemy Location model instance from PostgreSQL.

        Returns:
            MLRiskExplanation: Structured explanation payload with ranked attributions.
        """
        # 1. Normalize record preserving exact database values
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

        # 3. Extract exact 18 numerical predictors matching model training schema
        feat_dict = to_numerical_feature_dict(fv, include_police=False)
        feature_row = [feat_dict[name] for name in self.model_service.feature_names]

        # 4. Compute SHAP attributions and raw unconstrained model prediction
        base_val, raw_pred, raw_attributions = self.explainer.explain_sample(feature_row)

        # 5. Convert raw attributions to Pydantic FeatureAttribution schemas
        feature_attributions: List[FeatureAttribution] = [
            FeatureAttribution(
                feature_name=item["feature_name"],
                feature_value=item["feature_value"],
                shap_value=item["shap_value"],
                direction=item["direction"],
                human_label=item["human_label"],
            )
            for item in raw_attributions
        ]

        # 6. Extract top positive (risk-increasing) and top negative (risk-mitigating) contributors
        top_positive = [fa for fa in feature_attributions if fa.direction == "positive"][:3]
        top_negative = [fa for fa in feature_attributions if fa.direction == "negative"][:3]

        # 7. Compute API bounded risk score and canonical risk level
        bounded_score = round(min(max(raw_pred, 0.0), 100.0), 2)
        risk_lvl = classify_risk_score(bounded_score)

        return MLRiskExplanation(
            location_id=db_loc.id,
            name=db_loc.name,
            latitude=db_loc.latitude,
            longitude=db_loc.longitude,
            coordinate_source=db_loc.coordinate_source,
            risk_score=bounded_score,
            raw_prediction=round(raw_pred, 2),
            risk_level=risk_lvl,
            base_value=round(base_val, 2),
            model_type=self.model_service.model_type,
            training_data_mode=self.model_service.training_data_mode,
            explanation_method="SHAP",
            feature_attributions=feature_attributions,
            top_positive_contributors=top_positive,
            top_negative_contributors=top_negative,
            shap_disclaimer=(
                "SHAP values quantify model feature attribution relative to the training baseline "
                "expectation and do not represent empirical real-world causality."
            ),
        )


# Global singleton instance
_GLOBAL_EXPLANATION_SERVICE: Optional[RiskExplanationService] = None


def get_risk_explanation_service(force_reload: bool = False) -> RiskExplanationService:
    """Retrieve singleton RiskExplanationService instance, initializing once on first call."""
    global _GLOBAL_EXPLANATION_SERVICE
    if _GLOBAL_EXPLANATION_SERVICE is None or force_reload:
        _GLOBAL_EXPLANATION_SERVICE = RiskExplanationService()
    return _GLOBAL_EXPLANATION_SERVICE
