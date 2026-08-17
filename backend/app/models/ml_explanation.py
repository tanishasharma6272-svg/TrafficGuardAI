"""Pydantic schemas for SHAP feature attribution and model explainability responses."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class FeatureAttribution(BaseModel):
    """Individual feature attribution item computed via SHAP."""

    feature_name: str = Field(..., description="Standardized feature identifier")
    feature_value: float = Field(..., description="Observed/derived feature numerical value")
    shap_value: float = Field(..., description="Additive SHAP contribution to the raw prediction")
    direction: str = Field(..., description="'positive' (risk-increasing) or 'negative' (risk-mitigating)")
    human_label: str = Field(..., description="Descriptive human-readable label for dashboard display")


class MLRiskExplanation(BaseModel):
    """Complete SHAP explainability response for a monitored traffic location."""

    location_id: int = Field(..., description="Unique database location identifier")
    name: str = Field(..., description="Traffic intersection / corridor name")
    latitude: float = Field(..., description="WGS84 latitude coordinate")
    longitude: float = Field(..., description="WGS84 longitude coordinate")
    coordinate_source: str = Field(..., description="Geospatial survey source")

    # Risk Metrics
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Bounded predicted risk score in [0, 100]")
    raw_prediction: float = Field(..., description="Raw additive model prediction explained by SHAP (pre-clamping)")
    risk_level: str = Field(..., description="Categorical risk level (Low, Medium, High, Critical)")
    base_value: float = Field(..., description="Model baseline expected value E[f(X)]")

    # Provenance & Explainability Metadata
    model_type: str = Field(..., description="Type of trained regression model used for inference")
    training_data_mode: str = Field(
        default="SYNTHETIC_DEVELOPMENT",
        description="Data provenance mode of the training dataset",
    )
    explanation_method: str = Field(
        default="SHAP",
        description="Explainability framework used (SHAP)",
    )

    # Ranked Feature Attributions
    feature_attributions: List[FeatureAttribution] = Field(
        default_factory=list,
        description="All 18 numerical feature attributions sorted by absolute SHAP magnitude descending",
    )
    top_positive_contributors: List[FeatureAttribution] = Field(
        default_factory=list,
        description="Top risk-increasing features with positive SHAP attributions",
    )
    top_negative_contributors: List[FeatureAttribution] = Field(
        default_factory=list,
        description="Top risk-mitigating features with negative SHAP attributions",
    )

    shap_disclaimer: str = Field(
        default=(
            "SHAP values quantify model feature attribution relative to the training baseline "
            "expectation and do not represent empirical real-world causality."
        ),
        description="Scientific caveat regarding model attribution vs real-world causality",
    )
