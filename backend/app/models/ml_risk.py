"""Pydantic models for ML risk prediction API responses."""

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class MLRiskSummary(BaseModel):
    """Summarized ML risk assessment for a monitored traffic location."""

    id: int = Field(..., description="Unique database location identifier")
    name: str = Field(..., description="Traffic intersection / corridor name")
    latitude: float = Field(..., description="WGS84 latitude coordinate")
    longitude: float = Field(..., description="WGS84 longitude coordinate")
    coordinate_source: str = Field(..., description="Geospatial survey source")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="ML predicted risk score in [0, 100]")
    risk_level: str = Field(..., description="Categorical risk level (Low, Medium, High, Critical)")
    police_officers: int = Field(..., ge=0, description="Stationed police personnel count")
    model_type: str = Field(..., description="Type of trained regression model used for inference")
    training_data_mode: str = Field(
        default="SYNTHETIC_DEVELOPMENT",
        description="Data provenance mode of the training dataset",
    )


class MLRiskDetail(MLRiskSummary):
    """Comprehensive ML risk assessment with telemetry, derived factors, and model metadata."""

    traffic_speed: float = Field(..., ge=0.0, description="Observed traffic speed (km/h)")
    free_flow_speed: float = Field(..., gt=0.0, description="Statutory design speed limit (km/h)")
    traffic_volume: int = Field(..., ge=0, description="Observed vehicular volume (veh/day)")
    incident_frequency: float = Field(..., ge=0.0, description="Incident / breakdown rate")
    accident_history: float = Field(..., ge=0.0, description="Historical collision frequency index")
    road_factor: float = Field(..., ge=0.0, le=1.0, description="Infrastructure hazard index [0, 1]")
    population_factor: float = Field(..., ge=0.0, le=1.0, description="Pedestrian density friction index [0, 1]")

    # Kinematic & Engineered Features
    congestion_ratio: float = Field(..., ge=0.0, le=1.0, description="Speed loss delay ratio [0, 1]")
    speed_deficit: float = Field(..., ge=0.0, description="Speed deficit below free-flow (km/h)")
    traffic_pressure_composite: float = Field(..., ge=0.0, le=1.0, description="Volume-population pressure composite")

    # Explicitly labeled non-SHAP contributing factors
    contributing_factors: List[str] = Field(
        default_factory=list,
        description="Derived physical and kinematic hazard factors (NOT SHAP values)",
    )
    factor_attribution_method: str = Field(
        default="DERIVED_HEURISTIC_INDICATORS (NOT SHAP)",
        description="Attribution method used to compute contributing factors",
    )
    model_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Training provenance, feature count, and model configuration",
    )
