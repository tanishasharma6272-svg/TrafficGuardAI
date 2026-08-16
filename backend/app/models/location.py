"""Data models for traffic locations and risk assessment."""

from typing import Optional
from pydantic import BaseModel, Field


class Location(BaseModel):
    """Represents a traffic monitoring location with current traffic and risk metrics."""

    id: int = Field(..., description="Unique identifier for the location")
    name: str = Field(..., description="Name or description of the location")
    latitude: float = Field(..., description="Geographic latitude coordinate")
    longitude: float = Field(..., description="Geographic longitude coordinate")
    coordinate_source: str = Field(
        ...,
        description="Primary geospatial source or reference for the coordinates",
    )
    traffic_speed: float = Field(..., description="Current observed traffic speed (e.g., km/h) [DEMO DATA]")
    free_flow_speed: float = Field(..., description="Uncongested free flow speed limit (e.g., km/h) [DEMO DATA]")
    traffic_volume: int = Field(..., description="Estimated vehicle count over observation period [DEMO DATA]")
    incident_frequency: float = Field(..., description="Frequency of traffic incidents (scale 0.0 - 10.0) [DEMO DATA]")
    accident_history: float = Field(..., description="Historical accident severity/rate (scale 0.0 - 10.0) [DEMO DATA]")
    road_factor: float = Field(..., description="Road geometry and infrastructure hazard index (scale 0.0 - 1.0) [DEMO DATA]")
    population_factor: float = Field(..., description="Surrounding population and pedestrian density index (scale 0.0 - 1.0) [DEMO DATA]")
    police_officers: int = Field(..., description="Number of police officers currently assigned [DEMO DATA]")


class RiskSummary(BaseModel):
    """Brief risk summary for map and list views."""

    id: int
    name: str
    latitude: float
    longitude: float
    coordinate_source: str
    risk_score: float
    risk_level: str
    police_officers: int


class ContributingFactors(BaseModel):
    """Detailed breakdown of weighted risk calculation components."""

    congestion_component: float = Field(..., description="Weighted congestion contribution (max 30.0)")
    incident_frequency_component: float = Field(..., description="Weighted incident frequency contribution (max 25.0)")
    accident_history_component: float = Field(..., description="Weighted accident history contribution (max 20.0)")
    road_factor_component: float = Field(..., description="Weighted road factor contribution (max 15.0)")
    traffic_population_component: float = Field(..., description="Weighted traffic/population contribution (max 10.0)")


class RiskDetail(BaseModel):
    """Comprehensive risk information for a single location."""

    id: int
    name: str
    latitude: float
    longitude: float
    coordinate_source: str
    traffic_speed: float
    free_flow_speed: float
    traffic_volume: int
    incident_frequency: float
    accident_history: float
    road_factor: float
    population_factor: float
    police_officers: int
    congestion: float = Field(..., description="Calculated congestion ratio (0.0 to 1.0)")
    risk_score: float = Field(..., description="Composite risk score (0.0 to 100.0)")
    risk_level: str = Field(..., description="Categorical risk level (Low, Medium, High, Critical)")
    contributing_factors: ContributingFactors
