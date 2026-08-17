"""Pydantic data models and schemas for TrafficGuard AI Police Deployment Optimizer.

Defines strict input request validation, individual deployment placement schemas,
and comprehensive coverage metrics.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class DeploymentRequest(BaseModel):
    """Input parameters for police patrol deployment optimization request."""

    available_units: int = Field(
        ...,
        gt=0,
        description="Number of currently deployable police units (must be > 0)",
        examples=[3],
    )
    coverage_radius_km: float = Field(
        ...,
        gt=0.0,
        description="Effective patrol coverage radius in kilometers (must be > 0.0)",
        examples=[2.0],
    )
    min_risk_level: Optional[Literal["High", "Critical"]] = Field(
        default=None,
        description="Minimum risk level threshold for optimization ('High' optimizes Critical + High, 'Critical' optimizes Critical only)",
        examples=["High"],
    )


class SelectedDeploymentUnit(BaseModel):
    """Specific patrol unit placement recommendation and its covered nodes."""

    rank: int = Field(
        ...,
        ge=1,
        description="Greedy priority rank (1 = highest marginal risk coverage)",
    )
    location_id: int = Field(
        ...,
        description="Unique database identifier of the selected deployment location",
    )
    location_name: str = Field(
        ...,
        description="Name of the selected deployment location / corridor",
    )
    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="ML predicted risk score at this deployment location",
    )
    risk_level: str = Field(
        ...,
        description="Categorical risk level at this deployment location (e.g. Critical, High)",
    )
    latitude: float = Field(
        ...,
        description="WGS84 latitude coordinate of the deployment location",
    )
    longitude: float = Field(
        ...,
        description="WGS84 longitude coordinate of the deployment location",
    )
    covered_location_ids: List[int] = Field(
        ...,
        description="Unique database identifiers of all eligible locations covered by this unit",
    )
    covered_location_count: int = Field(
        ...,
        ge=0,
        description="Number of eligible locations covered by this unit",
    )
    covered_risk_score: float = Field(
        ...,
        ge=0.0,
        description="Aggregate risk score of all eligible locations covered by this unit",
    )


class BaselineMetrics(BaseModel):
    """Network-wide baseline metrics for eligible high-risk locations prior to deployment."""

    eligible_high_risk_locations: int = Field(
        ...,
        ge=0,
        description="Total count of monitored locations meeting the risk eligibility threshold",
    )
    total_eligible_risk_score: float = Field(
        ...,
        ge=0.0,
        description="Sum of ML risk scores across all eligible locations",
    )


class OptimizedMetrics(BaseModel):
    """Geographic and risk coverage metrics achieved by the recommended police deployment."""

    covered_locations: int = Field(
        ...,
        ge=0,
        description="Total distinct eligible locations within patrol radius of deployed units",
    )
    covered_risk_score: float = Field(
        ...,
        ge=0.0,
        description="Total distinct risk score covered across all selected patrol units",
    )
    risk_coverage_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of eligible risk score covered: (covered_risk_score / total_eligible_risk_score) * 100",
    )
    uncovered_risk_score: float = Field(
        ...,
        ge=0.0,
        description="Remaining uncovered eligible risk score: total_eligible_risk_score - covered_risk_score",
    )
    uncovered_risk_percent: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Percentage of eligible risk score remaining uncovered: (uncovered_risk_score / total_eligible_risk_score) * 100",
    )


class DeploymentRecommendationResponse(BaseModel):
    """Complete deployment recommendation payload returned by the optimization endpoint."""

    available_units: int = Field(
        ...,
        description="Requested number of available deployable units",
    )
    coverage_radius_km: float = Field(
        ...,
        description="Requested unit coverage radius in kilometers",
    )
    selected_units: List[SelectedDeploymentUnit] = Field(
        ...,
        description="Deterministic ordered list of optimal police deployment placements",
    )
    baseline_metrics: BaselineMetrics = Field(
        ...,
        description="Baseline risk metrics prior to unit allocation",
    )
    optimized_metrics: OptimizedMetrics = Field(
        ...,
        description="Post-deployment coverage metrics achieved by the allocated units",
    )
    algorithm: str = Field(
        default="GREEDY_COVERAGE_OPTIMIZER",
        description="Algorithmic optimizer used to compute placement recommendations",
    )
