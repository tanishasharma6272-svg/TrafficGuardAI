"""API routes for TrafficGuard AI Police Deployment Optimizer."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import Location as DBLocation
from app.models.deployment import (
    DeploymentRecommendationResponse,
    DeploymentRequest,
)
from app.services.deployment_optimizer import (
    LocationRiskNode,
    get_deployment_optimizer,
)
from app.services.risk_model_service import get_risk_model_service

router = APIRouter(prefix="/api/deployment", tags=["Police Deployment Optimizer"])


@router.post(
    "/recommend",
    response_model=DeploymentRecommendationResponse,
    summary="Compute optimal police patrol unit allocation",
    description=(
        "Optimizes police unit deployment across high-risk traffic corridors using a "
        "transparent greedy maximum-coverage algorithm on ML risk outputs. "
        "Guarantees deterministic recommendations without mutating database state."
    ),
)
def recommend_deployment(
    payload: DeploymentRequest,
    db: Session = Depends(get_db),
) -> DeploymentRecommendationResponse:
    """Execute police deployment optimization for given units, patrol radius, and risk threshold."""
    # 1. Fetch all monitored traffic locations from PostgreSQL
    db_locations = db.query(DBLocation).order_by(DBLocation.id).all()

    # 2. Run trained ML model inference to obtain pure risk predictions
    ml_service = get_risk_model_service()
    ml_summaries = ml_service.predict_all_locations(db_locations)

    # 3. Decouple ML predictions into pure location risk nodes for the optimizer
    location_nodes = [
        LocationRiskNode(
            id=item.id,
            name=item.name,
            latitude=item.latitude,
            longitude=item.longitude,
            risk_score=item.risk_score,
            risk_level=item.risk_level,
        )
        for item in ml_summaries
    ]

    # 4. Execute deterministic greedy coverage optimization
    optimizer = get_deployment_optimizer()
    return optimizer.optimize_deployment(
        locations=location_nodes,
        available_units=payload.available_units,
        coverage_radius_km=payload.coverage_radius_km,
        min_risk_level=payload.min_risk_level,
    )
