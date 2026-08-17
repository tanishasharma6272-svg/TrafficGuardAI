"""Dedicated ML Risk Prediction API routes for TrafficGuard AI."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import Location as DBLocation
from app.models.ml_risk import MLRiskDetail, MLRiskSummary
from app.services.risk_model_service import get_risk_model_service

router = APIRouter(prefix="/api/ml", tags=["ML Risk Assessment"])


@router.get(
    "/risk",
    response_model=List[MLRiskSummary],
    summary="Get ML-predicted risk overview for all monitored locations",
    description="Loads all monitored locations from PostgreSQL and runs trained ML model inference.",
)
def get_ml_risk_overview(db: Session = Depends(get_db)) -> List[MLRiskSummary]:
    """Retrieve ML risk predictions for all monitored traffic locations."""
    db_locations = db.query(DBLocation).order_by(DBLocation.id).all()
    service = get_risk_model_service()
    return service.predict_all_locations(db_locations)


@router.get(
    "/risk/{location_id}",
    response_model=MLRiskDetail,
    summary="Get detailed ML risk assessment for a specific location",
    description="Runs trained ML model inference on the specified PostgreSQL location and returns telemetry and derived factors.",
)
def get_ml_location_risk(
    location_id: int, db: Session = Depends(get_db)
) -> MLRiskDetail:
    """Retrieve comprehensive ML risk breakdown for a single location by ID."""
    db_loc = db.query(DBLocation).filter(DBLocation.id == location_id).first()
    if not db_loc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location with ID {location_id} not found in PostgreSQL database",
        )

    service = get_risk_model_service()
    return service.predict_location(db_loc)
