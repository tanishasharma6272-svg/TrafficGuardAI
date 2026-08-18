"""API endpoint for SHAP model explainability and feature attributions."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import Location as DBLocation
from app.models.ml_explanation import MLRiskExplanation
from app.providers import ProviderConfigurationError, ProviderFetchError
from app.services.risk_explanation_service import get_risk_explanation_service

router = APIRouter(prefix="/api/ml", tags=["ML Explainability (SHAP)"])


@router.get(
    "/explain/{location_id}",
    response_model=MLRiskExplanation,
    summary="Get SHAP feature attributions for a specific location",
    description="Loads the location via active TrafficProvider, computes SHAP attributions on the live feature vector, and returns ranked feature impacts.",
)
def get_location_explanation(
    location_id: int,
    db: Session = Depends(get_db),
) -> MLRiskExplanation:
    """Retrieve structured SHAP feature attributions for a single location by ID."""
    db_loc = db.query(DBLocation).filter(DBLocation.id == location_id).first()
    if not db_loc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location with ID {location_id} not found in PostgreSQL database",
        )

    service = get_risk_explanation_service()
    try:
        return service.explain_location(db_loc, db=db)
    except (ProviderFetchError, ProviderConfigurationError) as err:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Traffic provider failed for location {location_id}: {err}",
        ) from err
