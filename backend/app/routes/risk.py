"""API routes for traffic locations and risk assessment."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db import get_db
from app.db.models import Location as DBLocation
from app.models.location import Location, RiskDetail, RiskSummary
from app.services.risk_engine import calculate_risk_score

router = APIRouter(tags=["Traffic & Risk"])


def _db_to_pydantic(db_loc: DBLocation) -> Location:
    """Convert a database Location model instance to a Pydantic Location model."""
    return Location(
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
    )


@router.get(
    "/locations",
    response_model=List[Location],
    summary="Get all demo traffic locations",
    description="Returns the full list of simulated city traffic monitoring locations.",
)
def get_locations(db: Session = Depends(get_db)) -> List[Location]:
    """Retrieve all simulated demo traffic locations from the database."""
    db_locations = db.query(DBLocation).order_by(DBLocation.id).all()
    return [_db_to_pydantic(loc) for loc in db_locations]


@router.get(
    "/risk",
    response_model=List[RiskSummary],
    summary="Get risk overview for all locations",
    description="Returns calculated risk scores and risk levels for all traffic locations.",
)
def get_risk_overview(db: Session = Depends(get_db)) -> List[RiskSummary]:
    """Retrieve summarized risk assessments for all locations."""
    db_locations = db.query(DBLocation).order_by(DBLocation.id).all()
    summaries: List[RiskSummary] = []

    for db_loc in db_locations:
        loc = _db_to_pydantic(db_loc)
        risk_result = calculate_risk_score(loc)
        summaries.append(
            RiskSummary(
                id=loc.id,
                name=loc.name,
                latitude=loc.latitude,
                longitude=loc.longitude,
                coordinate_source=loc.coordinate_source,
                risk_score=risk_result["risk_score"],
                risk_level=risk_result["risk_level"],
                police_officers=loc.police_officers,
            )
        )

    return summaries


@router.get(
    "/risk/{location_id}",
    response_model=RiskDetail,
    summary="Get detailed risk assessment for a specific location",
    description="Returns comprehensive location data, calculated risk score, risk level, congestion, and factor breakdown.",
)
def get_location_risk(location_id: int, db: Session = Depends(get_db)) -> RiskDetail:
    """Retrieve detailed risk calculation breakdown for a single location by ID."""
    db_loc = db.query(DBLocation).filter(DBLocation.id == location_id).first()
    if not db_loc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Location with ID {location_id} not found",
        )

    loc = _db_to_pydantic(db_loc)
    risk_result = calculate_risk_score(loc)

    return RiskDetail(
        id=loc.id,
        name=loc.name,
        latitude=loc.latitude,
        longitude=loc.longitude,
        coordinate_source=loc.coordinate_source,
        traffic_speed=loc.traffic_speed,
        free_flow_speed=loc.free_flow_speed,
        traffic_volume=loc.traffic_volume,
        incident_frequency=loc.incident_frequency,
        accident_history=loc.accident_history,
        road_factor=loc.road_factor,
        population_factor=loc.population_factor,
        police_officers=loc.police_officers,
        congestion=risk_result["congestion"],
        risk_score=risk_result["risk_score"],
        risk_level=risk_result["risk_level"],
        contributing_factors=risk_result["contributing_factors"],
    )
